from pathlib import Path
import numpy as np
import pandas as pd

from estimar_exposiciones import FACTORES_DEFAULT
from backtesting import calcular_ic_serie
from optimizar_modelos import cargar_rendimientos, cargar_factores, resumir_ic, serie_ic
from ajustes import MODOS_PESO, pesos_por_ic
from backtesting_ajustado import (
    construir_insumos,
    rendimiento_esperado,
    ic_por_factor,
    encoger_betas_por_corte,
)

REGRESION = Path("salidas/regresion")
SALIDA = Path("salidas/calibracion")

FACTORES = FACTORES_DEFAULT
# meses entre re-estimaciones de beta (igual que optimizar_modelos.py)
FRECUENCIA_REESTIMACION = 12
REZAGO_FACTOR = 1
MIN_ACCIONES_IC = 10
MIN_PERIODOS = 24                # meses mínimos de IC para entrar a la comparación
ENTRENAR_CON_HISTORIAL = True    # igual que optimizar_modelos.py

# Técnica 1: ajuste por varianza del IC
# "ic_var" (media/varianza) y "ic_ratio" (media/desviación)
MODOS_IC = list(MODOS_PESO)
# meses de IC para calcular los pesos; None = expansiva
VENTANAS_IC = [24, 36, 60, None]
MIN_PERIODOS_IC = 12             # meses de IC necesarios antes de empezar a ponderar

# Técnica 2: shrinkage
KAPPAS = [0.5, 1.0, 2.0, 4.0]    # intensidad del encogimiento

# Si no hay comparacion_modelos.csv de un modelo, se usa esto:
CONFIG_DEFAULT = {"ventana_beta": None, "min_obs": 24, "ventana_factor": 12}

# (número, nombre, carpeta con su comparacion_modelos.csv, banderas de estimación)
MODELOS = [
    (1, "Modelo 1: estandarizado, sum(beta)=1", Path("salidas/backtesting"),
     {"estandarizar": True, "estandarizar_y": True, "restringir_suma": True}),
    (2, "Modelo 2: estandarizado, betas libres", Path("salidas/backtesting_sin_restriccion"),
     {"estandarizar": True, "estandarizar_y": True, "restringir_suma": False}),
    (3, "Modelo 3: sin estandarizar, sum(beta)=1", Path("salidas/backtesting_sin_estandarizar"),
     {"estandarizar": False, "estandarizar_y": False, "restringir_suma": True}),
    (4, "Modelo 4: sin estandarizar, betas libres",
     Path("salidas/backtesting_sin_restriccion_sin_estandarizar"),
     {"estandarizar": False, "estandarizar_y": False, "restringir_suma": False}),
]


def leer_config_base(carpeta: Path) -> dict | None:
    """Configuración ganadora de un modelo: la fila de su comparacion_modelos.csv
    con MAYOR IC MEDIO (no se usa el orden del archivo, que optimizar_modelos.py
    deja por IC Ratio)."""
    archivo = carpeta / "comparacion_modelos.csv"
    if not archivo.exists():
        return None
    comp = pd.read_csv(archivo)
    if "incluido_en_comparacion" in comp.columns:
        inc = comp["incluido_en_comparacion"].astype(
            str).str.strip().str.lower().isin(["true", "1"])
        if inc.any():
            comp = comp[inc]
    comp = comp.dropna(subset=["ic_medio"])
    if comp.empty:
        return None
    fila = comp.loc[comp["ic_medio"].idxmax()]
    vb = str(fila["ventana_beta_meses"]).strip().lower()
    ventana_beta = None if vb in (
        "completo", "none", "nan") else int(vb.replace("m", ""))
    return {"ventana_beta": ventana_beta,
            "min_obs": int(fila["min_obs"]),
            "ventana_factor": int(fila["ventana_factor_meses"]),
            "modelo_id": str(fila["modelo_id"])}


def interseccion_fechas(series: dict[str, pd.Series]) -> pd.DatetimeIndex:
    fechas = None
    for s in series.values():
        fechas = s.index if fechas is None else fechas.intersection(s.index)
    return fechas.sort_values()


def ordenar_por_ic_medio(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena los candidatos SOLO por IC medio (mayor primero). Agrega
    lugar_ic_medio (1 = mejor). Un empate exacto lo gana el que venía primero
    (orden estable)."""
    df = df.copy()
    inc = df["incluido"].astype(bool)
    df["lugar_ic_medio"] = df.loc[inc, "ic_medio"].rank(
        ascending=False, method="min")
    return df.sort_values("ic_medio", ascending=False, na_position="last",
                          kind="stable").reset_index(drop=True)


def tabla_metricas(series: dict[str, pd.Series], info: dict[str, dict],
                   min_periodos: int) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Métricas de IC de cada candidato sobre las fechas comunes, ordenados por IC medio."""
    incluidos = [c for c, s in series.items() if len(s) >= min_periodos]
    if not incluidos:
        incluidos = [c for c, s in series.items() if len(s) > 0]
    if not incluidos:
        raise RuntimeError(
            "Ningún candidato tiene meses de IC; revisa los insumos.")
    fechas = interseccion_fechas({c: series[c] for c in incluidos})
    if len(fechas) == 0:
        raise RuntimeError("Los candidatos no comparten ninguna fecha de IC.")

    filas = []
    for cid, s in series.items():
        incluido = cid in incluidos
        met = resumir_ic(s.loc[fechas]) if incluido else resumir_ic(s.iloc[:0])
        filas.append({"candidato": cid, **info[cid], **met,
                      "n_periodos_propio": len(s), "incluido": incluido})
    return ordenar_por_ic_medio(pd.DataFrame(filas)), fechas


def agregar_prueba_pareada(df: pd.DataFrame, series: dict[str, pd.Series],
                           fechas: pd.DatetimeIndex, ref_id: str) -> pd.DataFrame:
    """Diferencia mensual de IC contra la referencia (A) y su t-stat (pareado)."""
    df = df.copy()
    ref = series[ref_id].loc[fechas]
    ref_ratio = df.loc[df["candidato"] == ref_id, "ic_ratio"].iloc[0]
    dif, t, dif_ratio = [], [], []
    for cid, ratio in zip(df["candidato"], df["ic_ratio"]):
        if cid == ref_id:
            dif.append(np.nan)
            t.append(np.nan)
            dif_ratio.append(np.nan)
            continue
        d = series[cid].loc[fechas] - ref
        sd = d.std()
        dif.append(d.mean())
        t.append(d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.nan)
        dif_ratio.append(ratio - ref_ratio)
    df["dif_ic_medio_vs_A"] = dif
    df["dif_ic_ratio_vs_A"] = dif_ratio
    df["t_stat_dif_vs_A"] = t
    return df

# Calibración de UN modelo (A, B y C)


def calibrar_un_modelo(num, nombre, carpeta_bt, flags, rendimientos, factores_df, rendimiento_real):
    cfg = leer_config_base(carpeta_bt)
    if cfg is None:
        print(f"  AVISO: no encontré {carpeta_bt / 'comparacion_modelos.csv'}; "
              f"uso la configuración por defecto {CONFIG_DEFAULT}")
        cfg = {**CONFIG_DEFAULT, "modelo_id": "(configuración por defecto)"}
    etiqueta_vb = "completo" if cfg["ventana_beta"] is None else f"{cfg['ventana_beta']}m"
    print(f"  configuración base: beta={etiqueta_vb}, min_obs={cfg['min_obs']}, "
          f"factor={cfg['ventana_factor']}m   ({cfg['modelo_id']})")

    ins = construir_insumos(
        rendimientos, factores_df, FACTORES,
        ventana_beta=cfg["ventana_beta"], min_obs=cfg["min_obs"],
        frecuencia_meses=FRECUENCIA_REESTIMACION, ventana_factor=cfg["ventana_factor"],
        rezago_factor=REZAGO_FACTOR, **flags,
    )
    print(f"  {len(ins.cortes)} cortes de re-estimación de betas")

    candidatos: dict[str, dict] = {}

    def agregar(cid, version, parametros, er, pesos=None, lam=None):
        ic = calcular_ic_serie(er, rendimiento_real,
                               min_acciones=MIN_ACCIONES_IC)
        candidatos[cid] = {"version": version, "parametros": parametros,
                           "serie": serie_ic(ic).sort_index(), "er": er,
                           "pesos": pesos, "lam": lam}

    agregar("A_base", "A", "sin ajuste", rendimiento_esperado(ins))

    ic_fac = ic_por_factor(ins, rendimiento_real, MIN_ACCIONES_IC)
    if ic_fac.empty:
        print("  AVISO: no se pudo calcular el IC por factor; se omite la versión B.")
    else:
        for modo in MODOS_IC:
            for v in VENTANAS_IC:
                pesos = (pesos_por_ic(ic_fac, ventana=v, min_periodos=MIN_PERIODOS_IC, modo=modo)
                         .reindex(factores_df.index).fillna(1.0))
                etiqueta_v = "exp" if v is None else f"{v}m"
                agregar(f"B_{modo}_{etiqueta_v}", "B",
                        f"modo={modo}, ventana_IC={'expansiva' if v is None else str(v) + 'm'}",
                        rendimiento_esperado(ins, pesos=pesos), pesos=pesos)

    for kappa in KAPPAS:
        betas_enc, lam_df = encoger_betas_por_corte(ins, kappa)
        agregar(f"C_kappa{kappa:g}", "C", f"kappa={kappa:g}",
                rendimiento_esperado(ins, betas_por_corte=betas_enc), lam=lam_df)

    series = {c: d["serie"] for c, d in candidatos.items()}
    info = {c: {"version": d["version"], "parametros": d["parametros"]}
            for c, d in candidatos.items()}
    tabla_cand, fechas_cand = tabla_metricas(series, info, MIN_PERIODOS)
    tabla_cand.insert(0, "modelo", num)
    print(f"  {len(candidatos)} candidatos evaluados en las mismas {len(fechas_cand)} fechas "
          f"({fechas_cand[0]:%Y-%m} a {fechas_cand[-1]:%Y-%m})")

    ids = ["A_base"]
    for v in ("B", "C"):
        sub = tabla_cand[(tabla_cand["version"] == v) & tabla_cand["incluido"]]
        if len(sub):
            ids.append(sub.iloc[0]["candidato"])

    series_fin = {i: candidatos[i]["serie"] for i in ids}
    info_fin = {i: info[i] for i in ids}
    tabla_fin, fechas_fin = tabla_metricas(series_fin, info_fin, MIN_PERIODOS)
    tabla_fin = agregar_prueba_pareada(
        tabla_fin, series_fin, fechas_fin, "A_base")
    tabla_fin.insert(0, "modelo", num)
    tabla_fin["ganador_del_modelo"] = [i == 0 for i in range(len(tabla_fin))]

    carpeta = SALIDA / f"modelo{num}"
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla_cand.to_csv(carpeta / "candidatos.csv", index=False)
    tabla_fin.to_csv(carpeta / "finalistas.csv", index=False)
    for cid in ids:
        v = candidatos[cid]["version"]
        candidatos[cid]["serie"].rename("ic").to_csv(
            carpeta / f"ic_serie_{v}.csv")
        candidatos[cid]["er"].to_csv(carpeta / f"rendimiento_esperado_{v}.csv")
        if candidatos[cid]["pesos"] is not None:
            candidatos[cid]["pesos"].to_csv(carpeta / "pesos_ic_mejor_B.csv")
        if candidatos[cid]["lam"] is not None:
            candidatos[cid]["lam"].to_csv(
                carpeta / "lambda_por_corte_mejor_C.csv")

    ganador = tabla_fin.iloc[0]
    print(f"  A vs mejor B vs mejor C en {len(fechas_fin)} fechas -> ganador: "
          f"{ganador['version']} ({ganador['candidato']}), IC medio = {ganador['ic_medio']:.4f}")
    return {"cfg": cfg, "tabla_cand": tabla_cand, "tabla_fin": tabla_fin,
            "series_fin": {candidatos[i]["version"]: candidatos[i]["serie"] for i in ids},
            "info_fin": {candidatos[i]["version"]: {"candidato": i,
                                                    "parametros": candidatos[i]["parametros"]}
                         for i in ids}}


# Gráfico y definiciones
def grafico_ic_acumulado(resultados: dict, ruta: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado; se omite el gráfico)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, (num, r) in zip(axes.ravel(), resultados.items()):
        fechas = interseccion_fechas(r["series_fin"])
        for v, s in r["series_fin"].items():
            ax.plot(fechas, s.loc[fechas].cumsum(), label={
                    "A": "A base", "B": "B IC", "C": "C shrinkage"}[v])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"Modelo {num}", fontsize=10)
        ax.set_ylabel("IC acumulado")
        ax.legend(fontsize=8)
    fig.suptitle("IC acumulado: A (base), mejor B y mejor C en cada modelo")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def definiciones() -> pd.DataFrame:
    return pd.DataFrame([
        ("Versión A", "Modelo base: el ganador de su optimizar_modelos*.py, sin ajustes"),
        ("Versión B", "Ajuste por varianza del IC: E[R] = sum_k beta_k * w_k,t * factor_esperado_k. "
                      "w_k,t sale del IC pasado de cada factor: max(IC medio,0)/varianza (ic_var) o "
                      "/desviación (ic_ratio), con ventana de IC de 24/36/60 meses o expansiva; "
                      "promedio de pesos = 1; pesos 1 mientras no hay historia"),
        ("Versión C", "Shrinkage (Bayes empírico): beta_i = lambda_i*beta_i + (1-lambda_i)*beta_promedio, "
                      "lambda_i = 1/(1+kappa*ruido/señal_i). Conserva sum(beta)=1 si el modelo la tiene"),
        ("kappa", "Intensidad del shrinkage; mayor = más encogimiento (0 = modelo base)"),
        ("ventana_IC", "Meses de IC por factor que se usan para calcular los pesos de B"),
        ("Configuración base", "ventana de beta, min_obs y ventana de factor del ganador de cada modelo "
                               "(la de mayor IC medio en su comparacion_modelos.csv); B y C la comparten con A"),
        ("ic_medio / ic_std / ic_ratio / t_stat / hit_rate",
         "Mismas definiciones que optimizar_modelos.py"),
        ("Criterio de selección", "SOLO el IC medio (mayor es mejor), tanto para elegir la mejor configuración "
                                  "de cada versión como para elegir el mejor modelo de cada grupo y el global. "
                                  "IC std, IC Ratio, t-stat y hit rate se reportan pero no deciden"),
        ("lugar_ic_medio", "Lugar del candidato por IC medio entre los comparados (1 = mayor)"),
        ("dif_ic_medio_vs_A / dif_ic_ratio_vs_A",
         "Diferencia contra la versión A del mismo modelo"),
        ("t_stat_dif_vs_A", "t-stat de la diferencia mensual de IC contra A (prueba pareada); |t| > 2 sugiere "
                            "una diferencia real"),
        ("Sesgo de selección", "B y C se eligen como el mejor de su rejilla y A es una sola configuración, "
                               "así que B y C llevan ventaja; la prueba pareada y el promedio de la rejilla "
                               "(hoja Candidatos) ayudan a verlo"),
    ], columns=["concepto", "definición"])


# Programa principal
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo rendimientos y factores...")
    rendimiento_real = cargar_rendimientos(
        REGRESION / "rendimientos_miembros.csv")
    factores_df = cargar_factores(REGRESION / "factores.csv")
    if ENTRENAR_CON_HISTORIAL:
        rendimientos = cargar_rendimientos(REGRESION / "rendimientos.csv")
        print("  betas con historial completo; IC solo con miembros del Top 100")
    else:
        rendimientos = rendimiento_real
        print("  betas e IC solo con meses en el Top 100")

    resultados = {}
    for num, nombre, carpeta_bt, flags in MODELOS:
        print(f"\n{'=' * 70}\n{nombre}\n{'=' * 70}")
        resultados[num] = calibrar_un_modelo(num, nombre, carpeta_bt, flags,
                                             rendimientos, factores_df, rendimiento_real)

    resumen = pd.concat([r["tabla_fin"]
                        for r in resultados.values()], ignore_index=True)
    resumen.to_csv(SALIDA / "resumen_A_B_C_por_modelo.csv", index=False)

    series12, info12 = {}, {}
    for num, r in resultados.items():
        for v, s in r["series_fin"].items():
            cid = f"M{num}_{v}"
            series12[cid] = s
            info12[cid] = {"modelo": num, "version": v,
                           "candidato": r["info_fin"][v]["candidato"],
                           "parametros": r["info_fin"][v]["parametros"]}
    tabla12, fechas12 = tabla_metricas(series12, info12, MIN_PERIODOS)
    tabla12.insert(0, "mejor_global", [i == 0 for i in range(len(tabla12))])
    tabla12.to_csv(SALIDA / "finalistas_12_mismas_fechas.csv", index=False)
    pd.DataFrame(series12).loc[fechas12].rename_axis(
        "fecha").to_csv(SALIDA / "ic_series_finalistas.csv")
    grafico_ic_acumulado(resultados, SALIDA / "ic_acumulado_calibracion.png")

    with pd.ExcelWriter(SALIDA / "calibracion.xlsx", engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="Resumen_A_B_C", index=False)
        tabla12.to_excel(xw, sheet_name="Finalistas_12", index=False)
        for num, r in resultados.items():
            r["tabla_cand"].to_excel(
                xw, sheet_name=f"Candidatos_M{num}", index=False)
        definiciones().to_excel(xw, sheet_name="Definiciones", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 45)
    cols = ["modelo", "version", "parametros", "ic_medio", "ic_std", "ic_ratio",
            "t_stat", "n_periodos", "dif_ic_ratio_vs_A", "t_stat_dif_vs_A"]
    print(f"\n{'=' * 90}\nA (base) vs mejor B (varianza del IC) vs mejor C (shrinkage), por modelo\n{'=' * 90}")
    print(resumen[cols].round(4).to_string(index=False))

    print(f"\n{'=' * 90}\nLos {len(tabla12)} finalistas en las MISMAS {len(fechas12)} fechas "
          f"({fechas12[0]:%Y-%m} a {fechas12[-1]:%Y-%m})\n{'=' * 90}")
    print(tabla12[["mejor_global", "candidato", "modelo", "version", "ic_medio", "ic_std",
                   "ic_ratio", "lugar_ic_medio"]].round(4).to_string(index=False))

    g = tabla12.iloc[0]
    print(f"\n{'=' * 70}")
    print(
        f"MEJOR MODELO GLOBAL: Modelo {g['modelo']}, versión {g['version']}  ({g['parametros']})")
    print("  Criterio de selección: mayor IC medio")
    print(f"  IC medio = {g['ic_medio']:.4f}  <- criterio   IC std = {g['ic_std']:.4f}   "
          f"IC Ratio = {g['ic_ratio']:.4f}   t-stat = {g['t_stat']:.2f}")
    if g["version"] == "A":
        print("  Ningún ajuste superó al modelo base en este modelo.")
    print(f"{'=' * 70}")
    print(f"\nResultados completos en {SALIDA}")


if __name__ == "__main__":
    main()
