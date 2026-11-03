from pathlib import Path
import numpy as np
import pandas as pd

from optimizar_modelos import resumir_ic, serie_ic

MODELOS = [
    (1, "Modelo 1: estandarizado, sum(beta)=1", Path("salidas/backtesting")),
    (2, "Modelo 2: estandarizado, betas libres",
     Path("salidas/backtesting_sin_restriccion")),
    (3, "Modelo 3: sin estandarizar, sum(beta)=1",
     Path("salidas/backtesting_sin_estandarizar")),
    (4, "Modelo 4: sin estandarizar, betas libres", Path(
        "salidas/backtesting_sin_restriccion_sin_estandarizar")),
]
# modelo contra el que se prueban diferencias
REFERENCIA = 1
SALIDA = Path("salidas/comparacion_modelos")
COLUMNAS_METRICAS = ["ic_medio", "ic_std",
                     "ic_ratio", "t_stat", "hit_rate", "n_periodos"]


RANKS = [("rank_ic_medio", "ic_medio", False),
         ("rank_ic_std", "ic_std", True),
         ("rank_ic_ratio", "ic_ratio", False)]
ESENCIALES = ["modelo_id", "ic_medio", "ic_std", "ic_ratio"]


def _normalizar_comp(comp: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Deja comparacion_modelos.csv con todas las columnas que usa este script.
    Si el CSV se generó con una versión vieja de optimizar_modelos.py (sin
    ranking_promedio, rank_* o incluido_en_comparacion), las reconstruye con la
    misma definición que optimizar_modelos.py (lugares entre los modelos incluidos).
    El orden de las filas NO se toca: la fila 0 sigue siendo el ganador."""
    comp = comp.copy()
    faltan_esenciales = [c for c in ESENCIALES if c not in comp.columns]
    if faltan_esenciales:
        raise ValueError(f"{nombre}: comparacion_modelos.csv no tiene {faltan_esenciales}; "
                         "vuelve a correr su optimizar_modelos*.py")
    reconstruidas = []

    if "incluido_en_comparacion" not in comp.columns:
        comp["incluido_en_comparacion"] = True
        reconstruidas.append("incluido_en_comparacion")
    else:   # por si se leyó como texto
        comp["incluido_en_comparacion"] = (
            comp["incluido_en_comparacion"].astype(str).str.strip().str.lower().isin(["true", "1"]))

    for col in ("t_stat", "hit_rate", "n_periodos"):
        if col not in comp.columns:
            comp[col] = np.nan
            reconstruidas.append(col)

    inc = comp["incluido_en_comparacion"]
    for rank_col, metrica, ascendente in RANKS:
        if rank_col not in comp.columns:
            comp[rank_col] = np.nan
            comp.loc[inc, rank_col] = comp.loc[inc, metrica].rank(
                ascending=ascendente, method="min")
            reconstruidas.append(rank_col)
    if "ranking_promedio" not in comp.columns:
        comp["ranking_promedio"] = comp[[r for r, _, _ in RANKS]].mean(axis=1)
        reconstruidas.append("ranking_promedio")

    if reconstruidas:
        print(f"  AVISO: {nombre}: su comparacion_modelos.csv no traía {', '.join(reconstruidas)} "
              "(se generó con una versión anterior de optimizar_modelos*.py); se reconstruyeron. "
              "Si quieres los valores originales, vuelve a correr ese optimizar_modelos*.py.")
    return comp


def cargar_modelos():
    """{k: dict(nombre, comp, serie, ganador)} solo de los modelos con salidas."""
    cargados = {}
    for k, nombre, carpeta in MODELOS:
        f_comp, f_ic = carpeta / "comparacion_modelos.csv", carpeta / \
            "mejor_modelo_ic_serie.csv"
        if not (f_comp.exists() and f_ic.exists()):
            print(f"  AVISO: {nombre} omitido (faltan archivos en {carpeta}/)")
            continue
        comp = _normalizar_comp(pd.read_csv(f_comp), nombre)
        serie = serie_ic(pd.read_csv(f_ic, index_col=0, parse_dates=True))
        cargados[k] = {"nombre": nombre, "comp": comp, "serie": serie.sort_index(),
                       # comparacion_modelos.csv viene ordenado: fila 0 = ganador
                       "ganador": comp.iloc[0]}
    return cargados


# A) Ganador de cada modelo
def tabla_a(modelos):
    filas = []
    for k, m in modelos.items():
        g = m["ganador"]
        filas.append({"modelo": m["nombre"], "configuracion_ganadora": g["modelo_id"],
                      "ranking_promedio": g["ranking_promedio"],
                      **{c: g[c] for c in COLUMNAS_METRICAS}})
    return pd.DataFrame(filas)

# B) Ganadores en las mismas fechas (+ prueba pareada vs el modelo de referencia)


def tabla_b(modelos):
    fechas = None
    for m in modelos.values():
        fechas = m["serie"].index if fechas is None else fechas.intersection(
            m["serie"].index)
    fechas = fechas.sort_values()
    if len(fechas) < 2:
        print(
            "  AVISO: los ganadores no comparten fechas de IC; se omite la comparación B.")
        return pd.DataFrame(), fechas
    ref = modelos[REFERENCIA]["serie"].loc[fechas] if REFERENCIA in modelos else None
    filas = []
    for k, m in modelos.items():
        s = m["serie"].loc[fechas]
        fila = {"modelo": m["nombre"], "configuracion_ganadora": m["ganador"]["modelo_id"],
                **resumir_ic(s)}
        if ref is not None and k != REFERENCIA:
            d = s - ref
            sd = d.std()
            fila["dif_ic_vs_modelo_ref"] = d.mean()
            fila["t_stat_dif"] = d.mean() / (sd / np.sqrt(len(d))
                                             ) if sd > 0 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas), fechas


# C) Misma configuración en todos los modelos
def tabla_c(modelos):
    def pivote(col):
        d = {}
        for k, m in modelos.items():
            c = m["comp"]
            c = c[c["incluido_en_comparacion"].astype(
                bool)].set_index("modelo_id")[col]
            d[f"M{k}"] = c
        # solo configuraciones presentes en todos
        return pd.DataFrame(d).dropna()

    ratio, medio = pivote("ic_ratio"), pivote("ic_medio")
    if ratio.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    detalle = pd.concat(
        {"ic_ratio": ratio, "ic_medio": medio.reindex(ratio.index)}, axis=1)
    detalle.columns = [f"{a}_{b}" for a, b in detalle.columns]
    detalle.index.name = "configuracion"

    mejores = ratio.idxmax(axis=1).value_counts().reindex(
        ratio.columns).fillna(0).astype(int)
    resumen = pd.DataFrame({
        "n_configuraciones": len(ratio),
        "ic_ratio_medio_rejilla": ratio.mean(),
        "ic_ratio_mediano_rejilla": ratio.median(),
        "ic_medio_medio_rejilla": medio.mean(),
        "configs_donde_es_el_mejor": mejores,
    })
    resumen.index.name = "modelo"
    return detalle, resumen, ratio

# D) Efectos principales del diseño 2x2


def tabla_d(modelos):
    if not all(k in modelos for k in (1, 2, 3, 4)):
        print("  (la comparación D necesita los 4 modelos; se omite)")
        return pd.DataFrame()
    filas = []
    for col, etiqueta in (("ic_ratio", "IC Ratio"), ("ic_medio", "IC medio")):
        v = {}
        for k in (1, 2, 3, 4):
            c = modelos[k]["comp"]
            c = c[c["incluido_en_comparacion"].astype(
                bool)].set_index("modelo_id")[col]
            v[k] = c
        v = pd.DataFrame(v).dropna()
        if v.empty:
            continue
        m = v.mean()
        filas.append({
            "metrica (promedio de la rejilla)": etiqueta,
            "M1": m[1], "M2": m[2], "M3": m[3], "M4": m[4],
            "efecto_quitar_restriccion": ((m[2] - m[1]) + (m[4] - m[3])) / 2,
            "efecto_no_estandarizar": ((m[3] - m[1]) + (m[4] - m[2])) / 2,
            "interaccion": (m[4] - m[3]) - (m[2] - m[1]),
        })
    return pd.DataFrame(filas)


def grafico_ic_acumulado(modelos, fechas, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado; se omite el gráfico)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, m in modelos.items():
        ax.plot(fechas, m["serie"].loc[fechas].cumsum(), label=m["nombre"])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(
        "IC acumulado del modelo ganador de cada versión (mismas fechas)")
    ax.set_ylabel("suma acumulada del IC mensual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("Cargando resultados de los modelos...")
    modelos = cargar_modelos()
    if len(modelos) < 2:
        raise RuntimeError(
            "Se necesitan al menos 2 modelos con salidas para comparar.")
    print(f"  {len(modelos)} modelos: " + ", ".join(f"M{k}" for k in modelos))

    # aviso si un mismo modelo_id tiene distinto n_periodos entre versiones
    n = pd.DataFrame({f"M{k}": m["comp"].set_index("modelo_id")[
                     "n_periodos"] for k, m in modelos.items()})
    if (n.nunique(axis=1) > 1).any():
        print("  AVISO: el número de periodos de IC no es igual en todos los modelos "
              "(las fechas comunes difieren); la comparación C usa las fechas de cada backtest.")

    A = tabla_a(modelos)
    B, fechas = tabla_b(modelos)
    C_det, C_res, _ = tabla_c(modelos)
    D = tabla_d(modelos)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 45)
    print("\n" + "=" * 90 +
          "\nA) Ganador de cada modelo (su propio periodo)\n" + "=" * 90)
    print(A.round(4).to_string(index=False))
    if not B.empty:
        print("\n" + "=" * 90 + f"\nB) Ganadores en las MISMAS {len(fechas)} fechas "
              f"({fechas[0]:%Y-%m} a {fechas[-1]:%Y-%m})\n" + "=" * 90)
        print(B.round(4).to_string(index=False))
        print(f"   (dif_ic y t_stat_dif: diferencia mensual de IC vs {REFERENCIA and 'Modelo ' + str(REFERENCIA)}; "
              "|t| > 2 sugiere diferencia real)")
    if not C_res.empty:
        print("\n" + "=" * 90 +
              "\nC) Misma configuración en los modelos (resumen de la rejilla)\n" + "=" * 90)
        print(C_res.round(4).to_string())
    if not D.empty:
        print("\n" + "=" * 90 +
              "\nD) Efectos principales del diseño 2x2 (promedio de la rejilla)\n" + "=" * 90)
        print(D.round(4).to_string(index=False))

    # Guardado de resultados
    A.to_csv(SALIDA / "A_ganadores.csv", index=False)
    if not B.empty:
        B.to_csv(SALIDA / "B_ganadores_mismas_fechas.csv", index=False)
        pd.DataFrame({f"M{k}": m["serie"].loc[fechas] for k, m in modelos.items()}) \
            .rename_axis("fecha").to_csv(SALIDA / "ic_series_ganadores.csv")
        grafico_ic_acumulado(modelos, fechas, SALIDA / "ic_acumulado.png")
    if not C_det.empty:
        C_det.to_csv(SALIDA / "C_misma_configuracion.csv")
        C_res.to_csv(SALIDA / "C_resumen_rejilla.csv")
    if not D.empty:
        D.to_csv(SALIDA / "D_efectos_2x2.csv", index=False)

    definiciones = pd.DataFrame([
        ("Modelos", " | ".join(m["nombre"] for m in modelos.values())),
        ("A_ganadores", "Mejor configuración de cada modelo, con las métricas de su propio backtest"),
        ("B_mismas_fechas", "Los ganadores evaluados en la intersección de sus fechas de IC; "
                            "dif_ic_vs_modelo_ref y t_stat_dif = diferencia mensual de IC contra el modelo de referencia (prueba pareada)"),
        ("C_misma_configuracion",
         "IC Ratio e IC medio de cada una de las configuraciones de la rejilla en cada modelo"),
        ("C_resumen_rejilla", "Promedio/mediana sobre toda la rejilla y # de configuraciones donde cada modelo es el mejor por IC Ratio. "
                              "Menos sesgado por selección que comparar solo a los ganadores"),
        ("D_efectos_2x2", "efecto_quitar_restriccion = promedio de (M2-M1) y (M4-M3); "
                          "efecto_no_estandarizar = promedio de (M3-M1) y (M4-M2); "
                          "interaccion = (M4-M3) - (M2-M1)"),
        ("ic_medio / ic_std / ic_ratio / t_stat / hit_rate",
         "Ver definiciones de optimizar_modelos.py"),
    ], columns=["concepto", "definición"])
    with pd.ExcelWriter(SALIDA / "comparar_modelos.xlsx", engine="openpyxl") as xw:
        A.to_excel(xw, sheet_name="A_ganadores", index=False)
        if not B.empty:
            B.to_excel(xw, sheet_name="B_mismas_fechas", index=False)
        if not C_det.empty:
            C_res.to_excel(xw, sheet_name="C_resumen_rejilla")
            C_det.to_excel(xw, sheet_name="C_por_configuracion")
        if not D.empty:
            D.to_excel(xw, sheet_name="D_efectos_2x2", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)

    print(f"\nComparación guardada en {SALIDA}")


if __name__ == "__main__":
    main()
