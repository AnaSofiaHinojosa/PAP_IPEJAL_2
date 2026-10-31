"""
revisar_cobertura_minobs.py - Revisión de cobertura (entre la Parte 1 y la Parte 2).

Pregunta que contesta: en cada re-estimación de betas, de las empresas que
el modelo tiene que predecir, ¿cuántas tienen REALMENTE el mínimo de
observaciones (min_obs) para entrar a la regresión?

Regla del profesor: si en algún año menos del 80% de las empresas cumple el
mínimo, hay que agregar más datos.

Usa las MISMAS fechas de corte y ventanas que backtesting.py:
    - cortes: cada FRECUENCIA_DEFAULT meses (12 = una vez al año) sobre las
      fechas de factores.csv, saltando cortes con menos de min_obs meses
    - ventana_beta = None -> todo el histórico hasta el corte
      ventana_beta = N    -> desde (corte - N meses) hasta el corte

Empresas que se revisan en cada corte:
    las que están en el Top 100 en algún mes entre ese corte y el siguiente,
    o sea, las que el modelo tiene que predecir con los betas de ese corte.

Observación válida:
    un mes donde la acción tiene rendimiento Y todos los factores + rf tienen dato.
    Se cuenta de dos formas:
      - n_obs_miembros:  solo meses en el Top 100 (rendimientos_miembros.csv),
                         que es lo que usa HOY la regresión
      - n_obs_historial: todos los meses con precio (rendimientos.csv),
                         aunque la empresa todavía no estuviera en el Top 100

DIAGNÓSTICO de cada empresa que NO cumple con n_obs_miembros:
    "historial"  -> con su historial completo SÍ llega a min_obs:
                    se resuelve usando rendimientos.csv, sin descargar nada
    "descargar"  -> ni con su historial completo llega: le faltan precios
                    (hay que buscar más datos, p. ej. en FactSet)
    "factores"   -> la ventana no tiene suficientes meses con TODOS los
                    factores (límite de EDGAR al inicio, 2011-2012); no se
                    arregla con datos de la empresa

Lee:
    salidas/regresion/rendimientos_miembros.csv
    salidas/regresion/rendimientos.csv
    salidas/regresion/factores.csv
    salidas/regresion/membresia.csv   (si no existe, la membresía se toma
                                       de los meses con rendimiento)

Escribe en salidas/cobertura/:
    cobertura_minobs.csv     una fila por corte x ventana x min_obs
    cobertura_minobs.xlsx    hojas: Resumen_pct, Cobertura, Diagnostico,
                             Descargar_datos, Detalle_empresas, Definiciones
"""

from pathlib import Path
import pandas as pd

from estimar_exposiciones import FACTORES_DEFAULT
from backtesting import FRECUENCIA_DEFAULT

REGRESION = Path("salidas/regresion")
SALIDA = Path("salidas/cobertura")

# ------------------------------------------------------------------ #
# Parámetros — AJUSTA AQUÍ (que coincidan con optimizar_modelos.py)
# ------------------------------------------------------------------ #
FACTORES = FACTORES_DEFAULT
COLUMNA_RF = "rf"
MIN_OBS_GRID = [12, 18, 24]             # mínimos a revisar
VENTANAS_BETA = [None, 36, 60, 120]     # None = histórico completo; resto en meses
FRECUENCIA_REESTIMACION = FRECUENCIA_DEFAULT
UMBRAL = 0.80                           # % mínimo de empresas que deben cumplir


def cargar_panel(archivo: Path) -> pd.DataFrame:
    """Panel fecha x ticker (acepta formato largo o ancho)."""
    df = pd.read_csv(archivo)
    df["fecha"] = pd.to_datetime(df["fecha"])
    if {"fecha", "ticker"}.issubset(df.columns):
        col_valor = [c for c in df.columns if c not in ("fecha", "ticker")][0]
        ancho = df.pivot(index="fecha", columns="ticker", values=col_valor)
    else:
        ancho = df.set_index("fecha")
    return ancho.sort_index()


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo rendimientos, factores y membresía...")
    rend = cargar_panel(REGRESION / "rendimientos_miembros.csv")
    rend_hist = cargar_panel(REGRESION / "rendimientos.csv").reindex(
        index=rend.index, columns=rend.columns)

    factores_df = pd.read_csv(REGRESION / "factores.csv")
    factores_df["fecha"] = pd.to_datetime(factores_df["fecha"])
    factores_df = factores_df.set_index("fecha").sort_index()

    columnas = list(FACTORES) + ([COLUMNA_RF] if COLUMNA_RF in factores_df.columns else [])
    mes_factores_ok = factores_df[columnas].notna().all(axis=1)
    mes_factores_ok = mes_factores_ok.reindex(rend.index, fill_value=False)

    # True donde el mes cuenta como observación para la regresión
    obs_miembros = rend.notna() & mes_factores_ok.to_numpy()[:, None]
    obs_historial = rend_hist.notna() & mes_factores_ok.to_numpy()[:, None]

    # Primer mes con precio de cada empresa (para el diagnóstico)
    primer_precio = rend_hist.apply(lambda s: s.first_valid_index())

    archivo_memb = REGRESION / "membresia.csv"
    if archivo_memb.exists():
        membresia = cargar_panel(archivo_memb).fillna(0).astype(bool)
    else:
        print("  (no hay membresia.csv; se usa rendimientos_miembros como membresía)")
        membresia = rend.notna()

    # Mismas fechas de corte que construir_exposiciones_rolling()
    fechas = factores_df.index.sort_values()
    cortes = [f for i, f in enumerate(fechas) if i % FRECUENCIA_REESTIMACION == 0]

    filas, detalle, diagnostico = [], [], []
    for j, corte in enumerate(cortes):
        siguiente = cortes[j + 1] if j + 1 < len(cortes) else fechas[-1]
        # Empresas que se predicen con los betas de este corte: (corte, siguiente]
        memb_periodo = membresia[(membresia.index > corte) & (membresia.index <= siguiente)]
        empresas = memb_periodo.columns[memb_periodo.any(axis=0)]
        if len(empresas) == 0:
            continue
        om = obs_miembros.reindex(columns=empresas, fill_value=False)
        oh = obs_historial.reindex(columns=empresas, fill_value=False)

        fila_detalle = pd.DataFrame({"fecha_corte": corte, "ticker": empresas})
        for ventana in VENTANAS_BETA:
            etiqueta = "completo" if ventana is None else f"{ventana}m"
            inicio = None if ventana is None else corte - pd.DateOffset(months=ventana)
            n_meses_fechas = len(factores_df.loc[inicio:corte])
            n_meses_factores = int(mes_factores_ok.loc[inicio:corte].sum())
            n_obs_m = om.loc[inicio:corte].sum(axis=0)
            n_obs_h = oh.loc[inicio:corte].sum(axis=0)
            fila_detalle[f"n_obs_miembros_{etiqueta}"] = n_obs_m.to_numpy()
            fila_detalle[f"n_obs_historial_{etiqueta}"] = n_obs_h.to_numpy()

            for min_obs in MIN_OBS_GRID:
                if n_meses_fechas < min_obs:
                    continue  # el backtest también se salta este corte
                cumplen_m = n_obs_m >= min_obs
                cumplen_h = n_obs_h >= min_obs
                pct_m = cumplen_m.mean()
                pct_h = cumplen_h.mean()
                filas.append({
                    "fecha_corte": corte,
                    "anio": corte.year,
                    "ventana_beta_meses": etiqueta,
                    "min_obs": min_obs,
                    "n_empresas_a_predecir": len(empresas),
                    "n_cumplen_min_obs": int(cumplen_m.sum()),
                    "pct_cumplen": pct_m,
                    "alerta_menor_80": pct_m < UMBRAL,
                    "n_cumplen_con_historial": int(cumplen_h.sum()),
                    "pct_cumplen_con_historial": pct_h,
                    "alerta_con_historial": pct_h < UMBRAL,
                })

                # Diagnóstico de cada empresa que hoy no cumple
                for t in empresas[~cumplen_m.to_numpy()]:
                    if n_meses_factores < min_obs:
                        motivo = "factores"
                    elif cumplen_h[t]:
                        motivo = "historial"
                    else:
                        motivo = "descargar"
                    pp = primer_precio.get(t)
                    diagnostico.append({
                        "fecha_corte": corte,
                        "anio": corte.year,
                        "ventana_beta_meses": etiqueta,
                        "min_obs": min_obs,
                        "ticker": t,
                        "n_obs_miembros": int(n_obs_m[t]),
                        "n_obs_historial": int(n_obs_h[t]),
                        "faltan_para_min_obs": max(min_obs - int(n_obs_h[t]), 0),
                        "primer_mes_con_precio": pp.date() if pd.notna(pp) else None,
                        "diagnostico": motivo,
                    })
        detalle.append(fila_detalle)

    cobertura = pd.DataFrame(filas)
    detalle = pd.concat(detalle, ignore_index=True)
    diagnostico = pd.DataFrame(diagnostico)
    cobertura.to_csv(SALIDA / "cobertura_minobs.csv", index=False)

    # Lista única de empresas a las que les faltan precios
    if not diagnostico.empty:
        desc = diagnostico[diagnostico["diagnostico"] == "descargar"]
        descargar = (desc.groupby("ticker")
                     .agg(primer_mes_con_precio=("primer_mes_con_precio", "first"),
                          cortes_afectados=("fecha_corte",
                                            lambda s: ", ".join(sorted({f"{d:%Y-%m}" for d in s}))),
                          max_meses_faltantes=("faltan_para_min_obs", "max"))
                     .reset_index()
                     .sort_values("max_meses_faltantes", ascending=False))
    else:
        descargar = pd.DataFrame(columns=["ticker", "primer_mes_con_precio",
                                          "cortes_afectados", "max_meses_faltantes"])

    # Resumen ancho: filas = año del corte, columnas = (ventana, min_obs), valor = % que cumple
    resumen = cobertura.pivot_table(index="anio",
                                    columns=["ventana_beta_meses", "min_obs"],
                                    values="pct_cumplen")
    resumen_hist = cobertura.pivot_table(index="anio",
                                         columns=["ventana_beta_meses", "min_obs"],
                                         values="pct_cumplen_con_historial")

    # ---- Consola ---- #
    alertas = cobertura[cobertura["alerta_menor_80"]]
    print(f"\nCortes revisados: {cobertura['fecha_corte'].nunique()} "
          f"({cobertura['fecha_corte'].min():%Y-%m} a {cobertura['fecha_corte'].max():%Y-%m})")
    print(f"{'=' * 78}")
    if alertas.empty:
        print(f"OK: en todos los cortes y combinaciones cumple al menos el {UMBRAL:.0%}.")
    else:
        print(f"CORTES CON MENOS DEL {UMBRAL:.0%} (hoy -> usando el historial completo):")
        for _, a in alertas.iterrows():
            marca = "  <- sigue < 80%" if a["alerta_con_historial"] else ""
            print(f"  {a['fecha_corte']:%Y-%m}  ventana={a['ventana_beta_meses']:>8}  "
                  f"min_obs={a['min_obs']:>2}  ->  {a['n_cumplen_min_obs']:>3}/"
                  f"{a['n_empresas_a_predecir']} ({a['pct_cumplen']:.1%})  ->  "
                  f"{a['n_cumplen_con_historial']:>3}/{a['n_empresas_a_predecir']} "
                  f"({a['pct_cumplen_con_historial']:.1%}){marca}")
    print(f"{'=' * 78}")

    if not diagnostico.empty:
        print("\nEmpresas que NO cumplen, por diagnóstico (empresas únicas):")
        for motivo, texto in [("historial", "se resuelven con su historial (rendimientos.csv)"),
                              ("descargar", "les faltan precios -> descargar más datos"),
                              ("factores", "límite de factores al inicio (EDGAR)")]:
            n = diagnostico.loc[diagnostico["diagnostico"] == motivo, "ticker"].nunique()
            print(f"  {motivo:<10} {n:>3}  {texto}")
        if not descargar.empty:
            print("\nEmpresas a las que hay que DESCARGAR más datos:")
            for _, d in descargar.iterrows():
                print(f"  {d['ticker']:<8} primer precio {d['primer_mes_con_precio']}  "
                      f"faltan hasta {d['max_meses_faltantes']} meses  "
                      f"(cortes {d['cortes_afectados']})")

    definiciones = pd.DataFrame([
        ("fecha_corte", "Fecha en que el backtest re-estima los betas (misma regla que backtesting.py)"),
        ("n_empresas_a_predecir", "Empresas en el Top 100 en algún mes entre ese corte y el siguiente"),
        ("n_cumplen_min_obs / pct_cumplen", "Cuántas tienen >= min_obs meses válidos contando SOLO "
                                            "meses en el Top 100 (lo que usa hoy la regresión)"),
        ("n_cumplen_con_historial / pct_cumplen_con_historial",
         "Igual, pero contando todos los meses con precio (rendimientos.csv)"),
        ("alerta_menor_80", f"True si pct_cumplen < {UMBRAL:.0%}"),
        ("alerta_con_historial", f"True si incluso con el historial completo queda < {UMBRAL:.0%}"),
        ("mes válido", "Mes con rendimiento de la acción y todos los factores + rf"),
        ("ventana_beta_meses", "'completo' = todo el histórico hasta el corte; "
                               "número = desde (corte - N meses) hasta el corte"),
        ("diagnostico = historial", "Con su historial completo sí llega a min_obs: se resuelve con rendimientos.csv"),
        ("diagnostico = descargar", "Ni con su historial completo llega: faltan precios (buscar en FactSet)"),
        ("diagnostico = factores", "La ventana no tiene suficientes meses con todos los factores (EDGAR)"),
        ("faltan_para_min_obs", "Meses que le faltan a la empresa, aun con historial, para llegar a min_obs"),
        ("primer_mes_con_precio", "Primer mes con rendimiento en rendimientos.csv (desde 2010)"),
    ], columns=["concepto", "definición"])

    with pd.ExcelWriter(SALIDA / "cobertura_minobs.xlsx", engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="Resumen_pct")
        resumen_hist.to_excel(xw, sheet_name="Resumen_pct_historial")
        cobertura.to_excel(xw, sheet_name="Cobertura", index=False)
        diagnostico.to_excel(xw, sheet_name="Diagnostico", index=False)
        descargar.to_excel(xw, sheet_name="Descargar_datos", index=False)
        detalle.to_excel(xw, sheet_name="Detalle_empresas", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)

    print(f"\nResultados en {SALIDA}")


if __name__ == "__main__":
    main()