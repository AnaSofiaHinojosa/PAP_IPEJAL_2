from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

from estimar_exposiciones import FACTORES_DEFAULT
from backtesting import backtest_modelo, FRECUENCIA_DEFAULT

REGRESION = Path("salidas/regresion")
SALIDA = Path("salidas/backtesting")

FACTORES = FACTORES_DEFAULT
# None = histórico completo; resto en meses
VENTANAS_BETA = [None, 36, 60, 120]
# mínimo de meses para poder estimar una acción
MIN_OBS_GRID = [12, 18, 24]
# ventana del rolling mean del factor esperado (Parte 3)
VENTANAS_FACTOR = [3, 6, 12]
# cada cuántos meses se recalculan los betas
FRECUENCIA_REESTIMACION = FRECUENCIA_DEFAULT
REZAGO_FACTOR = 1                       # evita look-ahead en el factor esperado
# estandarizar factores en cada corte (consistente con la Parte 2)
ESTANDARIZAR = True
# estandarizar también el rendimiento en exceso en cada corte
ESTANDARIZAR_Y = True
# (z-score global, solo con datos hasta el corte; igual que la Parte 2)
# mínimo de acciones con dato para calcular IC ese mes
MIN_ACCIONES_IC = 10
# meses mínimos de IC para que un modelo entre a la comparación
MIN_PERIODOS = 24
# True  = los betas se estiman con el historial completo de precios de cada empresa
#         (rendimientos.csv), aunque todavía no estuviera en el Top 100.
#         Resuelve las empresas recién entradas (revisar_cobertura_minobs.py).
# False = solo con los meses en el Top 100 (rendimientos_miembros.csv), como antes.
# En ambos casos el IC se mide SOLO con las empresas del Top 100 de ese mes.
ENTRENAR_CON_HISTORIAL = True


def cargar_rendimientos(archivo: Path) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    df["fecha"] = pd.to_datetime(df["fecha"])
    if {"fecha", "ticker"}.issubset(df.columns):
        col_valor = "rendimiento" if "rendimiento" in df.columns else df.columns[2]
        ancho = df.pivot(index="fecha", columns="ticker", values=col_valor)
    else:
        ancho = df.set_index("fecha")
    return ancho.sort_index()


def cargar_factores(archivo: Path) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha").sort_index()


def serie_ic(ic) -> pd.Series:
    """Devuelve el IC mensual como Series (sin NaN), venga como Series o DataFrame."""
    if isinstance(ic, pd.DataFrame):
        col = "ic" if "ic" in ic.columns else ic.columns[0]
        ic = ic[col]
    return ic.dropna()


def resumir_ic(serie: pd.Series) -> dict:
    """Mismas métricas que el resumen de backtest_modelo, pero sobre la serie que se le pase."""
    n = len(serie)
    if n < 2:
        return {"ic_medio": np.nan, "ic_std": np.nan, "ic_ratio": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "n_periodos": n}
    medio = serie.mean()
    std = serie.std()
    return {
        "ic_medio": medio,
        "ic_std": std,
        "ic_ratio": medio / std if std > 0 else np.nan,
        "t_stat": medio / (std / np.sqrt(n)) if std > 0 else np.nan,
        "hit_rate": (serie > 0).mean(),
        "n_periodos": n,
    }


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo rendimientos y factores...")
    rendimientos_miembros = cargar_rendimientos(
        REGRESION / "rendimientos_miembros.csv")
    factores_df = cargar_factores(REGRESION / "factores.csv")
    # El rendimiento "real" (ex post) para el IC: solo empresas del Top 100 ese mes
    rendimiento_real = rendimientos_miembros
    # Panel con el que se estiman los betas
    if ENTRENAR_CON_HISTORIAL:
        rendimientos = cargar_rendimientos(REGRESION / "rendimientos.csv")
        print("  Betas con historial completo (rendimientos.csv); IC solo con miembros del Top 100")
    else:
        rendimientos = rendimientos_miembros
        print("  Betas e IC solo con meses en el Top 100 (rendimientos_miembros.csv)")
    print(
        f"  estandarizar factores = {ESTANDARIZAR}, estandarizar rendimientos = {ESTANDARIZAR_Y}")

    combinaciones = list(product(VENTANAS_BETA, MIN_OBS_GRID, VENTANAS_FACTOR))
    print(f"\nCorriendo {len(combinaciones)} modelos "
          f"({len(VENTANAS_BETA)} ventanas de beta x {len(MIN_OBS_GRID)} min_obs x "
          f"{len(VENTANAS_FACTOR)} ventanas de factor)...\n")

    parametros = {}
    resultados_por_modelo = {}

    for i, (ventana_beta, min_obs, ventana_factor) in enumerate(combinaciones, 1):
        etiqueta_ventana = "completo" if ventana_beta is None else f"{ventana_beta}m"
        modelo_id = f"beta={etiqueta_ventana}_minobs={min_obs}_factor={ventana_factor}m"
        print(f"[{i}/{len(combinaciones)}] {modelo_id}...", end=" ")

        resultado = backtest_modelo(
            rendimientos=rendimientos,
            factores_df=factores_df,
            rendimiento_real=rendimiento_real,
            factores=FACTORES,
            ventana_beta=ventana_beta,
            min_obs=min_obs,
            frecuencia_meses=FRECUENCIA_REESTIMACION,
            ventana_factor=ventana_factor,
            rezago_factor=REZAGO_FACTOR,
            estandarizar=ESTANDARIZAR,
            min_acciones_ic=MIN_ACCIONES_IC,
            estandarizar_y=ESTANDARIZAR_Y,     # pasa a estimar_matriz_exposiciones
        )
        resultados_por_modelo[modelo_id] = resultado
        parametros[modelo_id] = {
            "ventana_beta_meses": etiqueta_ventana,
            "min_obs": min_obs,
            "ventana_factor_meses": ventana_factor,
        }
        r = resultado["resumen"]
        print(f"IC medio={r['ic_medio']:.3f}  IC Ratio={r['ic_ratio']:.3f}  "
              f"n={r['n_periodos']}" if r["n_periodos"] else "sin suficientes periodos")

    # ---- Fechas comunes: solo modelos con suficientes meses de IC ---- #
    series = {m: serie_ic(res["ic_serie"])
              for m, res in resultados_por_modelo.items()}
    incluidos = [m for m, s in series.items() if len(s) >= MIN_PERIODOS]
    if not incluidos:
        print(
            f"\nAVISO: ningún modelo tiene {MIN_PERIODOS}+ meses de IC; se usan todos.")
        incluidos = [m for m, s in series.items() if len(s) > 0]

    fechas_comunes = series[incluidos[0]].index
    for m in incluidos[1:]:
        fechas_comunes = fechas_comunes.intersection(series[m].index)
    fechas_comunes = fechas_comunes.sort_values()

    if len(fechas_comunes) == 0:
        raise RuntimeError(
            "Los modelos no comparten ninguna fecha de IC; revisa la rejilla.")
    print(f"\nComparando {len(incluidos)} modelos en las MISMAS {len(fechas_comunes)} fechas: "
          f"{fechas_comunes[0]:%Y-%m} a {fechas_comunes[-1]:%Y-%m}")
    if len(fechas_comunes) < MIN_PERIODOS:
        print(
            f"AVISO: solo {len(fechas_comunes)} fechas en común (< {MIN_PERIODOS}).")

    # ---- Métricas sobre fechas comunes ---- #
    filas_resumen = []
    for modelo_id, s in series.items():
        incluido = modelo_id in incluidos
        metricas = resumir_ic(s.loc[fechas_comunes]
                              ) if incluido else resumir_ic(s.iloc[:0])
        filas_resumen.append({
            "modelo_id": modelo_id,
            **parametros[modelo_id],
            **metricas,
            # meses de IC que el modelo tenía por sí solo
            "n_periodos_propio": len(s),
            "incluido_en_comparacion": incluido,
        })
    comparacion = pd.DataFrame(filas_resumen)

    # ---- Ranking promedio ---- #
    comparacion["rank_ic_medio"] = comparacion["ic_medio"].rank(
        ascending=False, method="min")
    comparacion["rank_ic_std"] = comparacion["ic_std"].rank(
        ascending=True, method="min")
    comparacion["rank_ic_ratio"] = comparacion["ic_ratio"].rank(
        ascending=False, method="min")
    comparacion["ranking_promedio"] = comparacion[
        ["rank_ic_medio", "rank_ic_std", "rank_ic_ratio"]].mean(axis=1)
    comparacion = comparacion.sort_values(
        ["rank_ic_ratio", "ic_ratio"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)
    comparacion.to_csv(SALIDA / "comparacion_modelos.csv", index=False)

    # ---- Selección del modelo ganador ----
    ganador = comparacion.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"MODELO GANADOR: {ganador['modelo_id']}")
    print(
        f"  Ranking promedio = {ganador['ranking_promedio']:.2f}  <- criterio de selección")
    print(f"    lugar IC medio = {ganador['rank_ic_medio']:.0f}")
    print(f"    lugar IC std   = {ganador['rank_ic_std']:.0f}")
    print(f"    lugar IC Ratio = {ganador['rank_ic_ratio']:.0f}")
    print(f"  IC medio   = {ganador['ic_medio']:.4f}")
    print(f"  IC std     = {ganador['ic_std']:.4f}")
    print(f"  IC Ratio   = {ganador['ic_ratio']:.4f}")
    print(f"  t-stat     = {ganador['t_stat']:.2f}")
    print(f"  hit rate   = {ganador['hit_rate']:.1%}")
    print(f"  n periodos = {ganador['n_periodos']} (fechas comunes)")
    print(f"{'=' * 70}")

    resultado_ganador = resultados_por_modelo[ganador["modelo_id"]]
    resultado_ganador["ic_serie"].to_csv(SALIDA / "mejor_modelo_ic_serie.csv")
    resultado_ganador["rendimiento_esperado"].to_csv(
        SALIDA / "mejor_modelo_rendimiento_esperado.csv")

    carpeta_exp = SALIDA / "mejor_modelo_exposiciones"
    carpeta_exp.mkdir(exist_ok=True)
    for fecha, matriz in resultado_ganador["exposiciones_por_fecha"].items():
        matriz.to_csv(carpeta_exp / f"{fecha:%Y-%m}.csv")

    definiciones = pd.DataFrame([
        ("ic_medio", "Promedio de la correlación de Spearman mensual entre rendimiento "
                     "esperado (ex ante) y rendimiento real (ex post), en las fechas comunes"),
        ("ic_std", "Desviación estándar del IC mensual (qué tan ruidosa es la señal)"),
        ("ic_ratio", "ic_medio / ic_std — qué tan consistente es la señal en el tiempo"),
        ("t_stat", "ic_medio / (ic_std / sqrt(n)) — significancia estadística del IC"),
        ("hit_rate", "% de meses con IC > 0"),
        ("n_periodos", "Número de meses (fechas comunes a todos los modelos) usados para el IC"),
        ("n_periodos_propio",
         "Meses de IC que el modelo tenía por sí solo, antes de alinear fechas"),
        ("incluido_en_comparacion",
         f"False si el modelo tenía menos de {MIN_PERIODOS} meses de IC"),
        ("rank_ic_medio", "Lugar por IC medio (1 = mayor)"),
        ("rank_ic_std", "Lugar por IC std (1 = menor)"),
        ("rank_ic_ratio", "Lugar por IC Ratio (1 = mayor)"),
        ("ranking_promedio",
         "Promedio de los tres lugares — criterio de selección (menor gana)"),
        ("ventana_beta_meses", "'completo' = histórico completo (expanding); si es un "
                               "número, meses de ventana móvil usados para reestimar betas"),
        ("min_obs", "Mínimo de meses exigidos para estimar el beta de una acción"),
        ("ventana_factor_meses",
         "Ventana del rolling mean usado para el factor esperado (Parte 3)"),
    ], columns=["concepto", "definición"])

    with pd.ExcelWriter(SALIDA / "comparacion_modelos.xlsx", engine="openpyxl") as xw:
        comparacion.to_excel(xw, sheet_name="Comparacion", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)

    print(f"\nResultados completos en {SALIDA}")


if __name__ == "__main__":
    main()
