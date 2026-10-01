"""
optimizar_modelos.py - Parte 4: Backtesting + IC + Selección del modelo.

"Algoritmo de optimización a partir del IC": como las variables que se están
afinando son discretas por naturaleza (ventanas de 12/36/60 meses, no "12.37
meses"), lo correcto aquí NO es un optimizador de gradiente — es una
BÚSQUEDA EN REJILLA (grid search): se corre el backtest completo para cada
combinación posible de parámetros, y se elige la que dé el mejor IC Ratio.
Es simple, transparente (se puede graficar/tabular TODA la rejilla, no solo
el ganador) y es el estándar en la industria para "qué ventana usar".

Parámetros que se barren (rejilla):
    VENTANAS_BETA     ventana móvil para RE-ESTIMAR los betas de cada acción
                       (None = histórico completo, como en la Parte 2 original)
    MIN_OBS_GRID      mínimo de observaciones exigido en esa regresión
    VENTANAS_FACTOR   ventana del rolling mean del factor esperado (Parte 3)

Lee:
    salidas/regresion/rendimientos_miembros.csv
    salidas/regresion/factores.csv

Escribe en salidas/backtesting/:
    comparacion_modelos.csv      una fila por modelo, con sus métricas de IC
    comparacion_modelos.xlsx     lo mismo + hoja de definiciones
    mejor_modelo_ic_serie.csv    serie de tiempo del IC del modelo ganador
    mejor_modelo_rendimiento_esperado.csv
    mejor_modelo_exposiciones/   un CSV por fecha de reestimación del ganador
"""

from pathlib import Path
from itertools import product
import pandas as pd

from estimar_exposiciones import FACTORES_DEFAULT
from backtesting import backtest_modelo, FRECUENCIA_DEFAULT

REGRESION = Path("salidas/regresion")
SALIDA = Path("salidas/backtesting")

# ------------------------------------------------------------------ #
# Rejilla de modelos a comparar — AJUSTA AQUÍ
# ------------------------------------------------------------------ #
FACTORES = FACTORES_DEFAULT
VENTANAS_BETA = [None, 36, 60, 120]     # None = histórico completo; resto en meses
MIN_OBS_GRID = [12, 18, 24]             # mínimo de meses para poder estimar una acción
VENTANAS_FACTOR = [3, 6, 12]            # ventana del rolling mean del factor esperado (Parte 3)
FRECUENCIA_REESTIMACION = FRECUENCIA_DEFAULT  # cada cuántos meses se recalculan los betas
REZAGO_FACTOR = 1                       # evita look-ahead en el factor esperado
ESTANDARIZAR = True                     # estandarizar factores en cada corte (consistente con la Parte 2)
MIN_ACCIONES_IC = 10                    # mínimo de acciones con dato para calcular IC ese mes


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


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo rendimientos y factores...")
    rendimientos = cargar_rendimientos(REGRESION / "rendimientos_miembros.csv")
    factores_df = cargar_factores(REGRESION / "factores.csv")
    # El rendimiento "real" (ex post) para el IC es el mismo panel de rendimientos
    rendimiento_real = rendimientos

    combinaciones = list(product(VENTANAS_BETA, MIN_OBS_GRID, VENTANAS_FACTOR))
    print(f"\nCorriendo {len(combinaciones)} modelos "
          f"({len(VENTANAS_BETA)} ventanas de beta x {len(MIN_OBS_GRID)} min_obs x "
          f"{len(VENTANAS_FACTOR)} ventanas de factor)...\n")

    filas_resumen = []
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
        )
        resultados_por_modelo[modelo_id] = resultado
        r = resultado["resumen"]
        print(f"IC medio={r['ic_medio']:.3f}  IC Ratio={r['ic_ratio']:.3f}  "
              f"n={r['n_periodos']}" if r["n_periodos"] else "sin suficientes periodos")

        filas_resumen.append({
            "modelo_id": modelo_id,
            "ventana_beta_meses": etiqueta_ventana,
            "min_obs": min_obs,
            "ventana_factor_meses": ventana_factor,
            **r,
        })

    comparacion = pd.DataFrame(filas_resumen).sort_values("ic_ratio", ascending=False)
    comparacion.to_csv(SALIDA / "comparacion_modelos.csv", index=False)

    # ---- Selección del modelo ganador ----
    validos = comparacion[comparacion["n_periodos"] >= 24]  # al menos 2 años de IC para confiar en el ratio
    if validos.empty:
        validos = comparacion
    ganador = validos.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"MODELO GANADOR: {ganador['modelo_id']}")
    print(f"  IC medio   = {ganador['ic_medio']:.4f}")
    print(f"  IC std     = {ganador['ic_std']:.4f}")
    print(f"  IC Ratio   = {ganador['ic_ratio']:.4f}  <- criterio de selección")
    print(f"  t-stat     = {ganador['t_stat']:.2f}")
    print(f"  hit rate   = {ganador['hit_rate']:.1%}")
    print(f"  n periodos = {ganador['n_periodos']}")
    print(f"{'=' * 70}")

    resultado_ganador = resultados_por_modelo[ganador["modelo_id"]]
    resultado_ganador["ic_serie"].to_csv(SALIDA / "mejor_modelo_ic_serie.csv")
    resultado_ganador["rendimiento_esperado"].to_csv(SALIDA / "mejor_modelo_rendimiento_esperado.csv")

    carpeta_exp = SALIDA / "mejor_modelo_exposiciones"
    carpeta_exp.mkdir(exist_ok=True)
    for fecha, matriz in resultado_ganador["exposiciones_por_fecha"].items():
        matriz.to_csv(carpeta_exp / f"{fecha:%Y-%m}.csv")

    definiciones = pd.DataFrame([
        ("ic_medio", "Promedio de la correlación de Spearman mensual entre rendimiento "
                     "esperado (ex ante) y rendimiento real (ex post)"),
        ("ic_std", "Desviación estándar del IC mensual (qué tan ruidosa es la señal)"),
        ("ic_ratio", "ic_medio / ic_std — qué tan consistente es la señal en el tiempo "
                     "(criterio principal de selección)"),
        ("t_stat", "ic_medio / (ic_std / sqrt(n)) — significancia estadística del IC"),
        ("hit_rate", "% de meses con IC > 0"),
        ("n_periodos", "Número de meses usados para calcular el IC de ese modelo"),
        ("ventana_beta_meses", "'completo' = histórico completo (expanding); si es un "
                               "número, meses de ventana móvil usados para reestimar betas"),
        ("min_obs", "Mínimo de meses exigidos para estimar el beta de una acción"),
        ("ventana_factor_meses", "Ventana del rolling mean usado para el factor esperado (Parte 3)"),
    ], columns=["concepto", "definición"])

    with pd.ExcelWriter(SALIDA / "comparacion_modelos.xlsx", engine="openpyxl") as xw:
        comparacion.to_excel(xw, sheet_name="Comparacion", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)

    print(f"\nResultados completos en {SALIDA}")


if __name__ == "__main__":
    main()