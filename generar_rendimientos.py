"""
generar_rendimientos.py - Parte 3: Rendimiento esperado por acción.

Lee:
    salidas/exposiciones/matriz_exposiciones.csv   (Parte 2: beta de cada
                                                     acción a cada factor)
    salidas/regresion/factores.csv                 (Parte 1: rendimiento
                                                     mensual de cada factor)

Para cada ventana en VENTANAS (3, 6 y 12 meses por defecto):
    1. Calcula el rendimiento esperado de cada factor: z-score del rolling
       mean de los últimos `ventana` meses (rezagado, ver
       estimar_rendimiento_esperado.py).
    2. Calcula el rendimiento esperado de cada acción:
           E[R_i,t] = sum_k beta_i,k * factor_esperado_k,t

Escribe en salidas/rendimientos_esperados/:
    rendimientos_esperados_ventana{3,6,12}.csv   fecha x ticker
    rendimientos_esperados.xlsx                  una hoja por ventana +
                                                  hoja de resumen/cobertura
"""

from pathlib import Path
import pandas as pd

from estimar_exposiciones import FACTORES_DEFAULT
from estimar_rendimiento import (
    calcular_matriz_factores_esperados,
    calcular_rendimiento_esperado,
    REZAGO_DEFAULT,
    METODO_Z_DEFAULT,
)

EXPOSICIONES = Path("salidas/exposiciones/matriz_exposiciones.csv")
FACTORES_CSV = Path("salidas/regresion/factores.csv")
SALIDA = Path("salidas/rendimientos_esperados")

# Parámetros ajustables por el equipo
FACTORES = FACTORES_DEFAULT
VENTANAS = [3, 6, 12]          # meses del rolling mean; el enunciado pide probar estas 3
REZAGO_MESES = REZAGO_DEFAULT  # 1 = evita look-ahead bias; poner 0 si el profe pide sin rezago
METODO_Z = METODO_Z_DEFAULT    # "expandido" (recomendado) o "completo"
INCLUIR_ALPHA = False          # el enunciado solo pide sum(beta_k * factor_esperado_k)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo matriz de exposiciones (Parte 2) y factores (Parte 1)...")
    matriz_exposiciones = pd.read_csv(EXPOSICIONES, index_col="ticker")
    factores_df = pd.read_csv(FACTORES_CSV)
    factores_df["fecha"] = pd.to_datetime(factores_df["fecha"])
    factores_df = factores_df.set_index("fecha").sort_index()

    faltan = [f for f in FACTORES if f not in factores_df.columns]
    if faltan:
        raise ValueError(f"Faltan factores en factores.csv: {faltan}")

    # Empresas sin exposiciones válidas (n_obs insuficiente en la Parte 2)
    sin_beta = matriz_exposiciones[FACTORES].isna().any(axis=1)
    if sin_beta.any():
        print(f"  aviso: {sin_beta.sum()} empresas sin beta válido en la Parte 2 "
              f"quedarán con rendimiento esperado NaN: "
              f"{', '.join(matriz_exposiciones.index[sin_beta][:15])}"
              f"{' ...' if sin_beta.sum() > 15 else ''}")

    resumen_filas = []
    hojas_excel = {}

    for ventana in VENTANAS:
        print(f"\nVentana de {ventana} meses (rezago={REZAGO_MESES}, z-score={METODO_Z})...")
        factores_esperados = calcular_matriz_factores_esperados(
            factores_df, factores=FACTORES, ventana=ventana,
            rezago=REZAGO_MESES, metodo_z=METODO_Z,
        )
        rendimiento_esperado = calcular_rendimiento_esperado(
            matriz_exposiciones, factores_esperados,
            factores=FACTORES, incluir_alpha=INCLUIR_ALPHA,
        )

        n_meses_validos = rendimiento_esperado.notna().any(axis=1).sum()
        cobertura = rendimiento_esperado.notna().mean().mean()
        print(f"  {n_meses_validos} meses con al menos una acción con rendimiento esperado")
        print(f"  cobertura promedio (acción-mes con dato): {cobertura:.1%}")

        archivo = SALIDA / f"rendimientos_esperados_ventana{ventana}.csv"
        rendimiento_esperado.to_csv(archivo)
        hojas_excel[f"ventana_{ventana}m"] = rendimiento_esperado
        resumen_filas.append({
            "ventana_meses": ventana,
            "meses_con_dato": n_meses_validos,
            "cobertura_promedio": cobertura,
            "primera_fecha_valida": rendimiento_esperado.dropna(how="all").index.min(),
        })

    resumen = pd.DataFrame(resumen_filas)
    with pd.ExcelWriter(SALIDA / "rendimientos_esperados.xlsx", engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="Resumen", index=False)
        for nombre, df in hojas_excel.items():
            df.reset_index().rename(columns={"index": "fecha"}).to_excel(xw, sheet_name=nombre, index=False)

    print(f"\nRendimientos esperados guardados en {SALIDA}")


if __name__ == "__main__":
    main()