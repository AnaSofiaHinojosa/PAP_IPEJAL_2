"""
diagnostico_r2.py - ¿Por qué salen R^2 negativos en la matriz de exposiciones?

Compara, con la configuración ACTUAL de la Parte 2 (factores estandarizados,
rendimientos sin estandarizar, suma de betas = 1), los dos archivos de
rendimientos:
    rendimientos_miembros.csv  (solo meses en el Top 100 - como antes)
    rendimientos.csv           (historial completo - el cambio nuevo)

Para cada uno imprime:
  1. R^2 con la configuración actual (con restricción)
  2. R^2 SIN restricción (mismos datos, misma estandarización)
  3. Suma "natural" de los betas: la que sale sin restricción.
     Si es muy distinta de 1, la restricción está obligando a los betas a
     una escala que los datos no tienen -> R^2 negativo.
  4. Desviación estándar de los rendimientos vs. de los factores
     (las "unidades" de cada lado de la regresión).

No modifica ningún archivo. Correr desde la raíz del repo:
    python diagnostico_r2.py
"""

from pathlib import Path
import pandas as pd

from estimar_exposiciones import estimar_matriz_exposiciones, FACTORES_DEFAULT

REGRESION = Path("salidas/regresion")


def cargar(archivo: Path) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha").sort_index()


def main():
    factores_df = cargar(REGRESION / "factores.csv")
    rf = factores_df["rf"]
    desv_factores = factores_df[FACTORES_DEFAULT].std()

    print("Desviación estándar mensual de los factores CRUDOS:")
    print("  " + "  ".join(f"{f}={v:.3f}" for f, v in desv_factores.items()))
    print("  (estandarizados, todos pasan a 1.000)\n")

    for archivo in ["rendimientos_miembros.csv", "rendimientos.csv"]:
        rend = cargar(REGRESION / archivo)
        exceso = rend.sub(rf.reindex(rend.index), axis=0)

        con = estimar_matriz_exposiciones(rend, factores_df, estandarizar=True,
                                          estandarizar_y=False, restringir_suma=True)
        sin = estimar_matriz_exposiciones(rend, factores_df, estandarizar=True,
                                          estandarizar_y=False, restringir_suma=False)
        suma_natural = sin[FACTORES_DEFAULT].sum(axis=1)
        ok = con["r2"].notna()

        print(f"=== {archivo} ===")
        print(f"  empresas con beta:                       {ok.sum()}")
        print(f"  desv. típica del rendimiento en exceso:  {exceso.std().median():.3f} (mediana por empresa)")
        print(f"  R^2 CON restricción (config. actual):    mediana {con['r2'].median():8.3f}   "
              f"negativos: {(con['r2'] < 0).sum()}   < 0.1: {(con['r2'] < 0.1).sum()}")
        print(f"  R^2 SIN restricción:                     mediana {sin['r2'].median():8.3f}   "
              f"negativos: {(sin['r2'] < 0).sum()}")
        print(f"  suma natural de betas (sin restricción): mediana {suma_natural.median():.3f}   "
              f"rango [{suma_natural.min():.3f}, {suma_natural.max():.3f}]")
        print(f"  -> la restricción la obliga a ser 1.000\n")


if __name__ == "__main__":
    main()