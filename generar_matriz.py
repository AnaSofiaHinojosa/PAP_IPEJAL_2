"""
generar_matriz.py - Parte 2: Matriz de exposiciones.

Lee las salidas de la Parte 1 (construir_factores.py):
    salidas/regresion/rendimientos_miembros.csv   fecha x ticker (formato largo)
    salidas/regresion/factores.csv                fecha x factor (formato largo)

Para cada acción, estima su exposición (beta) a cada factor propuesto
mediante una regresión de su rendimiento (en exceso de la tasa libre de
riesgo) contra los factores, con la restricción sum(beta) = 1
(estimar_exposiciones.py).

Escribe en salidas/exposiciones/:
    matriz_exposiciones.csv    una fila por empresa, una columna por factor
    matriz_exposiciones.xlsx   lo mismo + hoja "Definiciones" y avisos de
                                calidad (R^2 bajo, pocas observaciones, etc.)

Antes de correr:
    - Correr main.py y construir_factores.py para que
      existan los archivos en salidas/regresion/.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from estimar_exposiciones import (
    estimar_matriz_exposiciones,
    FACTORES_DEFAULT,
    FACTOR_BASE_DEFAULT,
    MIN_OBS_DEFAULT,
    VALOR_RESTRICCION_DEFAULT,
)

REGRESION = Path("salidas/regresion")
SALIDA = Path("salidas/exposiciones")

# Parámetros ajustables por el equipo
FACTORES = FACTORES_DEFAULT              # MERCADO, VALUE, TAMANO, MOMENTUM, VOLATILIDAD, LIQUIDEZ
FACTOR_BASE = FACTOR_BASE_DEFAULT         # factor que se despeja de la restricción de suma
RESTRINGIR_SUMA = True                    # sum(beta) = VALOR_RESTRICCION
VALOR_RESTRICCION = VALOR_RESTRICCION_DEFAULT
MIN_OBS = MIN_OBS_DEFAULT                 # mínimo de meses en común para estimar una empresa
VENTANA_MESES = None                      # None = usa todo el historial disponible (20 años);
                                           # o un entero, ej. 120, para usar solo los últimos N meses
R2_AVISO = 0.10                           # avisa si el R^2 de una empresa es menor a esto


def cargar_rendimientos(archivo: Path) -> pd.DataFrame:
    """Lee rendimientos_miembros.csv (formato largo: fecha, ticker, ...) y lo
    pivotea a fecha x ticker. Si el archivo ya viene ancho (una columna por
    ticker), lo regresa tal cual."""
    df = pd.read_csv(archivo)
    df["fecha"] = pd.to_datetime(df["fecha"])
    if {"fecha", "ticker"}.issubset(df.columns):
        # formato largo -> pivotear. La columna de rendimiento puede llamarse
        # distinto según la versión de construir_factores.py.
        col_valor = "rendimiento" if "rendimiento" in df.columns else df.columns[2]
        ancho = df.pivot(index="fecha", columns="ticker", values=col_valor)
    else:
        ancho = df.set_index("fecha")
    return ancho.sort_index()


def cargar_factores(archivo: Path) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha").sort_index()


def recortar_ventana(rendimientos: pd.DataFrame, factores_df: pd.DataFrame,
                     meses: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not meses:
        return rendimientos, factores_df
    fechas = rendimientos.index.union(factores_df.index).sort_values()
    corte = fechas[-meses:] if len(fechas) > meses else fechas
    return rendimientos.loc[rendimientos.index.isin(corte)], \
           factores_df.loc[factores_df.index.isin(corte)]


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo rendimientos y factores de la Parte 1...")
    rendimientos = cargar_rendimientos(REGRESION / "rendimientos_miembros.csv")
    factores_df = cargar_factores(REGRESION / "factores.csv")
    print(f"  {rendimientos.shape[1]} empresas, {rendimientos.shape[0]} meses de rendimientos")
    print(f"  factores disponibles: {', '.join(factores_df.columns)}")

    faltan = [f for f in FACTORES if f not in factores_df.columns]
    if faltan:
        raise ValueError(f"Faltan factores en factores.csv: {faltan}")

    rendimientos, factores_df = recortar_ventana(rendimientos, factores_df, VENTANA_MESES)

    print(f"\nEstimando exposiciones (restricción sum(beta)={VALOR_RESTRICCION}: "
          f"{'activada' if RESTRINGIR_SUMA else 'desactivada'}, factor base = {FACTOR_BASE})...")
    matriz = estimar_matriz_exposiciones(
        rendimientos=rendimientos,
        factores_df=factores_df,
        factores=FACTORES,
        factor_base=FACTOR_BASE,
        restringir_suma=RESTRINGIR_SUMA,
        valor_restriccion=VALOR_RESTRICCION,
        min_obs=MIN_OBS,
    )

    n_ok = matriz["r2"].notna().sum()
    print(f"  {n_ok} / {len(matriz)} empresas con exposiciones estimadas "
          f"(el resto no alcanzó el mínimo de {MIN_OBS} meses en común)")

    if RESTRINGIR_SUMA:
        error_suma = (matriz["suma_beta"] - VALOR_RESTRICCION).abs()
        peor = error_suma.max()
        print(f"  chequeo de la restricción sum(beta)={VALOR_RESTRICCION}: "
              f"error máximo {peor:.2e} (debe ser ~0)")

    bajo_r2 = matriz[matriz["r2"] < R2_AVISO]
    if len(bajo_r2):
        print(f"  AVISO: {len(bajo_r2)} empresas con R^2 < {R2_AVISO} "
              f"(exposiciones poco confiables): {', '.join(bajo_r2.index[:15])}"
              f"{' ...' if len(bajo_r2) > 15 else ''}")

    matriz.to_csv(SALIDA / "matriz_exposiciones.csv")

    definiciones = pd.DataFrame([
        ("Método", "Regresión MCO del rendimiento mensual de cada acción "
                   "(en exceso de rf) contra los factores, con restricción "
                   f"sum(beta) = {VALOR_RESTRICCION}"),
        ("Restricción", f"Impuesta despejando el beta de '{FACTOR_BASE}' y "
                        "sustituyéndolo en la ecuación (método exacto, sin "
                        "optimización numérica); ver estimar_exposiciones.py"),
        ("Factores", ", ".join(FACTORES)),
        ("Ventana", "Todo el historial disponible" if not VENTANA_MESES
                    else f"Últimos {VENTANA_MESES} meses"),
        ("Mínimo de observaciones", f"{MIN_OBS} meses con dato simultáneo en "
                                    "la acción y en todos los factores"),
        ("alpha", "Intercepto de la regresión (rendimiento no explicado por los factores)"),
        ("r2", "R^2 de la regresión; valores bajos indican exposiciones poco confiables"),
        ("n_obs", "Meses usados en la regresión de esa empresa"),
        ("suma_beta", f"Suma de los betas; debe ser ~{VALOR_RESTRICCION} si la restricción está activa"),
    ], columns=["concepto", "definición"])

    with pd.ExcelWriter(SALIDA / "matriz_exposiciones.xlsx", engine="openpyxl") as xw:
        matriz.reset_index().to_excel(xw, sheet_name="Exposiciones", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)
        if len(bajo_r2):
            bajo_r2.reset_index().to_excel(xw, sheet_name="Avisos_R2_bajo", index=False)

    print(f"\nMatriz de exposiciones guardada en {SALIDA}")


if __name__ == "__main__":
    main()