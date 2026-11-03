from pathlib import Path
import numpy as np
import pandas as pd

from estimar_exposiciones import (
    estimar_matriz_exposiciones,
    FACTORES_DEFAULT,
    FACTOR_BASE_DEFAULT,
    MIN_OBS_DEFAULT,
    VALOR_RESTRICCION_DEFAULT,
    ESTANDARIZAR_FACTORES_DEFAULT,
    ESTANDARIZAR_Y_DEFAULT,
)

REGRESION = Path("salidas/regresion")
SALIDA = Path("salidas/exposiciones")

# Parámetros ajustables por el equipo
# MERCADO, VALUE, TAMANO, MOMENTUM, VOLATILIDAD, LIQUIDEZ
FACTORES = FACTORES_DEFAULT
# factor que se despeja de la restricción de suma
FACTOR_BASE = FACTOR_BASE_DEFAULT
RESTRINGIR_SUMA = True                    # sum(beta) = VALOR_RESTRICCION
VALOR_RESTRICCION = VALOR_RESTRICCION_DEFAULT
# mínimo de meses en común para estimar una empresa
MIN_OBS = MIN_OBS_DEFAULT
# None = usa todo el historial disponible (20 años);
VENTANA_MESES = None
# o un entero, ej. 120, para usar solo los últimos N meses
# avisa si el R^2 de una empresa es menor a esto
R2_AVISO = 0.10
# z-score por factor (toda su serie)
ESTANDARIZAR = ESTANDARIZAR_FACTORES_DEFAULT
# z-score global del exceso de rendimiento
ESTANDARIZAR_Y = True
# (una sola media/desv para TODAS las acciones).
# Necesario con factores estandarizados + sum(beta)=1:
# así ambos lados de la regresión quedan en
# desviaciones estándar (si no, R^2 ~ -14).
# True  = cada empresa se estima con TODO su historial de precios (rendimientos.csv),
#         aunque todavía no estuviera en el Top 100 (mismo criterio que optimizar_modelos.py).
# False = solo con los meses en que estuvo en el Top 100 (rendimientos_miembros.csv), como antes.
ENTRENAR_CON_HISTORIAL = True


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
    archivo_rend = "rendimientos.csv" if ENTRENAR_CON_HISTORIAL else "rendimientos_miembros.csv"
    print(f"  rendimientos de: {archivo_rend}")
    rendimientos = cargar_rendimientos(REGRESION / archivo_rend)
    factores_df = cargar_factores(REGRESION / "factores.csv")
    print(
        f"  {rendimientos.shape[1]} empresas, {rendimientos.shape[0]} meses de rendimientos")
    print(f"  factores disponibles: {', '.join(factores_df.columns)}")

    faltan = [f for f in FACTORES if f not in factores_df.columns]
    if faltan:
        raise ValueError(f"Faltan factores en factores.csv: {faltan}")

    rendimientos, factores_df = recortar_ventana(
        rendimientos, factores_df, VENTANA_MESES)

    print(
        f"  estandarizar factores = {ESTANDARIZAR}, estandarizar rendimientos = {ESTANDARIZAR_Y}")
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
        estandarizar=ESTANDARIZAR,
        estandarizar_y=ESTANDARIZAR_Y,
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

    # Guardar la media/desviación usadas para estandarizar los factores
    # (matriz.attrs se pierde al hacer to_csv, así que se guarda aparte).
    # La Parte 3 (generar_rendimientos.py) DEBE usar estos mismos valores
    # para que el factor esperado quede en la MISMA escala que los betas.
    estandarizacion = matriz.attrs.get("estandarizacion")
    if estandarizacion is not None:
        estandarizacion.rename_axis("factor").to_csv(
            SALIDA / "factores_estandarizacion.csv")
        print(f"  medias/desviaciones de estandarización guardadas en "
              f"{SALIDA / 'factores_estandarizacion.csv'} (la Parte 3 las necesita)")

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
        ("Estandarización", ("Factores: z-score de cada factor con toda su serie. " if ESTANDARIZAR else "Factores sin estandarizar. ")
         + ("Rendimientos en exceso: z-score global (una media y desv. para todas las acciones); "
            "los betas quedan en desv. del rendimiento por desv. del factor"
            if ESTANDARIZAR_Y else "Rendimientos en exceso sin estandarizar")),
        ("Datos de entrenamiento", "Todo el historial de precios de cada empresa (rendimientos.csv)"
                                   if ENTRENAR_CON_HISTORIAL else
                                   "Solo meses en el Top 100 (rendimientos_miembros.csv)"),
        ("alpha", "Intercepto de la regresión (rendimiento no explicado por los factores)"),
        ("r2", "R^2 de la regresión; valores bajos indican exposiciones poco confiables"),
        ("n_obs", "Meses usados en la regresión de esa empresa"),
        ("suma_beta",
         f"Suma de los betas; debe ser ~{VALOR_RESTRICCION} si la restricción está activa"),
    ], columns=["concepto", "definición"])

    with pd.ExcelWriter(SALIDA / "matriz_exposiciones.xlsx", engine="openpyxl") as xw:
        matriz.reset_index().to_excel(xw, sheet_name="Exposiciones", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)
        if len(bajo_r2):
            bajo_r2.reset_index().to_excel(xw, sheet_name="Avisos_R2_bajo", index=False)

    print(f"\nMatriz de exposiciones guardada en {SALIDA}")


if __name__ == "__main__":
    main()
