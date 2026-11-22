"""
Parte 2: matriz de exposiciones (betas) de los 4 modelos del diseño 2x2.

Uso (desde la raíz del repo):
    python generar_matriz.py                 # corre los modelos de MODELOS_A_CORRER (por defecto, los 4)
    python generar_matriz.py --modelo 2      # solo el modelo 2
    python generar_matriz.py --modelo 1 3    # modelos 1 y 3

Sustituye a generar_matriz_sin_restriccion.py, generar_matriz_sin_estandarizar.py
y generar_matriz_sin_restriccion_sin_estandarizar.py. Cada modelo escribe en la
misma carpeta que antes, así que el resto del pipeline no cambia.
"""

import argparse
from pathlib import Path

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
SALIDA = Path("salidas/exposiciones")   # carpeta del modelo 1

# ---------------------------------------------------------------------------
# Parámetros comunes a los 4 modelos
# ---------------------------------------------------------------------------
# MERCADO, VALUE, TAMANO, MOMENTUM, VOLATILIDAD, LIQUIDEZ
FACTORES = FACTORES_DEFAULT
# factor que se despeja de la restricción de suma
FACTOR_BASE = FACTOR_BASE_DEFAULT
VALOR_RESTRICCION = VALOR_RESTRICCION_DEFAULT
# mínimo de meses en común para estimar una empresa
MIN_OBS = MIN_OBS_DEFAULT
# None = usa todo el historial disponible (20 años);
# o un entero, ej. 120, para usar solo los últimos N meses
VENTANA_MESES = None
# avisa si el R^2 de una empresa es menor a esto
R2_AVISO = 0.10
# True  = cada empresa se estima con TODO su historial de precios (rendimientos.csv),
#         aunque todavía no estuviera en el Top 100 (mismo criterio que optimizar_modelos.py).
# False = solo con los meses en que estuvo en el Top 100 (rendimientos_miembros.csv), como antes.
ENTRENAR_CON_HISTORIAL = True

# ---------------------------------------------------------------------------
# Valores del modelo 1 (base). Se dejan con estos nombres porque otros
# scripts pueden leerlos como generar_matriz.ESTANDARIZAR, etc.
# ---------------------------------------------------------------------------
RESTRINGIR_SUMA = True                    # sum(beta) = VALOR_RESTRICCION
# z-score por factor (toda su serie)
ESTANDARIZAR = ESTANDARIZAR_FACTORES_DEFAULT
# z-score global del exceso de rendimiento (una sola media/desv para TODAS las acciones).
# Necesario con factores estandarizados + sum(beta)=1: así ambos lados de la
# regresión quedan en desviaciones estándar (si no, R^2 ~ -14).
ESTANDARIZAR_Y = True

# ---------------------------------------------------------------------------
# Diseño 2x2: qué cambia en cada modelo
# ---------------------------------------------------------------------------
MODELOS = {
    1: {
        "nombre": "Modelo 1 (base): factores estandarizados, sum(beta)=1",
        "restringir_suma": RESTRINGIR_SUMA,
        "estandarizar": ESTANDARIZAR,
        "estandarizar_y": ESTANDARIZAR_Y,
        "salida": SALIDA,
    },
    2: {
        "nombre": "Modelo 2: factores estandarizados, SIN restricción sum(beta)=1",
        "restringir_suma": False,
        "estandarizar": ESTANDARIZAR,
        "estandarizar_y": ESTANDARIZAR_Y,
        "salida": Path("salidas/exposiciones_sin_restriccion"),
        "comparar_con": 1,      # misma escala, solo difiere en la restricción
    },
    3: {
        "nombre": "Modelo 3: factores SIN estandarizar, sum(beta)=1",
        "restringir_suma": True,
        "estandarizar": False,
        "estandarizar_y": False,
        "salida": Path("salidas/exposiciones_sin_estandarizar"),
    },
    4: {
        "nombre": "Modelo 4: factores SIN estandarizar, SIN restricción sum(beta)=1",
        "restringir_suma": False,
        "estandarizar": False,
        "estandarizar_y": False,
        "salida": Path("salidas/exposiciones_sin_restriccion_sin_estandarizar"),
        "comparar_con": 3,      # misma escala, solo difiere en la restricción
    },
}

# Modelos que se corren con "python generar_matriz.py" sin argumentos.
# El orden importa: el 2 compara su R^2 contra el 1, y el 4 contra el 3.
MODELOS_A_CORRER = [1, 2, 3, 4]


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


def definiciones_modelo(cfg: dict) -> pd.DataFrame:
    """Hoja 'Definiciones' del Excel, según la configuración del modelo."""
    restringir = cfg["restringir_suma"]
    filas = [
        ("Modelo", cfg["nombre"]),
        ("Método", "Regresión MCO del rendimiento mensual de cada acción "
                   "(en exceso de rf) contra los factores, "
                   + (f"con restricción sum(beta) = {VALOR_RESTRICCION}" if restringir
                      else f"con las betas libres (sin sum(beta) = {VALOR_RESTRICCION})")),
    ]
    if restringir:
        filas.append(("Restricción", f"Impuesta despejando el beta de '{FACTOR_BASE}' y "
                                     "sustituyéndolo en la ecuación (método exacto, sin "
                                     "optimización numérica); ver estimar_exposiciones.py"))
    filas += [
        ("Factores", ", ".join(FACTORES)),
        ("Ventana", "Todo el historial disponible" if not VENTANA_MESES
                    else f"Últimos {VENTANA_MESES} meses"),
        ("Mínimo de observaciones", f"{MIN_OBS} meses con dato simultáneo en "
                                    "la acción y en todos los factores"),
        ("Estandarización",
         "Ninguna: factores y rendimientos en exceso en unidades de rendimiento mensual"
         if not cfg["estandarizar"] and not cfg["estandarizar_y"] else
         ("Factores: z-score de cada factor con toda su serie. " if cfg["estandarizar"]
          else "Factores sin estandarizar. ")
         + ("Rendimientos en exceso: z-score global (una media y desv. para todas las acciones); "
            "los betas quedan en desv. del rendimiento por desv. del factor"
            if cfg["estandarizar_y"] else "Rendimientos en exceso sin estandarizar")),
        ("Datos de entrenamiento", "Todo el historial de precios de cada empresa (rendimientos.csv)"
                                   if ENTRENAR_CON_HISTORIAL else
                                   "Solo meses en el Top 100 (rendimientos_miembros.csv)"),
        ("alpha", "Intercepto de la regresión (rendimiento no explicado por los factores)"),
        ("r2", "R^2 de la regresión; valores bajos indican exposiciones poco confiables"),
        ("n_obs", "Meses usados en la regresión de esa empresa"),
        ("suma_beta",
         f"Suma de los betas; debe ser ~{VALOR_RESTRICCION} si la restricción está activa"
         if restringir else
         f"Suma de los betas; informativa (en este modelo NO se fuerza a {VALOR_RESTRICCION})"),
    ]
    return pd.DataFrame(filas, columns=["concepto", "definición"])


def correr_modelo(modelo: int) -> None:
    cfg = MODELOS[modelo]
    salida = cfg["salida"]
    restringir = cfg["restringir_suma"]
    salida.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 75}\n{cfg['nombre']}\n{'=' * 75}")

    # Sin estandarizar no se escribe factores_estandarizacion.csv. Si quedara
    # uno de una corrida anterior, la Parte 3 lo usaría y estandarizaría por error.
    viejo = salida / "factores_estandarizacion.csv"
    if not cfg["estandarizar"] and viejo.exists():
        viejo.unlink()
        print(f"  (se eliminó {viejo} de una corrida anterior)")

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

    print(f"  estandarizar factores = {cfg['estandarizar']}, "
          f"estandarizar rendimientos = {cfg['estandarizar_y']}")
    print(f"\nEstimando exposiciones (restricción sum(beta)={VALOR_RESTRICCION}: "
          f"{'activada' if restringir else 'desactivada'}"
          f"{f', factor base = {FACTOR_BASE}' if restringir else ''})...")
    matriz = estimar_matriz_exposiciones(
        rendimientos=rendimientos,
        factores_df=factores_df,
        factores=FACTORES,
        factor_base=FACTOR_BASE,              # solo se usa con restricción
        restringir_suma=restringir,
        valor_restriccion=VALOR_RESTRICCION,
        min_obs=MIN_OBS,
        estandarizar=cfg["estandarizar"],
        estandarizar_y=cfg["estandarizar_y"],
    )

    n_ok = matriz["r2"].notna().sum()
    print(f"  {n_ok} / {len(matriz)} empresas con exposiciones estimadas "
          f"(el resto no alcanzó el mínimo de {MIN_OBS} meses en común)")

    if restringir:
        error_suma = (matriz["suma_beta"] - VALOR_RESTRICCION).abs()
        print(f"  chequeo de la restricción sum(beta)={VALOR_RESTRICCION}: "
              f"error máximo {error_suma.max():.2e} (debe ser ~0)")
    else:
        s = matriz["suma_beta"].dropna()
        print(f"  suma de betas (informativa, ya no es {VALOR_RESTRICCION}): "
              f"mín {s.min():.2f}  media {s.mean():.2f}  máx {s.max():.2f}")

        # Referencia contra el modelo con restricción de la misma escala
        ref = cfg.get("comparar_con")
        matriz_ref = MODELOS[ref]["salida"] / "matriz_exposiciones.csv" if ref else None
        if matriz_ref is not None and matriz_ref.exists():
            mr = pd.read_csv(matriz_ref, index_col="ticker")
            comunes = matriz.index.intersection(mr.index)
            print(f"  R^2 medio en {len(comunes)} empresas comunes: "
                  f"modelo {ref} = {mr.loc[comunes, 'r2'].mean():.3f}  |  "
                  f"modelo {modelo} = {matriz.loc[comunes, 'r2'].mean():.3f}  "
                  f"(el modelo {modelo} debe ser >= por tener menos restricciones)")

    bajo_r2 = matriz[matriz["r2"] < R2_AVISO]
    if len(bajo_r2):
        print(f"  AVISO: {len(bajo_r2)} empresas con R^2 < {R2_AVISO} "
              f"(exposiciones poco confiables): {', '.join(bajo_r2.index[:15])}"
              f"{' ...' if len(bajo_r2) > 15 else ''}")

    matriz.to_csv(salida / "matriz_exposiciones.csv")

    # Guardar la media/desviación usadas para estandarizar los factores
    # (matriz.attrs se pierde al hacer to_csv, así que se guarda aparte).
    # La Parte 3 (generar_rendimientos.py) DEBE usar estos mismos valores
    # para que el factor esperado quede en la MISMA escala que los betas.
    estandarizacion = matriz.attrs.get("estandarizacion")
    if estandarizacion is not None:
        estandarizacion.rename_axis("factor").to_csv(
            salida / "factores_estandarizacion.csv")
        print(f"  medias/desviaciones de estandarización guardadas en "
              f"{salida / 'factores_estandarizacion.csv'} (la Parte 3 las necesita)")

    with pd.ExcelWriter(salida / "matriz_exposiciones.xlsx", engine="openpyxl") as xw:
        matriz.reset_index().to_excel(xw, sheet_name="Exposiciones", index=False)
        definiciones_modelo(cfg).to_excel(xw, sheet_name="Definiciones", index=False)
        if len(bajo_r2):
            bajo_r2.reset_index().to_excel(xw, sheet_name="Avisos_R2_bajo", index=False)

    print(f"\nMatriz de exposiciones ({cfg['nombre'].split(':')[0]}) guardada en {salida}")


def main(modelos: list[int] | None = None) -> None:
    for m in (modelos or MODELOS_A_CORRER):
        correr_modelo(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parte 2: matriz de exposiciones de los modelos del diseño 2x2.")
    parser.add_argument("--modelo", type=int, nargs="+", choices=sorted(MODELOS),
                        help="Modelo(s) a correr. Sin este argumento se corren los de MODELOS_A_CORRER.")
    args = parser.parse_args()
    main(args.modelo)