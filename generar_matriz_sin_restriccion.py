from pathlib import Path
import pandas as pd

import generar_matriz as gm
from estimar_exposiciones import estimar_matriz_exposiciones

SALIDA = Path("salidas/exposiciones_sin_restriccion")
MATRIZ_MODELO_1 = Path("salidas/exposiciones/matriz_exposiciones.csv")
RESTRINGIR_SUMA = False     # diferencia del modelo inicial


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("MODELO 2: exposiciones SIN restricción sum(beta)=1")
    print("Leyendo rendimientos y factores de la Parte 1...")
    archivo_rend = "rendimientos.csv" if gm.ENTRENAR_CON_HISTORIAL else "rendimientos_miembros.csv"
    print(f"  rendimientos de: {archivo_rend}")
    rendimientos = gm.cargar_rendimientos(gm.REGRESION / archivo_rend)
    factores_df = gm.cargar_factores(gm.REGRESION / "factores.csv")
    print(f"  {rendimientos.shape[1]} empresas, {rendimientos.shape[0]} meses de rendimientos")

    faltan = [f for f in gm.FACTORES if f not in factores_df.columns]
    if faltan:
        raise ValueError(f"Faltan factores en factores.csv: {faltan}")

    rendimientos, factores_df = gm.recortar_ventana(rendimientos, factores_df, gm.VENTANA_MESES)

    print(f"  estandarizar factores = {gm.ESTANDARIZAR}, estandarizar rendimientos = {gm.ESTANDARIZAR_Y}")
    print("\nEstimando exposiciones (MCO sin restricción de suma)...")
    matriz = estimar_matriz_exposiciones(
        rendimientos=rendimientos,
        factores_df=factores_df,
        factores=gm.FACTORES,
        factor_base=gm.FACTOR_BASE,          # no se usa 
        restringir_suma=RESTRINGIR_SUMA,
        valor_restriccion=gm.VALOR_RESTRICCION,
        min_obs=gm.MIN_OBS,
        estandarizar=gm.ESTANDARIZAR,
        estandarizar_y=gm.ESTANDARIZAR_Y,
    )

    n_ok = matriz["r2"].notna().sum()
    print(f"  {n_ok} / {len(matriz)} empresas con exposiciones estimadas "
          f"(el resto no alcanzó el mínimo de {gm.MIN_OBS} meses en común)")

    s = matriz["suma_beta"].dropna()
    print(f"  suma de betas (informativa, ya no es 1): "
          f"mín {s.min():.2f}  media {s.mean():.2f}  máx {s.max():.2f}")

    # Referencia contra el modelo inicial
    if MATRIZ_MODELO_1.exists():
        m1 = pd.read_csv(MATRIZ_MODELO_1, index_col="ticker")
        comunes = matriz.index.intersection(m1.index)
        print(f"  R^2 medio en {len(comunes)} empresas comunes: "
              f"modelo 1 = {m1.loc[comunes, 'r2'].mean():.3f}  |  "
              f"modelo 2 = {matriz.loc[comunes, 'r2'].mean():.3f}  "
              "(el modelo 2 debe ser >= por tener menos restricciones)")

    bajo_r2 = matriz[matriz["r2"] < gm.R2_AVISO]
    if len(bajo_r2):
        print(f"  AVISO: {len(bajo_r2)} empresas con R^2 < {gm.R2_AVISO} "
              f"(exposiciones poco confiables): {', '.join(bajo_r2.index[:15])}"
              f"{' ...' if len(bajo_r2) > 15 else ''}")

    matriz.to_csv(SALIDA / "matriz_exposiciones.csv")

    # media/desviación de estandarización: para Parte 3 
    estandarizacion = matriz.attrs.get("estandarizacion")
    if estandarizacion is not None:
        estandarizacion.rename_axis("factor").to_csv(SALIDA / "factores_estandarizacion.csv")
        print(f"  medias/desviaciones de estandarización guardadas en "
              f"{SALIDA / 'factores_estandarizacion.csv'}")

    definiciones = pd.DataFrame([
        ("Modelo", "Modelo 2: betas SIN restricción de suma"),
        ("Método", "Regresión MCO del rendimiento mensual de cada acción (en exceso de rf) "
                   "contra los factores, con las betas libres (sin sum(beta) = 1)"),
        ("Factores", ", ".join(gm.FACTORES)),
        ("Ventana", "Todo el historial disponible" if not gm.VENTANA_MESES
                    else f"Últimos {gm.VENTANA_MESES} meses"),
        ("Mínimo de observaciones", f"{gm.MIN_OBS} meses con dato simultáneo en "
                                    "la acción y en todos los factores"),
        ("Estandarización", ("Factores: z-score de cada factor con toda su serie. " if gm.ESTANDARIZAR else "Factores sin estandarizar. ")
                            + ("Rendimientos en exceso: z-score global" if gm.ESTANDARIZAR_Y
                               else "Rendimientos en exceso sin estandarizar")),
        ("Datos de entrenamiento", "Todo el historial de precios de cada empresa (rendimientos.csv)"
                                   if gm.ENTRENAR_CON_HISTORIAL else
                                   "Solo meses en el Top 100 (rendimientos_miembros.csv)"),
        ("alpha", "Intercepto de la regresión"),
        ("r2", "R^2 de la regresión"),
        ("n_obs", "Meses usados en la regresión de esa empresa"),
        ("suma_beta", "Suma de los betas; informativa (en este modelo NO se fuerza a 1)"),
    ], columns=["concepto", "definición"])

    with pd.ExcelWriter(SALIDA / "matriz_exposiciones.xlsx", engine="openpyxl") as xw:
        matriz.reset_index().to_excel(xw, sheet_name="Exposiciones", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)
        if len(bajo_r2):
            bajo_r2.reset_index().to_excel(xw, sheet_name="Avisos_R2_bajo", index=False)

    print(f"\nMatriz de exposiciones (modelo 2) guardada en {SALIDA}")


if __name__ == "__main__":
    main()
