"""
Parte 3: rendimiento esperado mensual de cada acción, para los 4 modelos del diseño 2x2.

Uso (desde la raíz del repo):
    python generar_rendimientos.py                 # corre los modelos de MODELOS_A_CORRER (por defecto, los 4)
    python generar_rendimientos.py --modelo 3      # solo el modelo 3
    python generar_rendimientos.py --modelo 1 3    # modelos 1 y 3

Sustituye a generar_rendimientos_sin_restriccion.py, generar_rendimientos_sin_estandarizar.py
y generar_rendimientos_sin_restriccion_sin_estandarizar.py. Cada modelo lee la matriz que
escribió generar_matriz.py y escribe en la misma carpeta que antes.
"""

import argparse
from pathlib import Path

import pandas as pd

import generar_matriz as gm
from estimar_exposiciones import FACTORES_DEFAULT
from estimar_rendimiento import (
    aplicar_estandarizacion,
    calcular_matriz_factores_esperados,
    calcular_rendimiento_esperado,
    REZAGO_DEFAULT,
)

FACTORES_CSV = Path("salidas/regresion/factores.csv")

# Parámetros ajustables por el equipo
FACTORES = FACTORES_DEFAULT
# meses del rolling mean; el enunciado pide probar estas 3
VENTANAS = [3, 6, 12]
# 1 = evita look-ahead bias; poner 0 si el profe pide sin rezago
REZAGO_MESES = REZAGO_DEFAULT
# el enunciado solo pide sum(beta_k * factor_esperado_k)
INCLUIR_ALPHA = False

# Carpeta de salida de cada modelo. La carpeta de la matriz (entrada) y si el
# modelo estandariza se toman de generar_matriz.MODELOS, para no repetirlos.
SALIDAS = {
    1: Path("salidas/rendimientos_esperados"),
    2: Path("salidas/rendimientos_esperados_sin_restriccion"),
    3: Path("salidas/rendimientos_esperados_sin_estandarizar"),
    4: Path("salidas/rendimientos_esperados_sin_restriccion_sin_estandarizar"),
}

# Modelos que se corren con "python generar_rendimientos.py" sin argumentos.
MODELOS_A_CORRER = [1, 2, 3, 4]


def correr_modelo(modelo: int) -> None:
    cfg = gm.MODELOS[modelo]
    exposiciones = cfg["salida"] / "matriz_exposiciones.csv"
    estandarizacion = cfg["salida"] / "factores_estandarizacion.csv"
    salida = SALIDAS[modelo]

    print(f"\n{'=' * 75}\n{cfg['nombre']}\n{'=' * 75}")

    # En un modelo sin estandarizar, este archivo no debe existir: si existe,
    # la matriz se generó con estandarización y el E[R] saldría en otra escala.
    if not cfg["estandarizar"] and estandarizacion.exists():
        raise RuntimeError(
            f"Existe {estandarizacion}: la matriz del modelo {modelo} se generó con "
            f"estandarización. Vuelve a correr: python generar_matriz.py --modelo {modelo}")

    salida.mkdir(parents=True, exist_ok=True)

    print("Leyendo matriz de exposiciones (Parte 2) y factores (Parte 1)...")
    print(f"  matriz de: {exposiciones}")
    matriz_exposiciones = pd.read_csv(exposiciones, index_col="ticker")
    factores_df = pd.read_csv(FACTORES_CSV)
    factores_df["fecha"] = pd.to_datetime(factores_df["fecha"])
    factores_df = factores_df.set_index("fecha").sort_index()

    faltan = [f for f in FACTORES if f not in factores_df.columns]
    if faltan:
        raise ValueError(f"Faltan factores en factores.csv: {faltan}")

    if estandarizacion.exists():
        stats = pd.read_csv(estandarizacion, index_col="factor")
        print(f"  usando media/desviación de la Parte 2 ({estandarizacion}) "
              "para estandarizar los factores antes del rolling mean")
    elif cfg["estandarizar"]:
        stats = None
        print(f"  AVISO: no encontré {estandarizacion} -> se usan los factores "
              "crudos (sin estandarizar), aunque este modelo debería estandarizarlos. "
              f"Vuelve a correr: python generar_matriz.py --modelo {modelo}")
    else:
        stats = None
        print("  factores crudos (este modelo no estandariza)")
    factores_estandarizados = aplicar_estandarizacion(
        factores_df, FACTORES, stats)

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
        print(f"\nVentana de {ventana} meses (rezago={REZAGO_MESES})...")
        factores_esperados = calcular_matriz_factores_esperados(
            factores_estandarizados, factores=FACTORES, ventana=ventana,
            rezago=REZAGO_MESES,
        )
        rendimiento_esperado = calcular_rendimiento_esperado(
            matriz_exposiciones, factores_esperados,
            factores=FACTORES, incluir_alpha=INCLUIR_ALPHA,
        )

        n_meses_validos = rendimiento_esperado.notna().any(axis=1).sum()
        cobertura = rendimiento_esperado.notna().mean().mean()
        print(
            f"  {n_meses_validos} meses con al menos una acción con rendimiento esperado")
        print(f"  cobertura promedio (acción-mes con dato): {cobertura:.1%}")

        archivo = salida / f"rendimientos_esperados_ventana{ventana}.csv"
        rendimiento_esperado.to_csv(archivo)
        hojas_excel[f"ventana_{ventana}m"] = rendimiento_esperado
        resumen_filas.append({
            "ventana_meses": ventana,
            "meses_con_dato": n_meses_validos,
            "cobertura_promedio": cobertura,
            "primera_fecha_valida": rendimiento_esperado.dropna(how="all").index.min(),
        })

    resumen = pd.DataFrame(resumen_filas)
    with pd.ExcelWriter(salida / "rendimientos_esperados.xlsx", engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="Resumen", index=False)
        for nombre, df in hojas_excel.items():
            df.reset_index().rename(columns={"index": "fecha"}).to_excel(
                xw, sheet_name=nombre, index=False)

    print(f"\nRendimientos esperados (modelo {modelo}) guardados en {salida}")


def main(modelos: list[int] | None = None) -> None:
    for m in (modelos or MODELOS_A_CORRER):
        correr_modelo(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parte 3: rendimiento esperado de los modelos del diseño 2x2.")
    parser.add_argument("--modelo", type=int, nargs="+", choices=sorted(SALIDAS),
                        help="Modelo(s) a correr. Sin este argumento se corren los de MODELOS_A_CORRER.")
    args = parser.parse_args()
    main(args.modelo)
