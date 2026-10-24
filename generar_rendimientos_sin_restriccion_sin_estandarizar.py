from pathlib import Path

import generar_rendimientos as gr

BASE = Path("salidas/exposiciones_sin_restriccion_sin_estandarizar")
EXPOSICIONES = BASE / "matriz_exposiciones.csv"
ESTANDARIZACION = BASE / "factores_estandarizacion.csv"     # no debe existir
SALIDA = Path(
    "salidas/rendimientos_esperados_sin_restriccion_sin_estandarizar")


def main():
    print("MODELO 4: rendimientos esperados, betas SIN restricción y factores SIN estandarizar\n")
    if ESTANDARIZACION.exists():
        raise RuntimeError(
            f"Existe {ESTANDARIZACION}: la matriz del modelo 4 se generó con "
            "estandarización. Vuelve a correr generar_matriz_sin_restriccion_sin_estandarizar.py.")
    originales = (gr.EXPOSICIONES, gr.ESTANDARIZACION, gr.SALIDA)
    gr.EXPOSICIONES, gr.ESTANDARIZACION, gr.SALIDA = EXPOSICIONES, ESTANDARIZACION, SALIDA
    try:
        gr.main()
    finally:
        gr.EXPOSICIONES, gr.ESTANDARIZACION, gr.SALIDA = originales


if __name__ == "__main__":
    main()
