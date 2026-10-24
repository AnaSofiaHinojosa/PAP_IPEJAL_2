from pathlib import Path
import generar_rendimientos as gr

EXPOSICIONES = Path("salidas/exposiciones_sin_estandarizar/matriz_exposiciones.csv")
ESTANDARIZACION = Path("salidas/exposiciones_sin_estandarizar/factores_estandarizacion.csv")  # no debe existir
SALIDA = Path("salidas/rendimientos_esperados_sin_estandarizar")


def main():
    print("MODELO 3: rendimientos esperados con factores SIN estandarizar\n")
    if ESTANDARIZACION.exists():
        raise RuntimeError(
            f"Existe {ESTANDARIZACION}: la matriz del modelo 3 se generó con "
            "estandarización. Vuelve a correr generar_matriz_sin_estandarizar.py.")
    originales = (gr.EXPOSICIONES, gr.ESTANDARIZACION, gr.SALIDA)
    gr.EXPOSICIONES, gr.ESTANDARIZACION, gr.SALIDA = EXPOSICIONES, ESTANDARIZACION, SALIDA
    try:
        gr.main()
    finally:
        gr.EXPOSICIONES, gr.ESTANDARIZACION, gr.SALIDA = originales


if __name__ == "__main__":
    main()
