from pathlib import Path

import generar_rendimientos as gr

EXPOSICIONES = Path(
    "salidas/exposiciones_sin_restriccion/matriz_exposiciones.csv")
ESTANDARIZACION = Path(
    "salidas/exposiciones_sin_restriccion/factores_estandarizacion.csv")
SALIDA = Path("salidas/rendimientos_esperados_sin_restriccion")


def main():
    print("MODELO 2: rendimientos esperados con betas SIN restricción sum(beta)=1\n")
    gr.EXPOSICIONES = EXPOSICIONES
    gr.ESTANDARIZACION = ESTANDARIZACION
    gr.SALIDA = SALIDA
    gr.main()


if __name__ == "__main__":
    main()
