"""
generar_rendimientos_sin_restriccion_sin_estandarizar.py - MODELO 4:
rendimiento esperado por acción con betas sin restricción sum(beta)=1 y
factores sin estandarizar.

La lógica de cálculo es la de generar_rendimientos.py: solo se
le cambian las rutas para que lea la matriz del modelo 4 y escriba en carpetas
propias.

Como la Parte 2 del modelo 4 no genera factores_estandarizacion.csv,
generar_rendimientos.py no lo encuentra y usa los factores crudos. 

Lee:
    salidas/exposiciones_sin_restriccion_sin_estandarizar/matriz_exposiciones.csv
    salidas/regresion/factores.csv

Escribe en salidas/rendimientos_esperados_sin_restriccion_sin_estandarizar/:
    rendimientos_esperados_ventana{3,6,12}.csv
    rendimientos_esperados.xlsx

Antes de correr: generar_matriz_sin_restriccion_sin_estandarizar.py
"""

from pathlib import Path

import generar_rendimientos as gr

BASE = Path("salidas/exposiciones_sin_restriccion_sin_estandarizar")
EXPOSICIONES = BASE / "matriz_exposiciones.csv"
ESTANDARIZACION = BASE / "factores_estandarizacion.csv"     # no debe existir
SALIDA = Path("salidas/rendimientos_esperados_sin_restriccion_sin_estandarizar")


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
