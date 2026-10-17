"""
generar_rendimientos_sin_restriccion.py - MODELO 2, Parte 3: rendimiento
esperado por acción usando las betas SIN restricción sum(beta)=1.

La lógica de cálculo es igual a la de generar_rendimientos.py: 
solo le cambia las rutas para que lea la matriz del
modelo 2 y escriba en carpetas propias, sin pisar los resultados del modelo 1.

Lee:
    salidas/exposiciones_sin_restriccion/matriz_exposiciones.csv
    salidas/exposiciones_sin_restriccion/factores_estandarizacion.csv
    salidas/regresion/factores.csv

Escribe en salidas/rendimientos_esperados_sin_restriccion/:
    rendimientos_esperados_ventana{3,6,12}.csv
    rendimientos_esperados.xlsx

Antes de correr: generar_matriz_sin_restriccion.py
"""

from pathlib import Path

import generar_rendimientos as gr

EXPOSICIONES = Path("salidas/exposiciones_sin_restriccion/matriz_exposiciones.csv")
ESTANDARIZACION = Path("salidas/exposiciones_sin_restriccion/factores_estandarizacion.csv")
SALIDA = Path("salidas/rendimientos_esperados_sin_restriccion")


def main():
    print("MODELO 2: rendimientos esperados con betas SIN restricción sum(beta)=1\n")
    gr.EXPOSICIONES = EXPOSICIONES
    gr.ESTANDARIZACION = ESTANDARIZACION
    gr.SALIDA = SALIDA
    gr.main()


if __name__ == "__main__":
    main()
