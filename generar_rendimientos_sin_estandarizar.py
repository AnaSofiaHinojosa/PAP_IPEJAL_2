"""
generar_rendimientos_sin_estandarizar.py - MODELO 3, rendimiento
esperado por acción con los factores sin estandarizar.

La lógica de cálculo es la de generar_rendimientos.py: solo se
le cambian las rutas para que lea la matriz del modelo 3 y escriba en carpetas
propias.

Como no se genera factores_estandarizacion.csv,
generar_rendimientos.py no lo encuentra y usa los factores crudos.
"AVISO: no encontré ... factores_estandarizacion.csv": en este modelo es lo
esperado. El rolling mean (3, 6 y 12 meses) se calcula entonces
sobre el rendimiento mensual real del factor, y E[R] queda en unidades de
rendimiento mensual.

Lee:
    salidas/exposiciones_sin_estandarizar/matriz_exposiciones.csv
    salidas/regresion/factores.csv

Escribe en salidas/rendimientos_esperados_sin_estandarizar/:
    rendimientos_esperados_ventana{3,6,12}.csv
    rendimientos_esperados.xlsx

Antes de correr: generar_matriz_sin_estandarizar.py
"""

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
