"""
optimizar_modelos_sin_estandarizar.py - MODELO 3: backtesting walk-forward,
IC y selección, con los factores sin estandarizar.

Corre el mismo proceso de optimizar_modelos.py (misma rejilla de 36 modelos,
mismas fechas comunes, mismo criterio de selección) con dos cambios:
    ESTANDARIZAR   = False  -> factores crudos en cada corte (sin z-score)
    ESTANDARIZAR_Y = False  -> rendimiento en exceso crudo (misma escala que
                               los factores; ver generar_matriz_sin_estandarizar.py)

La restricción sum(beta) = 1 se mantiene(igual que el modelo 1).

En cada corte, backtesting.py ya sabe qué hacer sin estandarización: no calcula
media/desviación y el factor esperado (rolling mean) se obtiene del factor
crudo. Se importa optimizar_modelos,
se cambian sus banderas y su carpeta de salida, y al terminar se restauran.

Escribe en salidas/backtesting_sin_estandarizar/  (NO toca salidas/backtesting/):
    comparacion_modelos.csv / .xlsx
    mejor_modelo_ic_serie.csv
    mejor_modelo_rendimiento_esperado.csv
    mejor_modelo_exposiciones/
"""

from pathlib import Path

import optimizar_modelos as om

SALIDA_SIN_ESTANDARIZAR = Path("salidas/backtesting_sin_estandarizar")
ESTANDARIZAR = False        # <- la diferencia del modelo 1
ESTANDARIZAR_Y = False


def main():
    print("MODELO 3: backtesting con factores SIN estandarizar (sum(beta)=1 se mantiene)\n")
    originales = (om.SALIDA, om.ESTANDARIZAR, om.ESTANDARIZAR_Y)
    om.SALIDA, om.ESTANDARIZAR, om.ESTANDARIZAR_Y = SALIDA_SIN_ESTANDARIZAR, ESTANDARIZAR, ESTANDARIZAR_Y
    try:
        om.main()
    finally:
        om.SALIDA, om.ESTANDARIZAR, om.ESTANDARIZAR_Y = originales


if __name__ == "__main__":
    main()
