"""
optimizar_modelos_sin_restriccion_sin_estandarizar.py - MODELO 4: backtesting
walk-forward, IC y selección, con betas sin restricción sum(beta)=1 y factores
sin estandarizar.

Corre el mismo proceso de optimizar_modelos.py (misma rejilla de 36 modelos,
mismas fechas comunes, mismo criterio de selección) con tres cambios:
    restringir_suma = False  -> se agrega a cada llamada de backtest_modelo()
                                (igual que en el modelo 2)
    ESTANDARIZAR    = False  -> factores crudos en cada corte (como el modelo 3)
    ESTANDARIZAR_Y  = False  -> rendimiento en exceso crudo (como el modelo 3)

Se importa optimizar_modelos, se cambian sus banderas, su carpeta de salida y 
se envuelve backtest_modelo temporalmente; al terminar se restaura todo.

Escribe en salidas/backtesting_sin_restriccion_sin_estandarizar/:
    comparacion_modelos.csv / .xlsx
    mejor_modelo_ic_serie.csv
    mejor_modelo_rendimiento_esperado.csv
    mejor_modelo_exposiciones/
"""

from pathlib import Path

import optimizar_modelos as om

SALIDA_MODELO_4 = Path("salidas/backtesting_sin_restriccion_sin_estandarizar")
ESTANDARIZAR = False
ESTANDARIZAR_Y = False

_backtest_original = om.backtest_modelo


def _backtest_sin_restriccion(*args, **kwargs):
    """Igual que backtest_modelo, pero con las betas sin forzar suma = 1."""
    kwargs["restringir_suma"] = False
    return _backtest_original(*args, **kwargs)


def main():
    print("MODELO 4: backtesting con betas SIN restricción y factores SIN estandarizar\n")
    originales = (om.SALIDA, om.ESTANDARIZAR, om.ESTANDARIZAR_Y)
    om.SALIDA, om.ESTANDARIZAR, om.ESTANDARIZAR_Y = SALIDA_MODELO_4, ESTANDARIZAR, ESTANDARIZAR_Y
    om.backtest_modelo = _backtest_sin_restriccion
    try:
        om.main()
    finally:
        om.backtest_modelo = _backtest_original
        om.SALIDA, om.ESTANDARIZAR, om.ESTANDARIZAR_Y = originales


if __name__ == "__main__":
    main()
