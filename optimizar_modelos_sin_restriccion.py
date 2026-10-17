"""
optimizar_modelos_sin_restriccion.py - MODELO 2: backtesting walk-forward,
IC y selección, con las betas sin restricción sum(beta)=1.

Corre el mismo proceso de optimizar_modelos.py (misma rejilla de
36 modelos, mismas fechas comunes, mismo criterio de selección, mismas
estandarizaciones) pero cada vez que optimizar_modelos
llama a backtest_modelo(), se le agrega restringir_suma=False, valor que
baja por backtesting.py hasta estimar_matriz_exposiciones().

No se modifica ningún archivo existente: se importa optimizar_modelos, se
cambia su carpeta de salida y se envuelve backtest_modelo temporalmente.

Lee (igual que el modelo 1):
    salidas/regresion/rendimientos.csv
    salidas/regresion/rendimientos_miembros.csv
    salidas/regresion/factores.csv

Escribe en salidas/backtesting_sin_restriccion/  (NO toca salidas/backtesting/):
    comparacion_modelos.csv / .xlsx
    mejor_modelo_ic_serie.csv
    mejor_modelo_rendimiento_esperado.csv
    mejor_modelo_exposiciones/
"""

from pathlib import Path

import optimizar_modelos as om

SALIDA_SIN_RESTRICCION = Path("salidas/backtesting_sin_restriccion")

_backtest_original = om.backtest_modelo


def _backtest_sin_restriccion(*args, **kwargs):
    """Igual que backtest_modelo, pero con las betas sin forzar suma = 1."""
    kwargs["restringir_suma"] = False
    return _backtest_original(*args, **kwargs)


def main():
    print("MODELO 2: backtesting con betas SIN restricción sum(beta)=1\n")
    om.SALIDA = SALIDA_SIN_RESTRICCION
    om.backtest_modelo = _backtest_sin_restriccion
    try:
        om.main()
    finally:
        om.backtest_modelo = _backtest_original   # deja optimizar_modelos como estaba


if __name__ == "__main__":
    main()
