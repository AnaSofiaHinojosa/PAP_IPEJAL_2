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
