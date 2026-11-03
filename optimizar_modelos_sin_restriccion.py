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
