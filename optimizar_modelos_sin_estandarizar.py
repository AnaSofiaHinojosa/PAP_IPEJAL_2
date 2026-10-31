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
