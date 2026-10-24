from pathlib import Path

import generar_matriz as gm

SALIDA = Path("salidas/exposiciones_sin_estandarizar")
ESTANDARIZAR = False        # <- la diferencia del modelo 1
ESTANDARIZAR_Y = False


def main():
    print("MODELO 3: exposiciones con factores SIN estandarizar (sum(beta)=1 se mantiene)\n")

    # Si hubiera un archivo de estandarización de una corrida anterior en esta
    # carpeta, la Parte 3 lo usaría por error; se elimina.
    viejo = SALIDA / "factores_estandarizacion.csv"
    if viejo.exists():
        viejo.unlink()
        print(f"  (se eliminó {viejo} de una corrida anterior)")

    originales = (gm.SALIDA, gm.ESTANDARIZAR, gm.ESTANDARIZAR_Y)
    gm.SALIDA, gm.ESTANDARIZAR, gm.ESTANDARIZAR_Y = SALIDA, ESTANDARIZAR, ESTANDARIZAR_Y
    try:
        gm.main()
    finally:
        # deja generar_matriz como estaba
        gm.SALIDA, gm.ESTANDARIZAR, gm.ESTANDARIZAR_Y = originales


if __name__ == "__main__":
    main()
