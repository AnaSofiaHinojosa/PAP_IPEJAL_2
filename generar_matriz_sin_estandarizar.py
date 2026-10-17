"""
generar_matriz_sin_estandarizar.py - MODELO 3: matriz de exposiciones
con los factores sin estandarizar (sin z-score).

Es generar_matriz.py con dos cambios:
    ESTANDARIZAR   = False  -> los factores entran crudos (rendimiento mensual
                               del portafolio largo-corto), no (x - media)/desv
    ESTANDARIZAR_Y = False  -> el rendimiento en exceso de las acciones tampoco
                               se estandariza, para que ambos lados de la
                               regresión queden en la misma escala (rendimiento
                               mensual). Si solo uno se estandariza, con
                               sum(beta)=1 la regresión queda mal escalada.

La restricción sum(beta) = 1 se mantiene (igual que el modelo 1).

La lógica no se duplica, se reutiliza el main() de generar_matriz.py cambiando
sus banderas y su carpeta de salida. Todo lo demás (factores, FACTOR_BASE,
MIN_OBS, ventana, datos de entrenamiento) se toma de generar_matriz.py.

Como no hay estandarización, no se genera factores_estandarizacion.csv; la
Parte 3 interpreta esa ausencia como "usar los factores crudos".

Escribe en salidas/exposiciones_sin_estandarizar/  (NO toca salidas/exposiciones/)
"""

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
        gm.SALIDA, gm.ESTANDARIZAR, gm.ESTANDARIZAR_Y = originales   # deja generar_matriz como estaba


if __name__ == "__main__":
    main()
