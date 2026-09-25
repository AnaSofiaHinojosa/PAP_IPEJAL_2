"""
estimar_rendimiento.py - Parte 3 (Rendimiento esperado por acción).

Dos pasos:

1) Rendimiento esperado de CADA FACTOR en cada mes t:
       paso_a  = rolling mean del factor sobre los últimos `ventana` meses
                 (3, 6 o 12), es decir el promedio de sus últimos cambios.
       paso_b  = ese rolling mean se estandariza con z-score, para que todos
                 los factores queden en la misma escala y se puedan comparar
                 / sumar entre sí.
       rezago  = por defecto, el rolling mean se recorre 1 mes hacia adelante
                 (shift) para que el rendimiento esperado del mes t NO use el
                 rendimiento del factor del propio mes t (evita look-ahead
                 bias: en la práctica, al inicio del mes t solo conoces los
                 rendimientos de los factores hasta t-1).

2) Rendimiento esperado de CADA ACCIÓN en cada mes t:
       E[R_i,t] = sum_k  beta_i,k * factor_esperado_k,t
   donde beta_i,k viene de la matriz de exposiciones (Parte 2, constante en
   el tiempo para cada acción) y factor_esperado_k,t es el resultado del
   paso 1 (varía mes a mes).

Requiere: numpy, pandas
"""

from pathlib import Path
import numpy as np
import pandas as pd

from estimar_exposiciones import FACTORES_DEFAULT

VENTANA_DEFAULT = 12          # 3, 6 o 12 meses
REZAGO_DEFAULT = 1            # meses de rezago para evitar look-ahead bias
METODO_Z_DEFAULT = "expandido"  # "expandido" (sin ver el futuro) o "completo"


def _zscore(serie: pd.Series, metodo: str, min_periods: int) -> pd.Series:
    """Estandariza una serie con z-score.

    "expandido": usa la media/desviación estándar acumulada HASTA cada fecha
        (expanding window) -> no usa información futura, más correcto para
        un ejercicio predictivo.
    "completo": usa la media/desviación estándar de TODA la muestra -> más
        simple, pero técnicamente usa información futura (útil solo para
        describir la serie histórica completa, no para "predecir" en tiempo
        real).
    """
    if metodo == "expandido":
        media = serie.expanding(min_periods=min_periods).mean()
        std = serie.expanding(min_periods=min_periods).std()
    elif metodo == "completo":
        media = serie.mean()
        std = serie.std()
    else:
        raise ValueError("metodo debe ser 'expandido' o 'completo'")
    return (serie - media) / std


def calcular_factor_esperado(
    factor: pd.Series,
    ventana: int = VENTANA_DEFAULT,
    rezago: int = REZAGO_DEFAULT,
    metodo_z: str = METODO_Z_DEFAULT,
) -> pd.Series:
    """
    Rendimiento esperado de UN factor en cada fecha: rolling mean de los
    últimos `ventana` meses, rezagado `rezago` meses, y estandarizado con
    z-score. Regresa una pd.Series con el mismo índice que `factor` (con NaN
    donde no hay suficiente historia todavía).
    """
    promedio_movil = factor.rolling(window=ventana, min_periods=ventana).mean()
    if rezago:
        promedio_movil = promedio_movil.shift(rezago)
    return _zscore(promedio_movil, metodo=metodo_z, min_periods=ventana)


def calcular_matriz_factores_esperados(
    factores_df: pd.DataFrame,
    factores: list[str] | None = None,
    ventana: int = VENTANA_DEFAULT,
    rezago: int = REZAGO_DEFAULT,
    metodo_z: str = METODO_Z_DEFAULT,
) -> pd.DataFrame:
    """
    Aplica calcular_factor_esperado() a cada factor. Regresa un DataFrame
    fecha x factor con el "rendimiento esperado" (z-score del rolling mean)
    de cada factor en cada mes.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    columnas = {
        f: calcular_factor_esperado(factores_df[f], ventana=ventana, rezago=rezago, metodo_z=metodo_z)
        for f in factores
    }
    return pd.DataFrame(columnas, index=factores_df.index)


def calcular_rendimiento_esperado(
    matriz_exposiciones: pd.DataFrame,
    factores_esperados: pd.DataFrame,
    factores: list[str] | None = None,
    incluir_alpha: bool = False,
) -> pd.DataFrame:
    """
    Rendimiento esperado de CADA ACCIÓN en cada fecha:
        E[R_i,t] = sum_k beta_i,k * factor_esperado_k,t   (+ alpha_i si
        incluir_alpha=True)

    Parámetros
    ----------
    matriz_exposiciones : pd.DataFrame
        Salida de la Parte 2 (estimar_matriz_exposiciones), indexada por
        ticker, con una columna por factor (y opcionalmente "alpha").
    factores_esperados : pd.DataFrame
        Salida de calcular_matriz_factores_esperados: fecha x factor.
    factores : list[str], opcional
        Qué factores usar (deben existir en ambos DataFrames). Por defecto
        FACTORES_DEFAULT.
    incluir_alpha : bool
        Si True, le suma el alpha de cada acción al rendimiento esperado.
        Por defecto False (el enunciado solo pide la suma de
        exposición * rendimiento esperado del factor).

    Regresa
    -------
    pd.DataFrame fecha x ticker con el rendimiento esperado de cada acción
    en cada mes. Queda NaN donde no hay suficiente historia (arranque del
    rolling mean) o donde la acción no tiene beta estimado en la Parte 2.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)

    betas = matriz_exposiciones[factores]                    # ticker x factor
    fe = factores_esperados[factores]                         # fecha x factor

    # (fecha x factor) @ (factor x ticker) -> (fecha x ticker)
    productos = fe.to_numpy() @ betas.to_numpy().T
    resultado = pd.DataFrame(productos, index=fe.index, columns=betas.index)

    if incluir_alpha and "alpha" in matriz_exposiciones.columns:
        resultado = resultado.add(matriz_exposiciones["alpha"], axis=1)

    return resultado.sort_index(axis=1)