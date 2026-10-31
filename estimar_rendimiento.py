"""
estimar_rendimiento.py - Parte 3 (Rendimiento esperado por acción).

Dos pasos:

1) Rendimiento esperado de CADA FACTOR en cada mes t:
       El factor YA llega estandarizado (ver `aplicar_estandarizacion`, más
       abajo), con la MISMA media y desviación estándar que se usó en la
       Parte 2 para estimar los betas (archivo
       salidas/exposiciones/factores_estandarizacion.csv). Esto es clave:
       si los betas están calibrados en "cambio en rendimiento por 1
       desviación estándar del factor" (usando la media/desv de TODO el
       histórico), el factor esperado tiene que estar en esa MISMA escala
       para que beta * factor_esperado tenga sentido.

       Sobre ese factor ya estandarizado, se calcula:
       paso_a  = rolling mean de los últimos `ventana` meses (3, 6 o 12) del
                 factor estandarizado, es decir el promedio reciente de qué
                 tan arriba/abajo de su media histórica ha estado el factor.
       rezago  = por defecto, el rolling mean se recorre 1 mes hacia adelante
                 (shift) para que el rendimiento esperado del mes t NO use el
                 valor del propio mes t (evita look-ahead bias).

       OJO: a diferencia de versiones anteriores de este archivo, aquí NO se
       le vuelve a sacar z-score al rolling mean. Si ya se estandarizó el
       factor antes de promediarlo, estandarizarlo otra vez después de
       promediarlo cambia la escala (la desviación estándar de un promedio
       móvil de 12 meses es más chica que la del factor mensual original) y
       rompe la consistencia con los betas de la Parte 2.

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


def aplicar_estandarizacion(
    factores_df: pd.DataFrame,
    factores: list[str] | None,
    stats: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Estandariza factores_df con la media/desviación de `stats` (tal como se
    guardó en salidas/exposiciones/factores_estandarizacion.csv, Parte 2).

    Si `stats` es None (en la Parte 2 se corrió con ESTANDARIZAR=False),
    regresa los factores tal cual (sin transformar) — así el factor esperado
    queda en la misma escala (cruda) que usaron los betas en ese caso.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    f = factores_df[factores]
    if stats is None:
        return f
    faltan = [k for k in factores if k not in stats.index]
    if faltan:
        raise ValueError(f"Factores sin media/desviación guardada: {faltan}")
    return (f - stats.loc[factores, "media"]) / stats.loc[factores, "desv"]


def calcular_factor_esperado(
    factor_estandarizado: pd.Series,
    ventana: int = VENTANA_DEFAULT,
    rezago: int = REZAGO_DEFAULT,
) -> pd.Series:
    """
    Rendimiento esperado de UN factor en cada fecha: rolling mean de los
    últimos `ventana` meses del factor YA estandarizado, rezagado `rezago`
    meses. (Ya NO se estandariza otra vez aquí — ver docstring del módulo.)
    """
    promedio_movil = factor_estandarizado.rolling(window=ventana, min_periods=ventana).mean()
    if rezago:
        promedio_movil = promedio_movil.shift(rezago)
    return promedio_movil


def calcular_matriz_factores_esperados(
    factores_estandarizados: pd.DataFrame,
    factores: list[str] | None = None,
    ventana: int = VENTANA_DEFAULT,
    rezago: int = REZAGO_DEFAULT,
) -> pd.DataFrame:
    """
    Aplica calcular_factor_esperado() a cada factor (ya estandarizado, ver
    aplicar_estandarizacion). Regresa un DataFrame fecha x factor con el
    rendimiento esperado de cada factor en cada mes.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    columnas = {
        f: calcular_factor_esperado(factores_estandarizados[f], ventana=ventana, rezago=rezago)
        for f in factores
    }
    return pd.DataFrame(columnas, index=factores_estandarizados.index)


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
        Salida de calcular_matriz_factores_esperados: fecha x factor, YA en
        la misma escala (estandarizada) que se usó para ajustar los betas.
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