"""
backtesting.py - Parte 4, Parte 1 (Backtesting) y Parte 2 (IC y selección).

DISEÑO (para que sea walk-forward de verdad, sin look-ahead bias):

1) RE-ESTIMACIÓN DE BETAS POR VENTANA MÓVIL
   En vez de un solo beta fijo (todo el histórico, como en la Parte 2), aquí
   los betas se recalculan periódicamente usando SOLO datos hasta la fecha de
   corte:
       - `ventana_beta = None`  -> usa TODO el histórico disponible hasta esa
         fecha (expanding window; equivale a repetir la Parte 2 en cada
         corte, con más datos cada vez).
       - `ventana_beta = N`     -> usa solo los últimos N meses antes de esa
         fecha (rolling window fijo; permite comparar "todo el histórico" vs.
         "los últimos 12/36/60 meses", que es justo lo que pide el enunciado).
   `frecuencia_meses` controla cada cuántos meses se vuelve a correr la
   regresión (por defecto 12 = una vez al año), para no recalcular betas
   cada mes (ni computacionalmente necesario ni realista: nadie recalibra
   un modelo de factores todos los meses en la práctica).
   Los betas estimados en la fecha de corte `c` se usan para predecir TODOS
   los meses posteriores a `c` (hasta el siguiente corte) — nunca para
   predecir `c` o meses anteriores.

2) RENDIMIENTO ESPERADO WALK-FORWARD
   Para cada mes t: E[R_i,t] = sum_k beta_i,k(corte aplicable) *
   factor_esperado_k,t, donde beta_i,k viene del corte más reciente ANTES de
   t, y factor_esperado_k,t es el z-score del rolling mean del factor
   (Parte 3, también construido solo con información hasta t-1).

   CONSISTENCIA DE ESCALA (factor esperado vs. betas): en cada fecha de
   corte, los betas se ajustan contra el factor estandarizado CON LA MEDIA Y
   DESVIACIÓN DE ESE MISMO CORTE (solo datos hasta esa fecha — nunca con
   estadísticas calculadas con datos futuros). El factor esperado que se usa
   para predecir los meses siguientes a ese corte se construye con ESA MISMA
   media/desviación, no con una global ni con una recalculada cada mes. Así,
   para cada tramo entre dos cortes, beta y factor esperado están en la
   misma escala.

3) COEFICIENTE DE INFORMACIÓN (IC)
   Para cada mes t, correlación de Spearman (rank correlation) entre el
   rendimiento esperado de cada acción (ex ante, calculado con info hasta
   t-1) y su rendimiento real observado en t (ex post):
       IC_t = spearman( E[R_.,t] , R_.,t )
   Se arma una serie de tiempo de IC_t, y se resume con:
       IC medio, desviación estándar del IC, IC Ratio = IC medio / std(IC)
       (la versión del Information Ratio para el IC: qué tan grande es la
       señal promedio relativo a qué tan ruidosa/inestable es en el tiempo),
       t-stat, y hit rate (% de meses con IC > 0).

Requiere: numpy, pandas, scipy
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from estimar_exposiciones import (
    estimar_matriz_exposiciones,
    estandarizar_factores,
    FACTORES_DEFAULT,
    MIN_OBS_DEFAULT,
    ESTANDARIZAR_FACTORES_DEFAULT,
)
from estimar_rendimiento import calcular_factor_esperado

FRECUENCIA_DEFAULT = 12     # meses entre re-estimaciones de beta
MIN_ACCIONES_IC = 10        # mínimo de acciones con dato simultáneo para calcular IC ese mes


# --------------------------------------------------------------------- #
# 1) Re-estimación de betas por ventana móvil (walk-forward)
# --------------------------------------------------------------------- #
def construir_exposiciones_rolling(
    rendimientos: pd.DataFrame,
    factores_df: pd.DataFrame,
    factores: list[str] | None = None,
    ventana_beta: int | None = None,
    min_obs: int = MIN_OBS_DEFAULT,
    frecuencia_meses: int = FRECUENCIA_DEFAULT,
    estandarizar: bool = ESTANDARIZAR_FACTORES_DEFAULT,
    **kwargs_exposiciones,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[pd.Timestamp, pd.DataFrame | None]]:
    """
    Corre estimar_matriz_exposiciones() en varias fechas de corte a lo largo
    del tiempo, cada una usando solo datos HASTA esa fecha (y, si
    ventana_beta no es None, solo los últimos `ventana_beta` meses).

    Si estandarizar=True, en cada corte también se calcula (con
    estandarizar_factores) la media/desviación de ESE corte — solo con los
    datos disponibles hasta esa fecha, nunca con datos futuros — para poder
    construir después un factor esperado en la misma escala que esos betas.

    Regresa: (exposiciones_por_fecha, estadisticas_por_fecha)
        exposiciones_por_fecha   dict {fecha_corte: matriz_exposiciones}
        estadisticas_por_fecha   dict {fecha_corte: tabla media/desv, o None
                                  si estandarizar=False}
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    fechas = factores_df.index.sort_values()

    exposiciones = {}
    estadisticas = {}
    for i, fecha_corte in enumerate(fechas):
        if i % frecuencia_meses != 0:
            continue
        if ventana_beta is None:
            rend_v = rendimientos.loc[:fecha_corte]
            fact_v = factores_df.loc[:fecha_corte]
        else:
            inicio = fecha_corte - pd.DateOffset(months=ventana_beta)
            rend_v = rendimientos.loc[inicio:fecha_corte]
            fact_v = factores_df.loc[inicio:fecha_corte]

        if len(fact_v) < min_obs:
            continue  # todavía no hay suficiente historia para este corte

        matriz = estimar_matriz_exposiciones(
            rend_v, fact_v, factores=factores, min_obs=min_obs,
            estandarizar=estandarizar, **kwargs_exposiciones,
        )
        exposiciones[fecha_corte] = matriz
        estadisticas[fecha_corte] = matriz.attrs.get("estandarizacion")  # None si estandarizar=False

    return exposiciones, estadisticas


def _beta_aplicable(fecha: pd.Timestamp, cortes_ordenados: list[pd.Timestamp]) -> pd.Timestamp | None:
    """El corte más reciente ESTRICTAMENTE ANTERIOR a `fecha` (sin look-ahead)."""
    anteriores = [c for c in cortes_ordenados if c < fecha]
    return max(anteriores) if anteriores else None


# --------------------------------------------------------------------- #
# 2) Rendimiento esperado walk-forward
# --------------------------------------------------------------------- #
def _factor_esperado_por_corte(
    factores_df: pd.DataFrame,
    factores: list[str],
    estadisticas_por_fecha: dict[pd.Timestamp, pd.DataFrame | None],
    ventana_factor: int,
    rezago_factor: int,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Para cada corte, estandariza TODA la serie de factores con la
    media/desviación de ESE corte (calculada solo hasta esa fecha) y le saca
    el rolling mean rezagado. Se precalcula una vez por corte (no por mes)
    porque la transformación es la misma para todos los meses que ese corte
    va a predecir.
    """
    resultado = {}
    for corte, stats in estadisticas_por_fecha.items():
        f = factores_df[factores]
        if stats is not None:
            z = (f - stats.loc[factores, "media"]) / stats.loc[factores, "desv"]
        else:
            z = f
        columnas = {
            k: calcular_factor_esperado(z[k], ventana=ventana_factor, rezago=rezago_factor)
            for k in factores
        }
        resultado[corte] = pd.DataFrame(columnas, index=factores_df.index)
    return resultado


def calcular_rendimiento_esperado_backtest(
    exposiciones_por_fecha: dict[pd.Timestamp, pd.DataFrame],
    estadisticas_por_fecha: dict[pd.Timestamp, pd.DataFrame | None],
    factores_df: pd.DataFrame,
    factores: list[str] | None = None,
    ventana_factor: int = 12,
    rezago_factor: int = 1,
) -> pd.DataFrame:
    """
    Para cada mes t, usa el beta del corte aplicable (el más reciente ANTES
    de t) y el factor esperado construido con la MISMA media/desviación de
    ESE corte (consistencia de escala beta <-> factor esperado), para
    calcular E[R_i,t] = sum_k beta_i,k * factor_esperado_k,t.

    Regresa: DataFrame fecha x ticker. NaN donde todavía no hay ningún corte
    anterior disponible (arranque del backtest) o donde el factor esperado
    de ese mes está incompleto.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    cortes_ordenados = sorted(exposiciones_por_fecha.keys())
    factor_esperado_por_corte = _factor_esperado_por_corte(
        factores_df, factores, estadisticas_por_fecha, ventana_factor, rezago_factor,
    )

    filas = {}
    for fecha in factores_df.index:
        corte = _beta_aplicable(fecha, cortes_ordenados)
        if corte is None:
            continue
        fe = factor_esperado_por_corte[corte].loc[fecha, factores]
        if fe.isna().any():
            continue
        betas = exposiciones_por_fecha[corte][factores]
        filas[fecha] = pd.Series(betas.to_numpy() @ fe.to_numpy(), index=betas.index)

    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).T.sort_index()


# --------------------------------------------------------------------- #
# 3) Coeficiente de Información (IC)
# --------------------------------------------------------------------- #
def calcular_ic_serie(
    rendimiento_esperado: pd.DataFrame,
    rendimiento_real: pd.DataFrame,
    min_acciones: int = MIN_ACCIONES_IC,
) -> pd.DataFrame:
    """
    Para cada fecha en común, correlación de Spearman entre el rendimiento
    esperado (ex ante) y el rendimiento real (ex post) de las acciones.

    Regresa: DataFrame indexado por fecha con columnas "ic" y "n_acciones".
    """
    fechas = rendimiento_esperado.index.intersection(rendimiento_real.index)
    filas = {}
    for fecha in sorted(fechas):
        esp = rendimiento_esperado.loc[fecha]
        real = rendimiento_real.loc[fecha]
        datos = pd.concat([esp, real], axis=1, keys=["esperado", "real"]).dropna()
        if len(datos) < min_acciones:
            continue
        rho, _ = spearmanr(datos["esperado"], datos["real"])
        filas[fecha] = {"ic": rho, "n_acciones": len(datos)}
    return pd.DataFrame(filas).T if filas else pd.DataFrame(columns=["ic", "n_acciones"])


def resumen_ic(ic_df: pd.DataFrame) -> dict:
    """
    Análisis de consistencia de una serie de IC: media, desviación estándar,
    IC Ratio (media/std -> qué tan estable es la señal), t-stat y hit rate.
    """
    ic = ic_df["ic"].dropna() if "ic" in ic_df else pd.Series(dtype=float)
    n = len(ic)
    if n == 0:
        return {"ic_medio": np.nan, "ic_std": np.nan, "ic_ratio": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "n_periodos": 0}
    media = ic.mean()
    std = ic.std(ddof=1) if n > 1 else np.nan
    ic_ratio = media / std if std else np.nan
    t_stat = media / (std / np.sqrt(n)) if std else np.nan
    hit_rate = (ic > 0).mean()
    return {"ic_medio": media, "ic_std": std, "ic_ratio": ic_ratio,
            "t_stat": t_stat, "hit_rate": hit_rate, "n_periodos": n}


# --------------------------------------------------------------------- #
# Función de conveniencia: corre 1 y 2 y 3 juntos para UN modelo
# --------------------------------------------------------------------- #
def backtest_modelo(
    rendimientos: pd.DataFrame,
    factores_df: pd.DataFrame,
    rendimiento_real: pd.DataFrame,
    factores: list[str] | None = None,
    ventana_beta: int | None = None,
    min_obs: int = MIN_OBS_DEFAULT,
    frecuencia_meses: int = FRECUENCIA_DEFAULT,
    ventana_factor: int = 12,
    rezago_factor: int = 1,
    estandarizar: bool = ESTANDARIZAR_FACTORES_DEFAULT,
    min_acciones_ic: int = MIN_ACCIONES_IC,
    **kwargs_exposiciones,
) -> dict:
    """
    Corre el backtest completo de UN modelo (una combinación de parámetros) y
    regresa un dict con la serie de IC, su resumen, y las piezas intermedias
    (exposiciones por fecha, rendimiento esperado) por si se quieren
    inspeccionar o guardar.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)

    exposiciones_por_fecha, estadisticas_por_fecha = construir_exposiciones_rolling(
        rendimientos, factores_df, factores=factores, ventana_beta=ventana_beta,
        min_obs=min_obs, frecuencia_meses=frecuencia_meses, estandarizar=estandarizar,
        **kwargs_exposiciones,
    )
    rendimiento_esperado = calcular_rendimiento_esperado_backtest(
        exposiciones_por_fecha, estadisticas_por_fecha, factores_df,
        factores=factores, ventana_factor=ventana_factor, rezago_factor=rezago_factor,
    )
    ic_df = calcular_ic_serie(rendimiento_esperado, rendimiento_real, min_acciones=min_acciones_ic)
    resumen = resumen_ic(ic_df)

    return {
        "exposiciones_por_fecha": exposiciones_por_fecha,
        "estadisticas_por_fecha": estadisticas_por_fecha,
        "rendimiento_esperado": rendimiento_esperado,
        "ic_serie": ic_df,
        "resumen": resumen,
    }