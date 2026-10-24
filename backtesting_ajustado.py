import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backtesting import (
    construir_exposiciones_rolling,
    _beta_aplicable,
    _factor_esperado_por_corte,
)
from estimar_exposiciones import FACTORES_DEFAULT
from ajustes import errores_estandar_exposiciones, encoger_exposiciones


class Insumos:
    """Todo lo que se calcula una sola vez por modelo."""

    def __init__(self, rendimientos, factores_df, factores, ventana_beta, min_obs,
                 estandarizar, restringir_suma, estandarizar_y,
                 exposiciones, estadisticas, fe_por_corte):
        self.rendimientos = rendimientos
        self.factores_df = factores_df
        self.factores = factores
        self.ventana_beta = ventana_beta
        self.min_obs = min_obs
        self.estandarizar = estandarizar
        self.restringir_suma = restringir_suma
        self.estandarizar_y = estandarizar_y
        self.exposiciones = exposiciones            # {corte: matriz de exposiciones}
        self.estadisticas = estadisticas            # {corte: media/desv de los factores o None}
        self.fe_por_corte = fe_por_corte            # {corte: fecha x factor esperado}
        self.cortes = sorted(exposiciones.keys())
        self._errores = None

    def errores(self) -> dict:
        """{corte: matriz ticker x factor de errores estándar}. Se calcula la
        primera vez que se pide (solo la necesita el shrinkage), con la misma
        ventana de datos que usó cada corte para estimar sus betas."""
        if self._errores is None:
            salida = {}
            for corte in self.cortes:
                if self.ventana_beta is None:
                    rend_v = self.rendimientos.loc[:corte]
                    fact_v = self.factores_df.loc[:corte]
                else:
                    inicio = corte - pd.DateOffset(months=self.ventana_beta)
                    rend_v = self.rendimientos.loc[inicio:corte]
                    fact_v = self.factores_df.loc[inicio:corte]
                salida[corte] = errores_estandar_exposiciones(
                    rend_v, fact_v, self.factores,
                    restringir_suma=self.restringir_suma,
                    min_obs=self.min_obs,
                    estandarizar=self.estandarizar,
                    estandarizar_y=self.estandarizar_y,
                )
            self._errores = salida
        return self._errores


def construir_insumos(
    rendimientos: pd.DataFrame,
    factores_df: pd.DataFrame,
    factores: list[str] | None = None,
    ventana_beta: int | None = None,
    min_obs: int = 24,
    frecuencia_meses: int = 12,
    ventana_factor: int = 12,
    rezago_factor: int = 1,
    estandarizar: bool = True,
    restringir_suma: bool = True,
    estandarizar_y: bool = True,
) -> Insumos:
    """Betas por corte (walk-forward) y factor esperado por corte para UN modelo
    (una configuración de ventana de beta, min_obs y ventana de factor)."""
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    exposiciones, estadisticas = construir_exposiciones_rolling(
        rendimientos, factores_df, factores=factores, ventana_beta=ventana_beta,
        min_obs=min_obs, frecuencia_meses=frecuencia_meses, estandarizar=estandarizar,
        restringir_suma=restringir_suma, estandarizar_y=estandarizar_y,
    )
    fe_por_corte = _factor_esperado_por_corte(
        factores_df, factores, estadisticas, ventana_factor, rezago_factor,
    )
    return Insumos(rendimientos, factores_df, factores, ventana_beta, min_obs,
                   estandarizar, restringir_suma, estandarizar_y,
                   exposiciones, estadisticas, fe_por_corte)


def rendimiento_esperado(
    ins: Insumos,
    pesos: pd.DataFrame | None = None,
    betas_por_corte: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """E[R_i,t] = sum_k beta_i,k * w_k,t * factor_esperado_k,t   (fecha x ticker).

    pesos           : fecha x factor (técnica 1); None = todos 1 (modelo base)
    betas_por_corte : {corte: ticker x factor} (técnica 2); None = los betas
                      originales de cada corte

    Con ambos en None reproduce exactamente
    backtesting.calcular_rendimiento_esperado_backtest."""
    betas_src = betas_por_corte if betas_por_corte is not None else ins.exposiciones
    filas = {}
    for fecha in ins.factores_df.index:
        corte = _beta_aplicable(fecha, ins.cortes)        # corte estrictamente anterior
        if corte is None:
            continue
        fe = ins.fe_por_corte[corte].loc[fecha, ins.factores]
        if fe.isna().any():
            continue
        if pesos is not None:
            fe = fe * pesos.loc[fecha, ins.factores]
        betas = betas_src[corte][ins.factores]
        filas[fecha] = pd.Series(betas.to_numpy() @ fe.to_numpy(), index=betas.index)
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).T.sort_index()


def ic_por_factor(
    ins: Insumos,
    rendimiento_real: pd.DataFrame,
    min_acciones: int = 10,
) -> pd.DataFrame:
    """IC mensual de cada factor por separado (fecha x factor).

    Para el factor k en el mes t: correlación de Spearman entre lo que ese
    factor aporta al rendimiento esperado de cada acción,
        beta_i,k * factor_esperado_k,t ,
    y su rendimiento real en t. Como el factor esperado es el mismo para todas
    las acciones, equivale a signo(factor_esperado_k,t) * Spearman(beta_i,k, R_i,t).
    Se mide solo con las acciones que hay en rendimiento_real (miembros del
    Top 100 ese mes), igual que el IC del modelo completo."""
    filas = {}
    fechas = ins.factores_df.index.intersection(rendimiento_real.index)
    for fecha in fechas:
        corte = _beta_aplicable(fecha, ins.cortes)
        if corte is None:
            continue
        fe = ins.fe_por_corte[corte].loc[fecha, ins.factores]
        if fe.isna().any():
            continue
        betas = ins.exposiciones[corte][ins.factores]
        real = rendimiento_real.loc[fecha].rename("_real_")
        datos = betas.join(real, how="inner").dropna()
        if len(datos) < min_acciones:
            continue
        fila = {}
        for k in ins.factores:
            if fe[k] == 0:
                fila[k] = np.nan
                continue
            rho, _ = spearmanr(datos[k], datos["_real_"])
            fila[k] = np.sign(fe[k]) * rho
        filas[fecha] = fila
    if not filas:
        return pd.DataFrame(columns=ins.factores)
    return pd.DataFrame.from_dict(filas, orient="index")[ins.factores].sort_index()


def encoger_betas_por_corte(
    ins: Insumos,
    kappa: float,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Aplica el shrinkage a la matriz de betas de cada corte.

    Regresa ({corte: betas encogidos}, tabla con el lambda medio, mínimo y
    máximo de cada corte)."""
    errores = ins.errores()
    betas_enc, resumen = {}, {}
    for corte in ins.cortes:
        enc, lam = encoger_exposiciones(ins.exposiciones[corte][ins.factores],
                                        errores[corte], kappa=kappa)
        betas_enc[corte] = enc
        resumen[corte] = {"lambda_medio": lam.mean(),
                          "lambda_min": lam.min(),
                          "lambda_max": lam.max()}
    return betas_enc, pd.DataFrame.from_dict(resumen, orient="index")