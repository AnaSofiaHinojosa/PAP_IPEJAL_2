"""
estimar_exposiciones.py - Parte 2 (Matriz de exposiciones).

Estima, para cada acción, su exposición (beta) a cada factor mediante una
regresión de su rendimiento en exceso de rf contra los factores, imponiendo

        sum_k beta_k = valor_restriccion   (por defecto 1)

ESTANDARIZACIÓN (indicación del profesor: "por factor, no por observación")
  Cada factor se estandariza UNA SOLA VEZ con la media y la desviación
  estándar de toda su serie de tiempo (factores.csv):
        z_k,t = (f_k,t - media_k) / desv_k
  Todas las acciones se regresan contra los MISMOS factores estandarizados,
  así que los betas son comparables entre acciones (unidades: por desviación
  estándar del factor). rf no es un factor y no se estandariza.
  El rendimiento de las acciones se usa en exceso de rf, sin estandarizar
  (opcional: estandarizar_y=True usa la media y desviación de todos los
  rendimientos en exceso juntos).

CÓMO SE IMPONE LA RESTRICCIÓN (exacto, sin optimización numérica)
  beta_base = valor_restriccion - sum(beta_k, k != base), y se sustituye:
      y - valor_restriccion * X_base
          = alpha + sum_{k!=base} beta_k * (X_k - X_base) + e
  que es un MCO normal sobre variables transformadas.

Si restringir_suma=False, se corre MCO normal (sin restricción).

Requiere: numpy, pandas
"""

import numpy as np
import pandas as pd

FACTORES_DEFAULT = ["MERCADO", "VALUE", "TAMANO", "MOMENTUM", "VOLATILIDAD", "LIQUIDEZ"]
FACTOR_BASE_DEFAULT = "MERCADO"
MIN_OBS_DEFAULT = 24
VALOR_RESTRICCION_DEFAULT = 1.0
ESTANDARIZAR_FACTORES_DEFAULT = True    # z-score de cada factor con toda su serie
ESTANDARIZAR_Y_DEFAULT = False          # z-score global del exceso de rendimiento


def estandarizar_factores(factores_df: pd.DataFrame,
                          factores: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score de cada factor con la media y desviación de toda su serie.
    Regresa (factores estandarizados, tabla con media y desv de cada factor)."""
    f = factores_df[factores]
    media, desv = f.mean(), f.std(ddof=1)
    if (desv.isna() | (desv == 0)).any():
        raise ValueError(f"Factor sin variación o sin datos: {list(desv[desv.isna() | (desv == 0)].index)}")
    z = (f - media) / desv
    return z, pd.DataFrame({"media": media, "desv": desv})


def estimar_exposiciones_beta(
    y: pd.Series,
    X: pd.DataFrame,
    factores: list[str] | None = None,
    factor_base: str | None = None,
    restringir_suma: bool = True,
    valor_restriccion: float = VALOR_RESTRICCION_DEFAULT,
    min_obs: int = MIN_OBS_DEFAULT,
) -> dict:
    """
    Estima la exposición (beta) de UNA acción a cada factor.
    y : rendimiento en exceso de rf, indexado por fecha.
    X : factores (ya estandarizados si aplica), mismas fechas.
    Regresa dict con alpha, beta {factor: beta}, r2, n_obs, suma_beta.
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    factor_base = factor_base or FACTOR_BASE_DEFAULT
    if restringir_suma and factor_base not in factores:
        raise ValueError(f"factor_base '{factor_base}' debe estar en factores {factores}")

    datos = X[factores].copy()
    datos["_y_"] = y
    datos = datos.dropna()
    n_obs = len(datos)

    vacio = {
        "alpha": np.nan,
        "beta": {f: np.nan for f in factores},
        "r2": np.nan,
        "n_obs": n_obs,
        "suma_beta": np.nan,
    }
    if n_obs < min_obs:
        return vacio

    y_arr = datos["_y_"].to_numpy()

    if restringir_suma:
        otros = [f for f in factores if f != factor_base]
        y_t = y_arr - valor_restriccion * datos[factor_base].to_numpy()
        X_t = datos[otros].to_numpy() - datos[[factor_base]].to_numpy()
        A = np.column_stack([np.ones(n_obs), X_t])
        coef, *_ = np.linalg.lstsq(A, y_t, rcond=None)
        alpha = coef[0]
        gamma = dict(zip(otros, coef[1:]))
        beta_base = valor_restriccion - sum(gamma.values())
        beta = {factor_base: beta_base, **gamma}
        beta = {f: beta[f] for f in factores}
        y_hat = alpha + sum(beta[f] * datos[f].to_numpy() for f in factores)
    else:
        A = np.column_stack([np.ones(n_obs), datos[factores].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        alpha = coef[0]
        beta = dict(zip(factores, coef[1:]))
        y_hat = A @ coef

    ss_res = np.sum((y_arr - y_hat) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "n_obs": n_obs,
        "suma_beta": sum(beta.values()),
    }


def estimar_matriz_exposiciones(
    rendimientos: pd.DataFrame,
    factores_df: pd.DataFrame,
    rf: pd.Series | None = None,
    factores: list[str] | None = None,
    factor_base: str | None = None,
    restringir_suma: bool = True,
    valor_restriccion: float = VALOR_RESTRICCION_DEFAULT,
    min_obs: int = MIN_OBS_DEFAULT,
    estandarizar: bool = ESTANDARIZAR_FACTORES_DEFAULT,
    estandarizar_y: bool = ESTANDARIZAR_Y_DEFAULT,
) -> pd.DataFrame:
    """
    Corre estimar_exposiciones_beta() para cada acción y arma la matriz
    (una fila por empresa, una columna por factor + alpha, r2, n_obs,
    suma_beta).

    Orden: exceso sobre rf -> z-score de cada factor (una vez, toda su serie)
    -> regresión con restricción.

    La tabla con la media y desviación de cada factor queda en
    matriz.attrs["estandarizacion"] (útil para la Parte 3).
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)

    if rf is None and "rf" in factores_df.columns:
        rf = factores_df["rf"]

    # ---- Factores: una sola media y desviación por factor ----
    if estandarizar:
        X, stats = estandarizar_factores(factores_df, factores)
    else:
        X, stats = factores_df[factores], None

    # ---- Rendimiento en exceso de rf ----
    exceso = rendimientos
    if rf is not None:
        exceso = rendimientos.sub(rf.reindex(rendimientos.index), axis=0)
    if estandarizar_y:
        vals = exceso.to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        exceso = (exceso - vals.mean()) / vals.std(ddof=1)

    filas = {}
    for ticker in exceso.columns:
        resultado = estimar_exposiciones_beta(
            y=exceso[ticker],
            X=X,
            factores=factores,
            factor_base=factor_base,
            restringir_suma=restringir_suma,
            valor_restriccion=valor_restriccion,
            min_obs=min_obs,
        )
        fila = {"alpha": resultado["alpha"]}
        fila.update(resultado["beta"])
        fila["r2"] = resultado["r2"]
        fila["n_obs"] = resultado["n_obs"]
        fila["suma_beta"] = resultado["suma_beta"]
        filas[ticker] = fila

    matriz = pd.DataFrame.from_dict(filas, orient="index")
    matriz.index.name = "ticker"
    columnas = ["alpha"] + factores + ["r2", "n_obs", "suma_beta"]
    matriz = matriz[columnas].sort_index()
    if stats is not None:
        matriz.attrs["estandarizacion"] = stats
    return matriz