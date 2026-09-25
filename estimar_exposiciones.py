"""
estimar_exposiciones.py - Parte 2 (Matriz de exposiciones).

Función predeterminada que estima, para UNA acción, su exposición (beta) a
cada factor propuesto mediante una regresión de su rendimiento contra los
factores, imponiendo la restricción:

        sum_k beta_k = valor_restriccion   (por defecto 1)

CÓMO SE IMPONE LA RESTRICCIÓN (sin optimización numérica, de forma exacta)
  Se despeja un factor "base" de la restricción:
      beta_base = valor_restriccion - sum(beta_k para k != base)
  y se sustituye en la ecuación de la regresión:
      y = alpha + sum_k beta_k * X_k + e
        = alpha + beta_base * X_base + sum_{k!=base} beta_k * X_k + e
  Como beta_base = valor_restriccion - sum(beta_k):
      y - valor_restriccion * X_base
          = alpha + sum_{k!=base} beta_k * (X_k - X_base) + e
  El lado izquierdo y las (X_k - X_base) son datos conocidos, así que esto es
  una regresión MCO normal (con intercepto) sobre variables transformadas.
  De sus coeficientes se recuperan los beta_k (k != base) y, por la
  restricción, beta_base. El resultado es EXACTO: siempre suma
  valor_restriccion, sin necesidad de scipy.optimize ni programación
  cuadrática.

Si restringir_suma=False, se corre una regresión MCO normal (sin restricción).

Requiere: numpy, pandas
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Parámetros por defecto (el equipo los puede ajustar)
# ------------------------------------------------------------------ #
FACTORES_DEFAULT = ["MERCADO", "VALUE", "TAMANO", "MOMENTUM", "VOLATILIDAD", "LIQUIDEZ"]
FACTOR_BASE_DEFAULT = "MERCADO"     # factor que se despeja de la restricción
MIN_OBS_DEFAULT = 24                # mínimo de meses en común para poder estimar
VALOR_RESTRICCION_DEFAULT = 1.0     # sum(beta) = 1


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

    Parámetros
    ----------
    y : pd.Series
        Rendimiento (idealmente en exceso de la tasa libre de riesgo) de la
        acción, indexado por fecha.
    X : pd.DataFrame
        Rendimientos de los factores, mismas fechas que y en el índice.
        Columnas = nombres de los factores.
    factores : list[str], opcional
        Qué columnas de X usar. Por defecto FACTORES_DEFAULT.
    factor_base : str, opcional
        Factor que se despeja de la restricción de suma. Por defecto
        FACTOR_BASE_DEFAULT. Debe estar incluido en `factores`.
    restringir_suma : bool
        Si True (default), impone sum(beta) = valor_restriccion.
        Si False, corre MCO normal sin restricción.
    valor_restriccion : float
        Valor al que debe sumar el conjunto de betas (1 por defecto).
    min_obs : int
        Mínimo de observaciones (meses) con dato simultáneo en y y en todos
        los factores para poder estimar. Si no se alcanza, se regresa NaN.

    Regresa
    -------
    dict con:
        "alpha"      intercepto de la regresión
        "beta"       dict {nombre_factor: beta}
        "r2"         R^2 de la regresión (sobre los datos usados)
        "n_obs"      número de observaciones usadas
        "suma_beta"  suma de los betas (debe ser ~valor_restriccion si
                     restringir_suma=True; validación de la restricción)
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)
    factor_base = factor_base or FACTOR_BASE_DEFAULT
    if restringir_suma and factor_base not in factores:
        raise ValueError(f"factor_base '{factor_base}' debe estar en factores {factores}")

    # Alinear y con X y quedarnos solo con filas completas (sin NaN)
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
        # y transformada: y - valor_restriccion * factor_base
        y_t = y_arr - valor_restriccion * datos[factor_base].to_numpy()
        # X transformada: cada factor (menos el base) - el factor base
        X_t = datos[otros].to_numpy() - datos[[factor_base]].to_numpy()
        # Regresión MCO con intercepto: [1, X_t] @ [alpha, gamma...] = y_t
        A = np.column_stack([np.ones(n_obs), X_t])
        coef, *_ = np.linalg.lstsq(A, y_t, rcond=None)
        alpha = coef[0]
        gamma = dict(zip(otros, coef[1:]))
        beta_base = valor_restriccion - sum(gamma.values())
        beta = {factor_base: beta_base, **gamma}
        beta = {f: beta[f] for f in factores}  # mismo orden que `factores`

        # R^2 calculado sobre la ecuación ORIGINAL (y, no y transformada)
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
) -> pd.DataFrame:
    """
    Corre estimar_exposiciones_beta() para CADA acción (columna) en
    `rendimientos` y arma la matriz de exposiciones: una fila por empresa,
    una columna por factor (+ alpha, r2, n_obs, suma_beta).

    Parámetros
    ----------
    rendimientos : pd.DataFrame
        fecha x ticker, rendimiento simple mensual de cada acción
        (ej. salidas/regresion/rendimientos_miembros.csv ya pivoteado).
    factores_df : pd.DataFrame
        fecha x factor, con al menos las columnas en `factores` y,
        opcionalmente, "rf" si no se pasa `rf` por separado
        (ej. salidas/regresion/factores.csv).
    rf : pd.Series, opcional
        Tasa libre de riesgo mensual, mismo índice de fechas. Si se da (o si
        "rf" existe en factores_df), el rendimiento de cada acción se usa en
        EXCESO de rf antes de la regresión (estándar tipo Fama-French). Si es
        None y no hay columna "rf", se usa el rendimiento tal cual.
    (resto de parámetros: ver estimar_exposiciones_beta)

    Regresa
    -------
    pd.DataFrame indexado por ticker, con columnas:
        alpha, <un beta por factor>, r2, n_obs, suma_beta
    """
    factores = list(factores) if factores is not None else list(FACTORES_DEFAULT)

    if rf is None and "rf" in factores_df.columns:
        rf = factores_df["rf"]

    filas = {}
    for ticker in rendimientos.columns:
        y = rendimientos[ticker]
        if rf is not None:
            y = y - rf.reindex(y.index)
        resultado = estimar_exposiciones_beta(
            y=y,
            X=factores_df,
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
    return matriz[columnas].sort_index()