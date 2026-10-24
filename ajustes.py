import numpy as np
import pandas as pd

from estimar_exposiciones import (
    estandarizar_factores,
    FACTOR_BASE_DEFAULT,
    VALOR_RESTRICCION_DEFAULT,
)

PISO_SENAL = 0.05  # la varianza "verdadera" entre acciones no baja de 5% de la observada
MODOS_PESO = ("ic_var", "ic_ratio")


# (2) Errores estándar de los betas (insumo del shrinkage)
def _errores_una_accion(y: np.ndarray, X: np.ndarray, idx_base: int | None,
                        restringir_suma: bool, valor_restriccion: float) -> np.ndarray:
    """Error estándar de cada beta de UNA acción, con la misma transformación
    que estimar_exposiciones.estimar_exposiciones_beta."""
    n, p = X.shape
    if restringir_suma:
        otros = [j for j in range(p) if j != idx_base]
        y_t = y - valor_restriccion * X[:, idx_base]
        A = np.column_stack([np.ones(n), X[:, otros] - X[:, [idx_base]]])
    else:
        y_t = y
        A = np.column_stack([np.ones(n), X])

    gl = n - A.shape[1]
    if gl <= 0:
        return np.full(p, np.nan)

    coef, *_ = np.linalg.lstsq(A, y_t, rcond=None)
    resid = y_t - A @ coef
    s2 = float(resid @ resid) / gl
    cov = s2 * np.linalg.pinv(A.T @ A)

    if restringir_suma:
        # covarianza de los gamma (betas de los otros factores)
        cov_g = cov[1:, 1:]
        se = np.empty(p)
        se[otros] = np.sqrt(np.clip(np.diag(cov_g), 0.0, None))
        # beta_base = valor - sum(gamma)  ->  Var = 1' Cov(gamma) 1
        se[idx_base] = np.sqrt(max(float(cov_g.sum()), 0.0))
    else:
        se = np.sqrt(np.clip(np.diag(cov)[1:], 0.0, None))
    return se


def errores_estandar_exposiciones(
    rendimientos: pd.DataFrame,
    factores_df: pd.DataFrame,
    factores: list[str],
    factor_base: str | None = None,
    restringir_suma: bool = True,
    valor_restriccion: float = VALOR_RESTRICCION_DEFAULT,
    min_obs: int = 24,
    estandarizar: bool = True,
    estandarizar_y: bool = False,
) -> pd.DataFrame:
    """Matriz ticker x factor con el error estándar de cada beta.

    Replica el preprocesamiento de estimar_matriz_exposiciones (exceso sobre
    rf, z-score de factores y, si aplica, z-score global del rendimiento), de
    modo que los errores estándar corresponden a EXACTAMENTE los mismos betas
    que estima ese módulo."""
    factores = list(factores)
    factor_base = factor_base or FACTOR_BASE_DEFAULT
    idx_base = factores.index(factor_base) if restringir_suma else None

    rf = factores_df["rf"] if "rf" in factores_df.columns else None
    if estandarizar:
        X, _ = estandarizar_factores(factores_df, factores)
    else:
        X = factores_df[factores]

    exceso = rendimientos
    if rf is not None:
        exceso = rendimientos.sub(rf.reindex(rendimientos.index), axis=0)
    if estandarizar_y:
        vals = exceso.to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        exceso = (exceso - vals.mean()) / vals.std(ddof=1)

    filas = {}
    for ticker in exceso.columns:
        datos = X[factores].copy()
        datos["_y_"] = exceso[ticker]
        datos = datos.dropna()
        if len(datos) < min_obs:
            filas[ticker] = np.full(len(factores), np.nan)
            continue
        filas[ticker] = _errores_una_accion(
            datos["_y_"].to_numpy(), datos[factores].to_numpy(),
            idx_base, restringir_suma, valor_restriccion,
        )
    return pd.DataFrame.from_dict(filas, orient="index", columns=factores).sort_index()


# (2) Shrinkage de los betas
def encoger_exposiciones(
    betas: pd.DataFrame,
    errores: pd.DataFrame,
    kappa: float = 1.0,
    piso: float = PISO_SENAL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Encoge los betas de cada acción hacia el promedio de todas las acciones.

    betas   : ticker x factor (una matriz de exposiciones de un corte)
    errores : ticker x factor, error estándar de cada beta
    kappa   : intensidad (0 = sin encogimiento; mayor = más encogimiento)

    Regresa (betas_encogidos, lambda por acción). Las acciones sin beta
    válido siguen siendo NaN, igual que en el modelo base."""
    errores = errores.reindex(index=betas.index, columns=betas.columns)
    se2 = errores.pow(2)

    # Varianza observada entre acciones de cada beta = señal + ruido de estimación
    var_cs = betas.var(axis=0, ddof=1)
    tau2 = np.maximum(var_cs - se2.mean(axis=0), piso *
                      var_cs)   # señal "verdadera"

    # Cociente ruido/señal de cada acción, promediado sobre los factores
    q = se2.div(tau2, axis=1).mean(axis=1, skipna=False)
    lam = 1.0 / (1.0 + kappa * q)
    # sin error estándar -> sin encogimiento
    lam = lam.where(q.notna(), 1.0)
    # sin beta válido -> NaN
    lam = lam.where(betas.notna().all(axis=1))

    centro = betas.mean(axis=0)
    lam_v = lam.to_numpy()[:, None]
    encogidos = lam_v * betas.to_numpy() + (1.0 - lam_v) * \
        centro.to_numpy()[None, :]
    return pd.DataFrame(encogidos, index=betas.index, columns=betas.columns), lam


# (1) Pesos por factor a partir del IC
def pesos_por_ic(
    ic_df: pd.DataFrame,
    ventana: int | None = None,
    min_periodos: int = 12,
    modo: str = "ic_var",
) -> pd.DataFrame:
    """Peso de cada factor en cada fecha, calculado SOLO con el IC de fechas
    anteriores (shift(1): el IC del mes t nunca entra en el peso del mes t).

    ic_df        : fecha x factor, IC mensual de cada factor
    ventana      : meses de la ventana móvil; None = ventana expansiva
    min_periodos : meses mínimos de IC para empezar a ponderar
    modo         : "ic_var"   -> max(IC medio, 0) / varianza del IC
                   "ic_ratio" -> max(IC medio, 0) / desviación del IC

    Regresa fecha x factor, normalizado para que el peso promedio sea 1.
    Donde no hay historia suficiente (o ningún factor tiene IC medio > 0) el
    peso es 1 para todos: se comporta como el modelo base."""
    if modo not in MODOS_PESO:
        raise ValueError(f"modo debe ser uno de {MODOS_PESO}")

    hist = ic_df.shift(1)
    if ventana is None:
        r = hist.expanding(min_periods=min_periodos)
    else:
        r = hist.rolling(window=ventana, min_periods=min_periodos)
    media, var = r.mean(), r.var()

    positivo = media.clip(lower=0)
    crudo = positivo / var if modo == "ic_var" else positivo / np.sqrt(var)
    crudo = crudo.replace([np.inf, -np.inf], np.nan)

    pesos = pd.DataFrame(1.0, index=ic_df.index, columns=ic_df.columns)
    ok = crudo.notna().all(axis=1) & (crudo.sum(axis=1) > 0)
    if ok.any():
        pesos.loc[ok] = crudo.loc[ok].div(crudo.loc[ok].mean(axis=1), axis=0)
    return pesos
