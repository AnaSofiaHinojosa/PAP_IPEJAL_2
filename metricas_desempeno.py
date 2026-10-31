
import numpy as np
import pandas as pd

MESES_ANIO = 12


def curva_capital(r: pd.Series) -> pd.Series:
    """Valor de 1 peso invertido al inicio."""
    return (1.0 + r).cumprod()


def retorno_acumulado(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def retorno_anualizado(r: pd.Series) -> float:
    """Rendimiento anual geométrico (compuesto)."""
    if len(r) == 0:
        return np.nan
    crecimiento = float((1.0 + r).prod())
    return crecimiento ** (MESES_ANIO / len(r)) - 1.0 if crecimiento > 0 else np.nan


def vol_anual(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(MESES_ANIO)) if len(r) > 1 else np.nan


def sharpe_clasico(exceso: pd.Series) -> float:
    """Sharpe clásico anualizado a partir del rendimiento en exceso mensual."""
    if len(exceso) < 2:
        return np.nan
    sd = exceso.std(ddof=1)
    return float(exceso.mean() / sd * np.sqrt(MESES_ANIO)) if sd > 0 else np.nan


def sortino(exceso: pd.Series) -> float:
    """Sortino anualizado: penaliza solo la desviación a la baja (contra 0)."""
    if len(exceso) < 2:
        return np.nan
    abajo = np.minimum(exceso.to_numpy(), 0.0)
    dd = np.sqrt(np.mean(abajo ** 2))
    return float(exceso.mean() / dd * np.sqrt(MESES_ANIO)) if dd > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    """Peor caída desde un máximo previo (número negativo)."""
    if len(r) == 0:
        return np.nan
    capital = curva_capital(r)
    return float((capital / capital.cummax() - 1.0).min())


def serie_drawdown(r: pd.Series) -> pd.Series:
    capital = curva_capital(r)
    return capital / capital.cummax() - 1.0


def t_stat_media(r: pd.Series) -> float:
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else np.nan


def capm(exceso: pd.Series, mercado_exceso: pd.Series) -> dict:
    """Regresión exceso_estrategia = alpha + beta * exceso_mercado + e (OLS)."""
    d = pd.concat(
        [exceso.rename("y"), mercado_exceso.rename("x")], axis=1).dropna()
    n = len(d)
    vacio = {"alpha_anual": np.nan, "t_alpha": np.nan,
             "beta_mercado": np.nan, "r2_mercado": np.nan}
    if n < 6:
        return vacio
    X = np.column_stack([np.ones(n), d["x"].to_numpy()])
    y = d["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = float(resid @ resid) / (n - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "alpha_anual": float(coef[0] * MESES_ANIO),
        "t_alpha": float(coef[0] / se[0]) if se[0] > 0 else np.nan,
        "beta_mercado": float(coef[1]),
        "r2_mercado": float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else np.nan,
    }


def resumen_metricas(
    r: pd.Series,
    exceso: pd.Series | None = None,
    mercado_exceso: pd.Series | None = None,
) -> dict:
    """Métricas de una estrategia.

    r              : rendimiento mensual total de la estrategia
    exceso         : rendimiento mensual en exceso (para Sharpe, Sortino y CAPM);
                     si es None se usa r
    mercado_exceso : rendimiento en exceso del mercado (para alpha y beta)"""
    r = r.dropna()
    exceso = (exceso if exceso is not None else r).reindex(r.index).dropna()
    mdd = max_drawdown(r)
    anual = retorno_anualizado(r)
    out = {
        "n_meses": len(r),
        "retorno_medio_mensual": float(r.mean()) if len(r) else np.nan,
        "retorno_acumulado": retorno_acumulado(r),
        "retorno_anualizado": anual,
        "vol_anual": vol_anual(r),
        "sharpe_clasico": sharpe_clasico(exceso),
        "t_stat_media": t_stat_media(r),
        "sortino": sortino(exceso),
        "max_drawdown": mdd,
        "calmar": anual / abs(mdd) if mdd and mdd < 0 and not np.isnan(anual) else np.nan,
        "hit_rate": float((r > 0).mean()) if len(r) else np.nan,
        "mejor_mes": float(r.max()) if len(r) else np.nan,
        "peor_mes": float(r.min()) if len(r) else np.nan,
    }
    if mercado_exceso is not None:
        out.update(capm(exceso, mercado_exceso))
    return out


def comparar_pareado(a: pd.Series, b: pd.Series) -> dict:
    """Diferencia mensual a - b y su t-stat (prueba t pareada)."""
    d = (a - b).dropna()
    if len(d) < 2:
        return {"dif_media_anualizada": np.nan, "t_stat_dif": np.nan, "n_meses": len(d)}
    sd = d.std(ddof=1)
    return {
        "dif_media_anualizada": float(d.mean() * MESES_ANIO),
        "t_stat_dif": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else np.nan,
        "n_meses": len(d),
    }


def sharpe_movil(exceso: pd.Series, ventana: int = 36) -> pd.Series:
    """Sharpe clásico anualizado en ventana móvil de `ventana` meses."""
    media = exceso.rolling(ventana).mean()
    desv = exceso.rolling(ventana).std(ddof=1)
    return (media / desv * np.sqrt(MESES_ANIO)).where(desv > 0)


def treynor(exceso: pd.Series, beta: float) -> float:
    """Treynor anualizado: rendimiento en exceso medio / beta."""
    if len(exceso) == 0 or beta is None or np.isnan(beta) or beta == 0:
        return np.nan
    return float(exceso.mean() * MESES_ANIO / beta)