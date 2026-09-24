"""
Descarga de datos de Yahoo Finance para la construcción de factores del PAP
(Instituto de Pensiones de Jalisco).

Ventana: enero 2006 - fecha actual, frecuencia mensual. Por ticker entrega:
    - precio_ajustado : cierre ajustado por splits y dividendos
                        (rendimiento total -> Volatilidad y Momentum)
    - precio_cierre   : cierre ajustado solo por splits (sin dividendos)
                        -> para P/E y market cap
    - volumen         : volumen del mes (ajustado por splits) -> Liquidez
y guarda la lista de splits de cada ticker (self.splits), que
sec_edgar_fundamentals.py usa para poner la EPS y las acciones de EDGAR en
la misma base de acciones que estos precios.

Además: S&P 500 (^GSPC) para el factor Mercado y T-bill de 13 semanas
(^IRX) como tasa libre de riesgo aproximada.

Las fechas se guardan como FIN de mes (el cierre mensual de Yahoo es el
del último día hábil del mes), para empatar bien con la fecha de
publicación de los datos de EDGAR.

Requiere: pip install yfinance pandas --break-system-packages
"""

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import yfinance as yf


FECHA_INICIO = "2006-01-01"
FECHA_FIN = datetime.today().strftime("%Y-%m-%d")


@dataclass
class ResultadoDescarga:
    """Encapsula el resultado (éxito o error) de una descarga por ticker."""
    ticker: str
    ok: bool
    datos: pd.DataFrame = None
    error: str = ""


def _a_fin_de_mes(indice) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(indice)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.to_period("M").to_timestamp(how="end").normalize()


def _descargar_mensual(simbolo: str, inicio: str, fin: str) -> pd.DataFrame:
    data = yf.download(simbolo, start=inicio, end=fin, interval="1mo",
                       auto_adjust=False, actions=False, progress=False)
    if data is None or data.empty:
        raise ValueError("Yahoo Finance regresó datos vacíos "
                         "(ticker deslistado, renombrado o mal escrito)")
    # yfinance a veces regresa columnas de 2 niveles (campo, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(subset=["Close"])
    data.index = _a_fin_de_mes(data.index)
    return data[~data.index.duplicated(keep="last")]


def _descargar_splits(simbolo: str) -> pd.Series:
    """Splits del ticker: índice = fecha (sin zona horaria), valor = ratio."""
    splits = yf.Ticker(simbolo).splits
    if splits is None or len(splits) == 0:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(splits.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(splits.values.astype(float), index=idx.normalize())
    return s[s > 0].sort_index()


class YahooFactorDataFetcher:
    """
    Descarga y organiza los datos crudos de Yahoo Finance necesarios para
    construir los factores del PAP, para un universo de tickers y una
    ventana de fechas fija.
    """

    def __init__(self, tickers: list[str],
                 fecha_inicio: str = FECHA_INICIO,
                 fecha_fin: str = FECHA_FIN):
        self.tickers = tickers
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin
        self.precios: dict[str, pd.DataFrame] = {}
        self.splits: dict[str, pd.Series] = {}
        self.mercado: pd.DataFrame = pd.DataFrame()
        self.errores: dict[str, str] = {}

    def descargar_precios_mensuales(self) -> dict[str, ResultadoDescarga]:
        """Precio ajustado, precio de cierre y volumen mensual por ticker.
        Un ticker con problemas no tumba el proceso completo."""
        resultados = {}
        for ticker in self.tickers:
            try:
                data = _descargar_mensual(ticker, self.fecha_inicio, self.fecha_fin)
                try:
                    self.splits[ticker] = _descargar_splits(ticker)
                except Exception:
                    self.splits[ticker] = pd.Series(dtype=float)

                df = pd.DataFrame(index=data.index)
                df.index.name = "fecha"
                df["precio_ajustado"] = data["Adj Close"] if "Adj Close" in data else data["Close"]
                df["precio_cierre"] = data["Close"]
                df["volumen"] = data["Volume"]

                self.precios[ticker] = df
                resultados[ticker] = ResultadoDescarga(ticker, ok=True, datos=df)

            except Exception as e:
                self.errores[ticker] = str(e)
                resultados[ticker] = ResultadoDescarga(ticker, ok=False, error=str(e))

        return resultados

    def descargar_mercado_y_libre_riesgo(self) -> dict[str, ResultadoDescarga]:
        """S&P 500 (^GSPC) y T-bill 13 semanas (^IRX, en % anual) en un solo
        DataFrame mensual (self.mercado)."""
        series = {"^GSPC": "sp500", "^IRX": "tbill_13w"}
        resultados, columnas = {}, []
        for simbolo, nombre in series.items():
            try:
                data = _descargar_mensual(simbolo, self.fecha_inicio, self.fecha_fin)
                serie = data["Close"].rename(nombre)
                columnas.append(serie)
                resultados[simbolo] = ResultadoDescarga(simbolo, ok=True, datos=serie.to_frame())
            except Exception as e:
                self.errores[simbolo] = str(e)
                resultados[simbolo] = ResultadoDescarga(simbolo, ok=False, error=str(e))

        if columnas:
            self.mercado = pd.concat(columnas, axis=1)
            self.mercado.index.name = "fecha"
        return resultados

    def descargar_eps_trimestral(self) -> dict[str, pd.Series]:
        """EPS diluida de los últimos ~5 trimestres según Yahoo (estado de
        resultados trimestral). Solo sirve para completar los trimestres que
        el API de la SEC todavía no trae. {ticker: Serie fecha_cierre -> EPS}"""
        self.eps_trimestral = {}
        for ticker in self.precios:
            try:
                est = yf.Ticker(ticker).quarterly_income_stmt
                fila = next((f for f in ("Diluted EPS", "Basic EPS") if f in est.index), None)
                if fila is not None:
                    s = est.loc[fila].dropna()
                    s.index = pd.to_datetime(s.index)
                    self.eps_trimestral[ticker] = s.sort_index().astype(float)
            except Exception:
                pass
        return self.eps_trimestral

    def reporte_errores(self) -> pd.DataFrame:
        """Tickers que fallaron y por qué (deslistados, ticker mal escrito, etc.)."""
        return pd.DataFrame(
            [{"ticker": t, "error": e} for t, e in self.errores.items()]
        )