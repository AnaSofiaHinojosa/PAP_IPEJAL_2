
#USER_AGENT = ("Camila Zapata camila05danielazapatac@gmail.com")

"""
Complemento a yahoo_factor_data.py: obtiene de SEC EDGAR (API de "company
facts" / XBRL) lo que Yahoo Finance no da con suficiente historia:
    - EPS trimestral histórica          -> para el factor Value (P/E)
    - Acciones en circulación histórica -> para Tamaño y Liquidez

Cómo se limpian los datos:
- EPS: solo trimestres de ~3 meses (se descartan acumulados 6M/9M y
  anuales), se usa la PRIMERA vez que se publicó cada periodo (sin
  restatements -> sin look-ahead bias) y el Q4 = anual - (Q1+Q2+Q3).
  Se combinan EPS diluida / básica; si no hay, se calcula como utilidad
  neta / acciones promedio.
- Acciones: se prueban varios tags (portada "dei", balance, promedio
  ponderado) y se usa el que llega a la fecha MÁS RECIENTE. Esto importa en
  empresas con varias clases de acciones (CMCSA, UPS, MA, NKE, F...), donde
  el dato de portada deja de venir en 2010-2015.
- SPLITS: todo se convierte a la base de acciones de HOY (igual que los
  precios de Yahoo), usando la fecha de publicación de cada dato:
      EPS_hoy = EPS_reportada / factor ;  acciones_hoy = acciones * factor
  donde factor = producto de los splits posteriores a la publicación.
  Se hace ANTES de calcular el Q4 y el TTM, para no mezclar bases.
- ESCALA: se descartan registros de acciones a >10x o <0.1x de la mediana
  del tag (XBRL mal etiquetado por la empresa, p. ej. BRK 2010-2012 x10^6);
  quedan listados en reporte_cobertura_edgar.csv, columna descartes_escala.
- Clases con conversión (BRK.B): el factor es SOLO la conversión (1,500),
  sin volver a aplicar los splits de la clase B.
- Empresas reorganizadas (XOM, DIS, MDT, CI, GOOG...): la SEC les dio un
  CIK nuevo; se pega el historial del CIK anterior (solo datos publicados
  antes de que empezara a reportar el CIK nuevo).

IMPORTANTE:
- SEC EDGAR exige un User-Agent con nombre y correo real.
- Los datos XBRL existen más o menos desde 2009-2011, así que 2006-2009
  queda vacío en P/E y market cap. Es una limitación de la fuente.

Requiere: pip install requests pandas --break-system-packages
"""

from dataclasses import dataclass
import time
import requests
import pandas as pd

USER_AGENT = "Camila Zapata camila05danielazapatac@gmail.com"  # <-- CAMBIAR antes de correr
HEADERS = {"User-Agent": USER_AGENT}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json"

# Empresas que ya no están en company_tickers.json (deslistadas / compradas)
# pero cuyo historial sigue en EDGAR.
CIK_MANUAL = {
    "WBA": "1618921",  # Walgreens Boots Alliance (privada desde 2025)
}

# CIKs anteriores de empresas que se reorganizaron como holding / nueva
# sociedad. Se usan solo para datos publicados ANTES de que el CIK actual
# empezara a reportar. Verifica cualquier CIK nuevo en
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>
CIK_PREDECESORES = {
    "XOM": ["34088"],     # Exxon Mobil Corp (antes de ExxonMobil Holdings, 2026)
    "DIS": ["1001039"],   # Walt Disney Co antes de la compra de Fox (2019)
    "MDT": ["64670"],     # Medtronic Inc antes de Medtronic plc (2015)
    "CI": ["701221"],     # Cigna Corp antes de la compra de Express Scripts (2018)
    "GOOG": ["1288776"],  # Google Inc antes de Alphabet (2015)
    "GOOGL": ["1288776"],
    "WBA": ["104207"],    # Walgreen Co antes de Walgreens Boots Alliance (2014)
    # (cik, fecha de corte): se usa el CIK anterior para lo publicado ANTES
    # del corte y se TIRAN los datos del CIK actual anteriores al corte.
    # Hace falta cuando el CIK actual trae historia de OTRA empresa:
    # JCI: NO se usa Johnson Controls Inc (CIK 53669). La sucesora legal es
    # Tyco (CIK actual 833444) y Yahoo también da la historia de Tyco para
    # JCI (splits 2007 1:4 y 2012 2.01167 son de Tyco), así que precio y
    # fundamentales ya son de la misma empresa.
    "LIN": [("884905", "2018-10-31")],  # Praxair Inc antes de Linde plc
    "DELL": ["826083"],                 # Dell Inc (hasta 2013; privada 2013-2016)
}

# Acciones con varias clases donde la clase que cotiza NO es 1:1 con la que
# EDGAR usa para reportar. Berkshire reporta todo "por acción clase A
# equivalente" y 1 clase A = 1,500 clase B.
CONVERSION_CLASE = {"BRK.B": 1500}
# Tags que no sirven para ciertos tickers (p. ej. la portada de BRK solo
# trae las acciones clase A, no el total equivalente).
TAGS_EXCLUIDOS = {"BRK.B": {"dei:EntityCommonStockSharesOutstanding"}}

# (taxonomía, tag, unidad) en orden de preferencia.
EPS_TAGS = [
    ("us-gaap", "EarningsPerShareDiluted", "USD/shares"),
    ("us-gaap", "EarningsPerShareBasic", "USD/shares"),
    ("us-gaap", "EarningsPerShareBasicAndDiluted", "USD/shares"),
]
# Solo para completar el dato ANUAL (y derivar el Q4) cuando la empresa lo
# reporta con otro tag (FCX desde 2023: trimestres en EarningsPerShareDiluted
# pero el anual en IncomeLossFromContinuingOperationsPerDilutedShare).
EPS_TAGS_SOLO_ANUAL = [
    ("us-gaap", "IncomeLossFromContinuingOperationsPerDilutedShare", "USD/shares"),
    ("us-gaap", "IncomeLossFromContinuingOperationsPerBasicShare", "USD/shares"),
]
UTILIDAD_TAGS = [
    ("us-gaap", "NetIncomeLoss", "USD"),
    ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic", "USD"),
    ("us-gaap", "ProfitLoss", "USD"),
]
ACCIONES_PROMEDIO_TAGS = [
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
    ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted", "shares"),
]
SHARES_TAGS = [
    ("dei", "EntityCommonStockSharesOutstanding", "shares"),
    ("us-gaap", "CommonStockSharesOutstanding", "shares"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
    ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted", "shares"),
]

FORMAS_VALIDAS = ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT")

# Un dato de EDGAR se arrastra (forward-fill) como máximo este número de
# días después de publicado. Si la empresa dejó de reportar ese dato, la
# celda queda vacía en vez de repetir un valor viejo por años.
DIAS_MAX_VIGENCIA = 220


def normalizar_ticker_sec(ticker: str) -> str:
    """La SEC usa guion para clases de acciones: BRK.B -> BRK-B."""
    return ticker.upper().replace(".", "-")


def factor_split(fechas: pd.Series, splits: pd.Series | None) -> pd.Series:
    """Para cada fecha: producto de los splits que ocurrieron DESPUÉS.
    (cuántas acciones de hoy equivalen a 1 acción de esa fecha)"""
    fechas = pd.to_datetime(pd.Series(fechas)).reset_index(drop=True)
    factor = pd.Series(1.0, index=fechas.index)
    if splits is None or len(splits) == 0:
        return factor
    for fecha_split, ratio in splits.items():
        if ratio and ratio > 0:
            factor[fechas < pd.Timestamp(fecha_split)] *= float(ratio)
    return factor


@dataclass
class ResultadoEdgar:
    ticker: str
    ok: bool
    datos: pd.DataFrame = None
    error: str = ""


class SecEdgarFundamentalsFetcher:
    """Descarga EPS y acciones en circulación históricas desde SEC EDGAR
    para un universo de tickers, con manejo de errores por ticker.

    splits: {ticker: Serie con índice = fecha del split, valor = ratio}
            (los da YahooFactorDataFetcher.splits). Sin splits, los datos
            quedan como se reportaron."""

    def __init__(self, tickers: list[str], pausa_entre_requests: float = 0.15,
                 splits: dict[str, pd.Series] | None = None):
        self.tickers = [t.upper() for t in tickers]
        self.pausa = pausa_entre_requests  # SEC pide no golpear la API muy rápido
        self.splits = splits or {}
        self.cik_por_ticker: dict[str, str] = {}
        self.eps_historico: dict[str, pd.DataFrame] = {}      # trimestral + eps_ttm
        self.acciones_historico: dict[str, pd.DataFrame] = {}
        self.errores: dict[str, str] = {}
        self.cobertura: dict[str, dict] = {}
        self._descartes: dict[str, list[str]] = {}  # registros tirados por escala
        self._ticker_actual = None

    # ------------------------------------------------------------------ #
    # Descarga
    # ------------------------------------------------------------------ #
    def cargar_mapeo_ticker_cik(self) -> None:
        """Descarga el mapeo oficial ticker -> CIK que publica la SEC."""
        if "tu_correo@ejemplo.com" in USER_AGENT:
            raise RuntimeError("Cambia USER_AGENT en sec_edgar_fundamentals.py "
                               "por tu nombre y correo real (la SEC lo exige).")
        try:
            resp = requests.get(TICKER_CIK_URL, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.cik_por_ticker = {
                row["ticker"].upper(): str(row["cik_str"])
                for row in data.values()
            }
        except Exception as e:
            raise RuntimeError(f"No se pudo descargar el mapeo ticker->CIK: {e}")

    def _buscar_cik(self, ticker: str) -> str | None:
        t = normalizar_ticker_sec(ticker)
        return (self.cik_por_ticker.get(t)
                or self.cik_por_ticker.get(ticker.upper())
                or CIK_MANUAL.get(ticker.upper())
                or CIK_MANUAL.get(t))

    def _descargar_company_facts(self, cik: str) -> dict:
        url = FACTS_URL_TEMPLATE.format(cik=int(cik))
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _fecha_min_filed(facts: dict):
        fechas = [r.get("filed")
                  for tax in facts.get("facts", {}).values()
                  for nodo in tax.values()
                  for regs in nodo.get("units", {}).values()
                  for r in regs if r.get("filed")]
        return min(fechas) if fechas else None

    def _facts_con_predecesores(self, ticker: str, facts: dict) -> tuple[dict, list[str]]:
        """Agrega al JSON del CIK actual los registros de CIKs anteriores
        publicados antes de que el CIK actual empezara a reportar."""
        predecesores = CIK_PREDECESORES.get(ticker, [])
        if not predecesores:
            return facts, []
        corte_default = self._fecha_min_filed(facts) or "9999-12-31"
        usados = []
        for entrada in predecesores:
            cik_p, corte = (entrada if isinstance(entrada, tuple) else (entrada, None))
            if corte:
                # quitar del CIK actual lo publicado antes del corte
                for tags in facts.get("facts", {}).values():
                    for nodo in tags.values():
                        for unidad, regs in nodo.get("units", {}).items():
                            nodo["units"][unidad] = [r for r in regs
                                                     if (r.get("filed") or "0000") >= corte]
            else:
                corte = corte_default
            try:
                fp = self._descargar_company_facts(cik_p)
                time.sleep(self.pausa)
            except Exception:
                continue
            usados.append(cik_p)
            for tax, tags in fp.get("facts", {}).items():
                destino_tax = facts.setdefault("facts", {}).setdefault(tax, {})
                for tag, nodo in tags.items():
                    destino_tag = destino_tax.setdefault(tag, {"units": {}})
                    for unidad, regs in nodo.get("units", {}).items():
                        previos = [r for r in regs if (r.get("filed") or "9999") < corte]
                        destino_tag.setdefault("units", {}).setdefault(unidad, []).extend(previos)
        return facts, usados

    # ------------------------------------------------------------------ #
    # Extracción y limpieza
    # ------------------------------------------------------------------ #
    def _registros(self, facts: dict, taxonomia: str, tag: str, unidad: str,
                   tipo: str | None = None) -> pd.DataFrame:
        """Registros 10-K/10-Q de un tag, ya en base de acciones de hoy.
        tipo: 'por_accion' (EPS), 'acciones' o None (montos en USD)."""
        cols = ["start", "end", "filed", "valor", "form", "fy", "fp", "tag", "factor"]
        nombre = f"{taxonomia}:{tag}"
        if nombre in TAGS_EXCLUIDOS.get(self._ticker_actual, set()):
            return pd.DataFrame(columns=cols)
        nodo = facts.get("facts", {}).get(taxonomia, {}).get(tag, {})
        registros = nodo.get("units", {}).get(unidad, [])
        filas = [
            {"start": r.get("start"), "end": r.get("end"), "filed": r.get("filed"),
             "valor": r.get("val"), "form": r.get("form"),
             "fy": r.get("fy"), "fp": r.get("fp")}
            for r in registros
            if r.get("form") in FORMAS_VALIDAS and r.get("val") is not None
        ]
        df = pd.DataFrame(filas, columns=cols[:-2])
        if df.empty:
            return pd.DataFrame(columns=cols)
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.dropna(subset=["end", "filed"]).reset_index(drop=True)
        df["valor"] = df["valor"].astype(float)
        df["tag"] = nombre

        # splits: a base de acciones de hoy, según la fecha de publicación.
        # Si el ticker reporta en "equivalentes" de otra clase (BRK: por acción
        # clase A), la conversión YA está en la base de hoy (1 A = 1,500 B
        # después del split de 2010), así que NO se multiplican además los
        # splits de la clase B: eso contaba el split dos veces (factor 75,000).
        if self._ticker_actual in CONVERSION_CLASE:
            f = pd.Series(float(CONVERSION_CLASE[self._ticker_actual]), index=df.index)
        else:
            f = factor_split(df["filed"], self.splits.get(self._ticker_actual))
        df["factor"] = f.values
        if tipo == "por_accion":
            df["valor"] = df["valor"] / df["factor"]
        elif tipo == "acciones":
            df["valor"] = df["valor"] * df["factor"]
            df = self._quitar_escala_erronea(df, nombre)
        return df

    def _quitar_escala_erronea(self, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
        """Descarta registros de acciones con escala imposible (XBRL mal
        etiquetado por la empresa: p. ej. BRK 2010-2012 trae las acciones
        x1,000,000). Ya en base de hoy, las acciones de una empresa no se
        mueven 10x en 20 años salvo error, así que se compara contra la
        mediana de TODOS los registros del tag. Se hace ANTES de elegir la
        primera publicación de cada periodo, para que si el dato original
        venía mal se use el de la siguiente publicación (comparativo)."""
        if len(df) < 5:
            return df
        med = df["valor"].median()
        if not med or med <= 0:
            return df
        ratio = df["valor"] / med
        malos = (ratio > 10) | (ratio < 0.1)
        if malos.any():
            self._descartes.setdefault(self._ticker_actual, []).extend(
                f"{nombre} {r.end:%Y-%m-%d} (filed {r.filed:%Y-%m-%d}) = {r.valor:.3g}"
                for r in df[malos].itertuples())
        return df[~malos].reset_index(drop=True)

    @staticmethod
    def _trimestres_point_in_time(df: pd.DataFrame, sumable: bool = True) -> pd.DataFrame:
        """
        De registros con duración (start-end), deja una fila por trimestre:
          - primera publicación de cada periodo (evita look-ahead por restatements)
          - trimestres de ~3 meses (80-100 días)
          - Q4 faltante = anual - (Q1+Q2+Q3) si `sumable`; si no (promedios
            de acciones), se usa el dato anual como aproximación del Q4.
        """
        if df.empty or df["start"].isna().all():
            return pd.DataFrame()
        df = df.dropna(subset=["start"]).copy()
        df["dias"] = (df["end"] - df["start"]).dt.days
        df = df.sort_values("filed").drop_duplicates(["start", "end"], keep="first")

        q = df[df["dias"].between(80, 100)].copy()
        q = q.sort_values("filed").drop_duplicates("end", keep="first")
        q["origen"] = "reportado"
        anual = df[df["dias"].between(350, 380)].drop_duplicates("end", keep="first")

        # El Q4 derivado del 10-K se calcula SIEMPRE y compite con un Q4
        # "reportado" por fecha de publicación. Antes, si un 10-K posterior
        # traía el Q4 como comparativo de 3 meses (1-2 años después), ese
        # dato tardío bloqueaba el Q4 derivado a tiempo, el TTM quedaba
        # "público" hasta años después y se abrían huecos de P/E (D, GOOGL).
        nuevos = []
        for _, a in anual.iterrows():
            dentro = q[(q["start"] >= a["start"] - pd.Timedelta(days=7))
                       & (q["end"] < a["end"] - pd.Timedelta(days=7))
                       & (q["filed"] <= a["filed"])]
            fila = a.copy()
            if sumable:
                if len(dentro) != 3:
                    continue
                fila["valor"] = a["valor"] - dentro["valor"].sum()
                fila["origen"] = "Q4 = anual - 3 trimestres"
            else:
                fila["origen"] = "Q4 = dato anual (aprox.)"
            fila["start"] = (dentro["end"].max() + pd.Timedelta(days=1)) if len(dentro) else a["start"]
            fila["dias"] = (fila["end"] - fila["start"]).days
            nuevos.append(fila)

        if nuevos:
            q = pd.concat([q, pd.DataFrame(nuevos)], ignore_index=True)
        # Una fila por trimestre: la que se publicó primero (fines de periodo
        # a <=7 días se consideran el mismo trimestre: años de 52/53 semanas).
        # En empate de fecha gana el reportado sobre el derivado.
        q["_deriv"] = (q["origen"] != "reportado").astype(int)
        q = q.sort_values(["filed", "_deriv"]).drop(columns="_deriv").reset_index(drop=True)
        conservar, fines = [], []
        for i, e in q["end"].items():
            if all(abs((e - f).days) > 7 for f in fines):
                conservar.append(i)
                fines.append(e)
        return q.loc[conservar].sort_values("end").reset_index(drop=True)

    @staticmethod
    def _agregar_ttm(q: pd.DataFrame) -> pd.DataFrame:
        """EPS TTM = suma de 4 trimestres CONSECUTIVOS. La fecha en que el TTM
        ya era público es la publicación más reciente de esos 4 trimestres."""
        q = q.sort_values("end").reset_index(drop=True)
        q["eps_ttm"] = q["valor"].rolling(4).sum()
        q["filed_ttm"] = pd.concat([q["filed"].shift(i) for i in range(4)], axis=1).max(axis=1)
        span = (q["end"] - q["end"].shift(3)).dt.days
        sin_ttm = ~span.between(250, 290)  # huecos -> no hay TTM
        q.loc[sin_ttm, "eps_ttm"] = float("nan")
        q.loc[sin_ttm, "filed_ttm"] = pd.NaT
        return q

    def _eps_reportada(self, facts: dict) -> pd.DataFrame:
        """Une EPS diluida / básica: por cada trimestre usa el tag preferido.
        Después completa los Q4 que falten con el dato anual de CUALQUIER
        tag de EPS (el trimestral y el anual a veces vienen en tags distintos)."""
        partes, anuales = [], []
        for prioridad, (tax, tag, unidad) in enumerate(EPS_TAGS + EPS_TAGS_SOLO_ANUAL):
            reg = self._registros(facts, tax, tag, unidad, tipo="por_accion")
            if reg.empty or reg["start"].isna().all():
                continue
            a = reg.dropna(subset=["start"]).copy()
            a["dias"] = (a["end"] - a["start"]).dt.days
            a = a[a["dias"].between(350, 380)].sort_values("filed").drop_duplicates("end")
            if not a.empty:
                anuales.append(a.assign(prioridad=prioridad))
            if (tax, tag, unidad) in EPS_TAGS:
                q = self._trimestres_point_in_time(reg)
                if not q.empty:
                    q["prioridad"] = prioridad
                    partes.append(q)
        if not partes:
            return pd.DataFrame()
        todo = pd.concat(partes, ignore_index=True)
        todo = todo.sort_values(["prioridad", "filed"]).drop_duplicates("end", keep="first")
        todo = todo.drop(columns="prioridad").sort_values("end").reset_index(drop=True)
        if anuales:
            todo = self._completar_q4(todo, pd.concat(anuales, ignore_index=True))
        return todo

    @staticmethod
    def _completar_q4(q: pd.DataFrame, anual: pd.DataFrame) -> pd.DataFrame:
        """Q4 = anual - (Q1+Q2+Q3) para los años que todavía no tienen Q4,
        usando el anual del tag de mayor prioridad que lo tenga."""
        anual = anual.sort_values(["prioridad", "filed"]).drop_duplicates("end")
        nuevos = []
        siete = pd.Timedelta(days=7)
        for _, a in anual.iterrows():
            if ((q["end"] - a["end"]).abs() <= siete).any():
                continue
            dentro = q[(q["start"] >= a["start"] - siete) & (q["end"] < a["end"] - siete)
                       & (q["filed"] <= a["filed"])]
            if len(dentro) != 3:
                continue
            fila = a.drop(labels=["prioridad"]).copy()
            fila["valor"] = a["valor"] - dentro["valor"].sum()
            fila["start"] = dentro["end"].max() + pd.Timedelta(days=1)
            fila["dias"] = (fila["end"] - fila["start"]).days
            fila["origen"] = "Q4 = anual - 3 trimestres"
            nuevos.append(fila)
        if not nuevos:
            return q
        return pd.concat([q, pd.DataFrame(nuevos)], ignore_index=True).sort_values("end").reset_index(drop=True)

    def _eps_calculada(self, facts: dict) -> pd.DataFrame:
        """EPS = utilidad neta / acciones promedio. Se prueban todas las
        combinaciones de tags y se usa la que llega a la fecha MÁS RECIENTE
        (antes se usaba la primera que funcionara, aunque se hubiera quedado
        en 2022, como FCX)."""
        opciones = []
        for pu, (tax_u, tag_u, un_u) in enumerate(UTILIDAD_TAGS):
            ni = self._trimestres_point_in_time(self._registros(facts, tax_u, tag_u, un_u))
            if ni.empty:
                continue
            for ps, (tax_s, tag_s, un_s) in enumerate(ACCIONES_PROMEDIO_TAGS):
                sh = self._trimestres_point_in_time(
                    self._registros(facts, tax_s, tag_s, un_s, tipo="acciones"), sumable=False)
                if sh.empty:
                    continue
                m = ni.merge(sh[["end", "valor", "filed", "factor"]], on="end", suffixes=("", "_acc"))
                m = m[m["valor_acc"] > 0]
                if len(m) >= 4:
                    m["valor"] = m["valor"] / m["valor_acc"]
                    m["filed"] = m[["filed", "filed_acc"]].max(axis=1)
                    m["factor"] = m["factor_acc"]  # el ajuste relevante es el de las acciones
                    m["tag"] = f"calculado: {tag_u} / {tag_s}"
                    m["origen"] = m["origen"] + " (EPS calculada)"
                    m = m.drop(columns=["valor_acc", "filed_acc", "factor_acc"]).reset_index(drop=True)
                    opciones.append((m["end"].max(), -(pu * 10 + ps), m))
        if not opciones:
            return pd.DataFrame()
        return max(opciones, key=lambda o: (o[0], o[1]))[2]

    def _eps(self, facts: dict) -> pd.DataFrame:
        """Por trimestre se prefiere la EPS reportada. La calculada (utilidad
        neta / acciones promedio) solo entra en trimestres donde la reportada
        falta o se conoció tarde (>150 días después del cierre, p. ej. solo
        como comparativo del año siguiente) y la calculada se conoció antes.
        Casos: GOOGL 2016 (EPS por clase), BRK después de 2013."""
        rep = self._eps_reportada(facts)
        calc = self._eps_calculada(facts)
        # se quitan valores imposibles en cada fuente ANTES de combinar, para
        # que un trimestre imposible en la reportada se pueda rellenar con la
        # calculada (si ésta pasa la prueba de coincidencia)
        if not rep.empty:
            rep = self._quitar_eps_imposibles(rep)
        if not calc.empty:
            calc = self._quitar_eps_imposibles(calc)
        if not rep.empty and not calc.empty and not self._calculada_coincide(rep, calc):
            calc = pd.DataFrame()
        if rep.empty:
            eps = calc
        elif calc.empty:
            eps = rep
        else:
            rep = rep.assign(_fuente=0)
            calc = calc.assign(_fuente=1)
            a_tiempo = rep["filed"] <= rep["end"] + pd.Timedelta(days=150)
            todo = pd.concat([rep, calc], ignore_index=True)
            # clave de orden: reportada a tiempo primero; luego la que se
            # publicó antes; en empate, la reportada
            todo["_orden"] = todo["filed"]
            todo.loc[todo.index[:len(rep)][a_tiempo.values], "_orden"] = pd.Timestamp("1900-01-01")
            todo = todo.sort_values(["_orden", "_fuente"]).reset_index(drop=True)
            conservar, fines = [], []
            for i, e in todo["end"].items():
                if all(abs((e - f).days) > 7 for f in fines):
                    conservar.append(i)
                    fines.append(e)
            eps = (todo.loc[conservar].drop(columns=["_orden", "_fuente"])
                   .sort_values("end").reset_index(drop=True))
        eps = self._quitar_eps_imposibles(eps)
        if len(eps) < 4:
            raise ValueError("Sin EPS utilizable (probé " +
                             ", ".join(t for _, t, _ in EPS_TAGS) +
                             " y utilidad neta / acciones promedio)")
        return self._agregar_ttm(eps)

    def _calculada_coincide(self, rep: pd.DataFrame, calc: pd.DataFrame) -> bool:
        """La EPS calculada (utilidad / acciones) solo se usa para rellenar
        si, en los trimestres donde hay ambas, se parece a la reportada
        (mediana del cociente entre 0.8 y 1.25, con al menos 4 trimestres).
        Si no, algún tag viene con escala o definición distinta (TMO, ICE)."""
        m = pd.merge_asof(calc.sort_values("end")[["end", "valor"]],
                          rep.sort_values("end")[["end", "valor"]].rename(columns={"valor": "rep"}),
                          on="end", tolerance=pd.Timedelta(days=7), direction="nearest")
        m = m.dropna()
        m = m[m["rep"].abs() > 0.05]
        if len(m) < 4:
            ok = False
        else:
            ratio = (m["valor"] / m["rep"]).median()
            ok = 0.8 <= ratio <= 1.25
        if not ok:
            self._descartes.setdefault(self._ticker_actual, []).append(
                "EPS calculada NO usada (no coincide con la reportada)")
        return ok

    def _quitar_eps_imposibles(self, eps: pd.DataFrame) -> pd.DataFrame:
        """Quita trimestres con |EPS| > 50 veces la mediana de sus vecinos.
        Los eventos reales más extremos de la muestra (GE 2018, FCX 2008,
        MCK 2020, PLD 2010) están entre 10x y 35x; arriba de 50x son datos
        mal etiquetados (TMO 2008 = 500,000; ICE 2016 = 24 millones) o EPS de
        una sociedad vacía antes de una fusión (Linde plc Q3 2018 = -182)."""
        if len(eps) < 5:
            return eps
        e = eps.sort_values("end").reset_index(drop=True)
        absx = e["valor"].abs()
        ref = absx.rolling(9, center=True, min_periods=4).median()
        malos = (ref > 0) & (absx > 50 * ref)
        if malos.any():
            self._descartes.setdefault(self._ticker_actual, []).extend(
                f"EPS {r.end:%Y-%m-%d} = {r.valor:.3g} (imposible)" for r in e[malos].itertuples())
        return e[~malos].reset_index(drop=True)

    def _rellenar_huecos(self, base: pd.DataFrame, otros: list[pd.DataFrame],
                         dias: int = 120) -> pd.DataFrame:
        """El tag elegido puede dejar de venir por años y luego regresar
        (SCHW 2021-2025, GOOGL 2014-2015: acciones reportadas por clase). En
        esos huecos se usan los datos de los otros tags (ya en base de hoy):
        se agrega una observación de otro tag solo si no hay ninguna del tag
        elegido a menos de `dias` días de su fecha de publicación."""
        base = base.copy()
        for otro in otros:
            if otro is None or otro.empty:
                continue
            fb = base["filed"].values
            nuevos = [r for _, r in otro.iterrows()
                      if (abs(fb - r["filed"].to_datetime64()) > pd.Timedelta(days=dias).to_timedelta64()).all()]
            if nuevos:
                base = pd.concat([base, pd.DataFrame(nuevos)], ignore_index=True)
        return base.sort_values("filed").reset_index(drop=True)

    def _quitar_picos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Acciones: quita una observación aislada que se aleja >40% de sus
        dos vecinas cuando las vecinas coinciden entre sí (±10%). Ej. WFC
        2023-07 (la mitad) o HPQ 2017-11 (una décima): un solo reporte mal
        capturado, no un cambio real."""
        if len(df) < 3:
            return df
        d = df.sort_values("filed").reset_index(drop=True)
        v = d["valor"]
        prev, sig = v.shift(1), v.shift(-1)
        pico = ((sig / prev - 1).abs() < 0.10) & ((v / prev - 1).abs() > 0.40)
        if pico.any():
            self._descartes.setdefault(self._ticker_actual, []).extend(
                f"acciones {r.end:%Y-%m-%d} = {r.valor:.3g} (pico aislado)" for r in d[pico].itertuples())
        return d[~pico].reset_index(drop=True)

    def _acciones(self, facts: dict) -> pd.DataFrame:
        """Prueba todos los tags y se queda con el que llega a la fecha más
        reciente (empate -> el de mayor prioridad)."""
        candidatos = []
        for prioridad, (tax, tag, unidad) in enumerate(SHARES_TAGS):
            df = self._registros(facts, tax, tag, unidad, tipo="acciones")
            if df.empty:
                continue
            if tag.startswith("WeightedAverage"):
                df = self._trimestres_point_in_time(df, sumable=False)
                if df.empty:
                    continue
            else:
                df = df.sort_values("filed").drop_duplicates("end", keep="first")
            df = df[df["valor"] > 0]
            # quita valores absurdos (p. ej. "100 acciones" al crear una holding)
            if len(df) >= 3:
                df = df[df["valor"] > 0.1 * df["valor"].median()]
            if df.empty:
                continue
            candidatos.append((df["filed"].max(), -prioridad, len(df), df))

        if not candidatos:
            raise ValueError("Sin acciones en circulación utilizables (probé " +
                             ", ".join(f"{x}:{t}" for x, t, _ in SHARES_TAGS) + ")")
        # "al día" = su último dato está a menos de ~4 meses del más reciente;
        # entre los que están al día gana el de mayor prioridad.
        ultimo = max(c[0] for c in candidatos)
        mejor = max(candidatos,
                    key=lambda c: (c[0] >= ultimo - pd.Timedelta(days=120), c[1], c[2]))[3]
        mejor = self._rellenar_huecos(mejor, [c[3] for c in candidatos if c[3] is not mejor])
        mejor = self._quitar_picos(mejor)
        cols = [c for c in ["end", "filed", "valor", "form", "fy", "fp", "tag", "origen", "factor"]
                if c in mejor]
        return mejor[cols].sort_values("filed").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Orquestación
    # ------------------------------------------------------------------ #
    def descargar_fundamentales(self) -> dict[str, ResultadoEdgar]:
        """Para cada ticker: CIK -> company facts (+ predecesores) -> EPS y
        acciones. EPS y acciones se intentan por separado."""
        if not self.cik_por_ticker:
            self.cargar_mapeo_ticker_cik()

        resultados = {}
        for ticker in self.tickers:
            self._ticker_actual = ticker
            cob = {"ticker": ticker, "cik": None, "cik_predecesores": "", "empresa_sec": None,
                   "eps_ok": False, "eps_tag": None, "eps_trimestres": 0,
                   "eps_ttm_desde": None, "eps_ttm_hasta": None,
                   "acciones_ok": False, "acciones_tag": None, "acciones_obs": 0,
                   "acciones_desde": None, "acciones_hasta": None,
                   "splits_aplicados": "", "descartes_escala": "", "error": ""}
            errores = []
            self._descartes.pop(ticker, None)
            try:
                cik = self._buscar_cik(ticker)
                if cik is None:
                    raise ValueError("Ticker no encontrado en el mapeo de la SEC "
                                     "(puede ser extranjero, ADR o ya no cotizar; "
                                     "si reporta en EE.UU. agrégalo a CIK_MANUAL)")
                cob["cik"] = cik
                facts = self._descargar_company_facts(cik)
                cob["empresa_sec"] = facts.get("entityName")
                facts, usados = self._facts_con_predecesores(ticker, facts)
                cob["cik_predecesores"] = ", ".join(usados)
                spl = self.splits.get(ticker)
                if spl is not None and len(spl):
                    cob["splits_aplicados"] = "; ".join(
                        f"{pd.Timestamp(d):%Y-%m-%d} {r:g}:1" for d, r in spl.items())

                try:
                    eps = self._eps(facts)
                    self.eps_historico[ticker] = eps
                    ttm = eps.dropna(subset=["eps_ttm"])
                    cob.update(eps_ok=True, eps_tag=", ".join(sorted(eps["tag"].unique())),
                               eps_trimestres=len(eps),
                               eps_ttm_desde=ttm["end"].min().date() if len(ttm) else None,
                               eps_ttm_hasta=ttm["end"].max().date() if len(ttm) else None)
                except Exception as e:
                    errores.append(f"EPS: {e}")

                try:
                    acc = self._acciones(facts)
                    self.acciones_historico[ticker] = acc
                    cob.update(acciones_ok=True, acciones_tag=acc["tag"].iloc[0],
                               acciones_obs=len(acc),
                               acciones_desde=acc["filed"].min().date(),
                               acciones_hasta=acc["filed"].max().date())
                except Exception as e:
                    errores.append(f"Acciones: {e}")

            except Exception as e:
                errores.append(str(e))

            if self._descartes.get(ticker):
                cob["descartes_escala"] = "; ".join(self._descartes[ticker])

            if errores:
                self.errores[ticker] = " | ".join(errores)
                cob["error"] = self.errores[ticker]
            self.cobertura[ticker] = cob
            resultados[ticker] = ResultadoEdgar(ticker, ok=not errores, error=cob["error"])
            time.sleep(self.pausa)  # buena práctica para no saturar la API de la SEC

        self._ticker_actual = None
        return resultados

    def reporte_errores(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": t, "error": e} for t, e in self.errores.items()]
        )

    def reporte_cobertura(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.cobertura.values()))


# ---------------------------------------------------------------------- #
# Relleno de los trimestres MÁS RECIENTES con Yahoo
# ---------------------------------------------------------------------- #
def completar_eps_con_yahoo(eps_df: pd.DataFrame, eps_yahoo: pd.Series | None,
                            dias_publicacion: int = 45) -> tuple[pd.DataFrame, int]:
    """El API de company facts de la SEC a veces no trae los últimos 1-3
    trimestres de algunas empresas (MDLZ solo hasta Q1 2026, C nada de 2026).
    Se agregan desde Yahoo SOLO los trimestres posteriores al último de
    EDGAR, y SOLO si Yahoo coincide con EDGAR en los trimestres que ambos
    tienen (mediana del cociente entre 0.9 y 1.1; así también se detecta si
    Yahoo está en otra base de acciones).
    Fecha de publicación aproximada = fin del trimestre + 45 días (plazo
    del 10-Q para emisoras grandes: 40 días).
    Regresa (eps_df con TTM recalculado, número de trimestres agregados)."""
    if eps_df is None or eps_df.empty or eps_yahoo is None or len(eps_yahoo) == 0:
        return eps_df, 0
    y = pd.Series(eps_yahoo).dropna()
    y.index = pd.to_datetime(y.index)
    e = eps_df.sort_values("end").reset_index(drop=True)
    # coincidencia en trimestres comunes (±7 días de fecha de cierre)
    comunes = []
    for fecha, val in y.items():
        cerca = e[(e["end"] - fecha).abs() <= pd.Timedelta(days=7)]
        if len(cerca) and abs(cerca["valor"].iloc[0]) > 0.05:
            comunes.append(val / cerca["valor"].iloc[0])
    if len(comunes) < 2 or not (0.9 <= pd.Series(comunes).median() <= 1.1):
        return eps_df, 0
    ultimo = e["end"].max()
    nuevos = [{"start": f - pd.Timedelta(days=90), "end": f,
               "filed": f + pd.Timedelta(days=dias_publicacion),
               "form": "Yahoo", "fy": None, "fp": None, "valor": float(v),
               "origen": "Yahoo (EDGAR sin el dato)", "factor": 1.0, "tag": "yahoo:DilutedEPS"}
              for f, v in y.items() if f > ultimo + pd.Timedelta(days=7)]
    if not nuevos:
        return eps_df, 0
    e = pd.concat([e.drop(columns=["eps_ttm", "filed_ttm"], errors="ignore"),
                   pd.DataFrame(nuevos)], ignore_index=True)
    return SecEdgarFundamentalsFetcher._agregar_ttm(e), len(nuevos)


# ---------------------------------------------------------------------- #
# Construcción de series mensuales
# ---------------------------------------------------------------------- #
def _vigente_por_filed(indice_mensual, fechas, valores) -> pd.Series:
    """Último valor publicado a cada fin de mes, sin arrastrar datos de más
    de DIAS_MAX_VIGENCIA días."""
    s = pd.Series(pd.Series(valores).values, index=pd.to_datetime(pd.Series(fechas).values))
    s = s[s.index.notna()].dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return pd.Series(float("nan"), index=indice_mensual)
    return s.reindex(indice_mensual, method="ffill",
                     tolerance=pd.Timedelta(days=DIAS_MAX_VIGENCIA))


def eps_ttm_mensual(indice_mensual: pd.DatetimeIndex, eps_df: pd.DataFrame) -> pd.Series:
    """EPS TTM (base de acciones de hoy) vigente en cada fin de mes."""
    return _vigente_por_filed(indice_mensual, eps_df["filed_ttm"], eps_df["eps_ttm"])


def acciones_mensual(indice_mensual: pd.DatetimeIndex, acciones_df: pd.DataFrame) -> pd.Series:
    """Acciones en circulación (base de acciones de hoy) vigentes en cada fin de mes."""
    return _vigente_por_filed(indice_mensual, acciones_df["filed"], acciones_df["valor"])


def construir_pe_mensual(precio_cierre: pd.Series, eps_df: pd.DataFrame) -> pd.Series:
    """P/E = precio de cierre (ajustado por splits, sin dividendos) / EPS TTM
    (también en base de hoy). EPS TTM <= 0 -> NaN."""
    precio = precio_cierre.squeeze()
    eps = eps_ttm_mensual(precio.index, eps_df)
    return (precio / eps.where(eps > 0)).rename("pe")


def construir_market_cap_mensual(precio_cierre: pd.Series, acciones_df: pd.DataFrame) -> pd.Series:
    """Market cap = precio de cierre x acciones (ambos en base de hoy)."""
    precio = precio_cierre.squeeze()
    return (precio * acciones_mensual(precio.index, acciones_df)).rename("market_cap")