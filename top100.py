"""
top100.py - Selección ANUAL del universo (regla del profesor, versión 2):

  Cada año de tenencia A, se toman las empresas que estaban en el S&P 500
  al cierre del año anterior (31/12/A-1) y se escogen las 100 con mayor
  market cap en esa fecha. Ese grupo se mantiene todo el año A. Cuenta
  cualquier empresa que haya estado en el índice, aunque después la
  hayan comprado, haya quebrado o haya salido del índice a los pocos meses.

  Fuentes (Datos.py):
    - FactSet Screening (SP500_AAAA.xlsx): market cap al 31/12 y Perm. ID.
    - Holdings del SPY por año (Kaggle + State Street): el peso en el SPY es
      proporcional al market cap ajustado por flotación, así que ordena igual.
"""

import pandas as pd

import re
import difflib

# Equivalencias manuales: nombre de la empresa tal como viene en los holdings
# de cualquier año -> ticker ACTUAL de esa MISMA empresa (el que se usa para
# descargar Yahoo/EDGAR), o None si la empresa desapareció (fue comprada,
# quebró, se fusionó en otra). Con la regla anual, las empresas con None SÍ
# cuentan en el top 100 de los años en que estuvieron en el índice, pero
# quedan como SIN_FUENTE: Yahoo no tiene sus precios (FactSet sí).
#
# Por qué hace falta: la fuente histórica (Kaggle/SPY) reasigna los tickers
# de empresas compradas al ticker actual del comprador, y a veces a un ticker
# equivocado. Ej. en 2007: United Technologies aparece como "UAL" (United
# Airlines), Kraft Foods como "KHC", Wachovia como "WFC". Comparar solo por
# ticker haría descargar los datos del comprador en lugar de los de la empresa.
#
# Solo están los casos que ya revisamos. Las filas que salgan como REVISAR
# en revision_top100_anual.csv se deciden a mano y se agregan aquí.
EQUIVALENCIAS_MANUALES = {
    # ================================================================== #
    # Regla: misma empresa que cambió de nombre/ticker o su sucesora legal
    # -> su ticker actual. Empresa que fue COMPRADA por otra -> None (sus
    # datos no son los del comprador, aunque la fuente le ponga su ticker).
    # ================================================================== #

    # --- misma empresa, cambió de nombre / ticker ---------------------
    "Google, Inc (Class A)": "GOOGL",            # Google -> Alphabet (2015)
    "WellPoint, Inc": "ELV",                      # WellPoint -> Anthem -> Elevance
    "United Technologies Corp": "RTX",            # UTX -> RTX (2020); la fuente dice UAL
    "Kraft Foods, Inc (Class A)": "MDLZ",         # Kraft Foods -> Mondelez (2012); la fuente dice KHC
    "Wal-Mart Stores, Inc": "WMT",                # Wal-Mart Stores -> Walmart Inc (2018)
    "Bank of New York Mellon Corp": "BNY",        # BNY Mellon cambió de ticker BK -> BNY
    "Thermo Electron Corp": "TMO",                # Thermo Electron -> Thermo Fisher (2006); la fuente dice AVT
    "Ingersoll-Rand Co, Ltd (Class A)": "TT",     # Ingersoll-Rand -> Trane Technologies (2020); la fuente dice IR
    "CVS Caremark Corp": "CVS",                   # CVS Caremark -> CVS Health (2014)
    "Schlumberger, Ltd": "SLB",                   # Schlumberger -> SLB (2022)
    "FPL Group, Inc": "NEE",                      # FPL Group -> NextEra Energy (2010)
    "McGraw-Hill Cos, Inc": "SPGI",               # McGraw-Hill -> S&P Global (2016)
    "Praxair, Inc": "LIN",                        # Praxair + Linde AG -> Linde plc (2018). DECIDIR (nuevo CIK)
    "Hewlett-Packard Co": "HPQ",                  # HP Co -> HP Inc (2015, HPE se escindió)
    "Dominion Resources, Inc": "D",               # -> Dominion Energy (2017)
    "AmerisourceBergen Corp": "COR",              # -> Cencora (2023)
    "Franklin Resources, Inc": "BEN",             # -> Franklin Templeton
    "Hartford Financial Services Group, Inc": "HIG",
    "Laboratory Corp of America Holdings": "LH",  # -> Labcorp Holdings (2024)
    "Network Appliance, Inc": "NTAP",             # -> NetApp (2008)
    "Pulte Homes, Inc": "PHM",                    # -> PulteGroup (2010)
    "Apache Corp": "APA",                         # -> APA Corp (holding, 2021)
    "Torchmark Corp": "GL",                       # -> Globe Life (2019)
    "Symantec Corp": "GEN",                       # -> NortonLifeLock -> Gen Digital
    "CB Richard Ellis Group, Inc (Class A)": "CBRE",
    "Boston Properties, Inc": "BXP",
    "PerkinElmer, Inc": "RVTY",                   # -> Revvity (2023)
    "Tyco Electronics, Ltd": "TEL",               # -> TE Connectivity
    # fusiones donde la fuente eligió mal cuál empresa sobrevive:
    "BB&T Corp": "TFC",                           # BB&T (sucesora legal) -> Truist
    "Stanley Works": "SWK",                       # Stanley (compradora) -> Stanley Black & Decker
    "ACE, Ltd": "CB",                             # ACE compró a Chubb y tomó su nombre (2016)
    "Coach, Inc": "TPR",                          # Coach -> Tapestry (2017)
    "Tyco International, Ltd": "JCI",             # Tyco (sucesora legal) compró a Johnson Controls (2016)

    # --- compradas / desaparecidas: sin fuente en Yahoo (SIN_FUENTE) ---
    "Merrill Lynch & Co, Inc": None,              # BAC (2009)
    "Wachovia Corp": None,                        # WFC (2008)
    "Washington Mutual, Inc": None,               # quebró; activos a JPM (2008)
    "Schering-Plough Corp": None,                 # MRK (2009)
    "Wyeth": None,                                # PFE (2009)
    "Marathon Oil Corp": None,                    # COP (2024)
    "EMC Corp": None,                             # Dell (2016)
    "Sprint Nextel Corp": None,                   # T-Mobile (2020)
    "Time Warner, Inc": None,                     # AT&T (2018)
    "Monsanto Co": None,                          # Bayer (2018)
    "Burlington Northern Santa Fe Corp": None,    # Berkshire (2010); la fuente dice BRK.B
    "Forest Laboratories, Inc": None,             # Actavis -> AbbVie; la fuente dice ABBV
    "Novellus Systems, Inc": None,                # Lam Research (2012)
    "Applera Corp -- Applied Biosystems Group": None,  # Life Tech -> Thermo (2014)
    "QLogic Corp": None,                          # Cavium -> Marvell
    "SanDisk Corp": None,                         # Western Digital (2016); ver EQUIVALENCIAS_HASTA
    "Tesoro Corp": None,                          # Marathon Petroleum (2018)
    "Coventry Health Care, Inc": None,            # Aetna (2013)
    "American Standard Cos, Inc": None,           # Trane Inc -> Ingersoll-Rand (2008)
    "Juniper Networks, Inc": None,                # HPE (2025)
    "Hilton Hotels Corp": None,                   # Blackstone (2007); HLT actual es otra sociedad
    "SunTrust Banks, Inc": None,                  # fusión con BB&T (BB&T sucesora legal)
    "Black & Decker Corp": None,                  # Stanley Works (2010)
    "Chubb Corp": None,                           # comprada por ACE (2016)
    "Johnson Controls, Inc": None,                # comprada por Tyco (2016), que tomó el nombre JCI
    "Family Dollar Stores, Inc": None,            # Dollar Tree (2015)
    "Citizens Communications Co": None,           # -> Frontier; CFG es otro banco
    "L-3 Communications Holdings, Inc": None,     # Harris (sucesora legal) -> L3Harris
    "Dow Jones & Co, Inc": None,                  # News Corp (2007)
    "El Paso Corp": None,                         # Kinder Morgan (2012)
    "Allied Waste Industries, Inc": None,         # Republic Services (2008)
    "TXU Corp": None,                             # LBO (2007)
    "Solectron Corp": None,                       # Flextronics (2007)
    "IMS Health, Inc": None,                      # Quintiles (sucesora legal) -> IQVIA
    "Integrys Energy Group, Inc": None,           # WEC (2015)
    "Liz Claiborne, Inc": None,                   # -> Kate Spade -> Tapestry (2017)
    "Novell, Inc": None,
    "CA, Inc": None,                              # Broadcom (2018)
    "CBS Corp (Class B)": None,                   # Skydance (2025): Paramount Skydance es otra sociedad
    "Precision Castparts Corp": None,             # Berkshire (2016)
    "Allergan, Inc": None,                        # Actavis -> AbbVie
    "LSI Logic Corp": None,                       # Avago (2014)
    "Dynegy, Inc (Class A)": None,                # Vistra (sucesora legal, 2018)
    "Viacom, Inc (Class B)": None,                # CBS (sucesora legal, 2019)
    "HJ Heinz Co": None,                          # 3G/Berkshire (2013); KHC es otra sociedad
    "Mylan Laboratories, Inc": None,              # Upjohn (sucesora legal) -> Viatris
    # --- mismo nombre/ticker pero OTRA empresa (el nombre engaña) -------
    "Broadcom Corp (Class A)": None,              # comprada por Avago (2016), que tomó el nombre
    "Constellation Energy Group, Inc": None,      # comprada por Exelon (2012); CEG actual = escisión 2022
    "Apollo Group, Inc (Class A)": None,          # Univ. of Phoenix; APO actual = Apollo Global Mgmt
    "Cooper Industries, Ltd (Class A)": None,     # comprada por Eaton (2012); COO = Cooper Companies
    "News Corp (Class A)": None,                  # ver EQUIVALENCIAS_HASTA (solo hasta 2013)
    # Dell Inc se hizo privada en 2013; Dell Technologies es otra sociedad.
    # Además Yahoo no tiene precios de DELL antes de 2016-08 y 2016-2018 son
    # del tracking stock DVMT (el "split" 1.806 de 2018-12-28 es su canje).
    # Para volver a incluirla, borra esta línea.
    "Dell, Inc": None,

    # --- casos de la revisión 2010-2026 (nombres como vienen en holdings) ---
    "Facebook, Inc. Class A": "META",             # Facebook -> Meta Platforms (2021)
    "TheCoca-ColaCo.": "KO",
    "Schlumberger NV": "SLB",
    "Raytheon Technologies Corp.": "RTX",         # -> RTX Corp (2023)
    "Priceline Group, Inc.": "BKNG",              # -> Booking Holdings (2018)
    "priceline.com, Inc.": "BKNG",
    "Anthem, Inc.": "ELV",                        # -> Elevance (2022)
    "DowDuPont, Inc.": "DD",                      # DowDuPont -> DuPont de Nemours (2019, sucesora legal)
    "Raytheon Co.": None,                         # fusión con UTX (UTX sucesora legal, 2020)
    "Dow Chemical Co.": None,                     # -> DowDuPont (2017); DOW actual = escisión 2019
    "Allergan PLC": None,                         # AbbVie (2020)
    "Actavis PLC": None,                          # = Allergan PLC (cambió de nombre 2015)
    "Celgene Corp.": None,                        # Bristol-Myers Squibb (2019)
    "Twenty-First Century Fox, Inc. (Class A)": None,  # Disney (2019)
    "Express Scripts Holding Co.": None,          # Cigna (2018)
    "Express Scripts, Inc.": None,
    "DIRECTV": None,                              # AT&T (2015)
    "Anadarko Petroleum Corp.": None,             # Occidental (2019)
    "Time Warner Cable, Inc.": None,              # Charter (2016)
    "Aetna, Inc.": None,                          # CVS (2018)
    "Activision Blizzard, Inc.": None,            # Microsoft (2023)
    "Walgreen Co.": None,                         # -> WBA; se hizo privada (2025), Yahoo ya no tiene precios
    "Walgreens Boots Alliance, Inc.": None,
    "Google, Inc. (Class C)": "GOOGL",            # misma empresa: una sola serie (GOOGL)
    "Alphabet, Inc. Class A": "GOOGL",
    "Alphabet, Inc. Class C": "GOOGL",
    "DIRECTV (Class A)": None,
}

# Equivalencias que solo aplican HASTA cierto año, porque después el mismo
# nombre corresponde a OTRA empresa: nombre -> (ticker o None, último año).
EQUIVALENCIAS_HASTA = {
    "SanDisk Corp": (None, 2016),         # comprada por WDC (2016); SNDK actual = escisión 2025
    "News Corp (Class A)": (None, 2013),  # vieja News Corp -> 21st Century Fox; NWS actual = escisión 2013
}

# Filas de los holdings que no son acciones (efectivo, divisas, futuros)
NO_ACCIONES = {"US DOLLAR", "CASH", "CASH USD", "USD CASH"}


def clave_equivalencia(nombre: str) -> str:
    """Normalización ligera para buscar en EQUIVALENCIAS_MANUALES: sin
    puntuación, espacios ni 'The' inicial, pero SIN quitar Inc/Corp/Class,
    para no confundir 'Broadcom Corp' (comprada) con 'Broadcom Inc' (AVGO).
    'Google, Inc (Class A)' = 'Google, Inc. (Class A)'."""
    s = str(nombre).strip().lower()
    s = re.sub(r"^the\s*", "", s)
    return re.sub(r"[^a-z0-9]", "", s)

_PALABRAS_VACIAS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COS", "COMPANY",
    "COMPANIES", "LTD", "PLC", "LLC", "LP", "HOLDING", "HOLDINGS", "GROUP",
    "THE", "NEW", "DE", "CL", "CLASS", "SHARES", "SHS", "NV", "SA", "AG",
    "AND", "A", "B", "C", "ORD", "COM",
}


def normalizar_nombre(nombre: str) -> str:
    """'Johnson & Johnson' y 'JOHNSON + JOHNSON' -> 'JOHNSON JOHNSON'."""
    s = re.sub(r"[^A-Z0-9 ]", " ", str(nombre).upper())
    return " ".join(p for p in s.split() if p not in _PALABRAS_VACIAS)


def normalizar_ticker(ticker: str) -> str:
    """BRK/B, BRK-B, BRK B -> BRK.B"""
    return re.sub(r"[\s/\-]", ".", str(ticker).strip().upper())


def similitud_nombres(a: str, b: str) -> float:
    """0-1. Tolera espacios ('EXXON MOBIL' vs 'EXXONMOBIL') y nombres que
    contienen al otro ('MOTOROLA' vs 'MOTOROLA SOLUTIONS')."""
    na, nb = normalizar_nombre(a), normalizar_nombre(b)
    if not na or not nb:
        return 0.0
    r1 = difflib.SequenceMatcher(None, na, nb).ratio()
    r2 = difflib.SequenceMatcher(None, na.replace(" ", ""), nb.replace(" ", "")).ratio()
    ta, tb = set(na.split()), set(nb.split())
    contenido = 0.9 if (ta <= tb or tb <= ta) else 0.0
    return max(r1, r2, contenido)


# ====================================================================== #
# Selector anual
# ====================================================================== #
# Estados de cada empresa-año:
#   OK          nombre coincide con el de su ticker en la fuente más reciente
#   FACTSET     viene de FactSet (el ticker ya es el de la misma acción)
#   MANUAL      resuelto con EQUIVALENCIAS_MANUALES
#   FUERA_HOY   su ticker ya no está en el índice hoy, se descarga tal cual
#   REVISAR     mismo ticker que una empresa de hoy pero nombre distinto:
#               la fuente probablemente le reasignó el ticker -> SIN datos
#               hasta decidirlo a mano en EQUIVALENCIAS_MANUALES
#   SIN_FUENTE  EQUIVALENCIAS_MANUALES dice que desapareció (None)
#   DUPLICADA   otra empresa del mismo año apunta al mismo ticker
CON_DATOS = ["OK", "FACTSET", "MANUAL", "FUERA_HOY"]


class AnnualTop100Selector:
    """
    Recibe el universo COMPLETO por año de tenencia (columnas Year, Company,
    Ticker, Weight y opcionalmente Fuente), es decir, la salida de
    Datos.combinar_fuentes(...).

    Salidas:
      seleccion : una fila por (año, empresa) del top 100
                  Year, Rank, Firma_id, Ticker, Company, Weight, Estado, Fuente
      revision  : TODAS las empresas-año del índice con su estado
      resumen   : una fila por empresa seleccionada alguna vez
    """

    def __init__(self, universe_df: pd.DataFrame, anio_inicio: int = 2010,
                 anio_fin: int | None = None, top_n: int = 100,
                 equivalencias: dict | None = None, umbral_nombre: float = 0.80):
        self.df = universe_df.copy()
        if "Fuente" not in self.df:
            self.df["Fuente"] = "Holdings SPY"
        self.df["Ticker_norm"] = self.df["Ticker"].map(normalizar_ticker)
        # Quita efectivo y otras filas que no son acciones (tickers "-", ".")
        es_accion = (self.df["Ticker_norm"].str.contains(r"[A-Z]", regex=True) &
                     ~self.df["Company"].astype(str).str.strip().str.upper().isin(NO_ACCIONES))
        if (~es_accion).any():
            print(f"  Se quitan {int((~es_accion).sum())} filas que no son acciones "
                  f"({', '.join(sorted(set(self.df.loc[~es_accion, 'Company'].astype(str))))[:120]})")
        self.df = self.df[es_accion]
        self.anio_inicio = anio_inicio
        self.anio_fin = anio_fin or int(self.df["Year"].max())
        self.top_n = top_n
        equiv = EQUIVALENCIAS_MANUALES if equivalencias is None else equivalencias
        self.equivalencias = {clave_equivalencia(k): v for k, v in equiv.items()}
        self.equiv_hasta = {clave_equivalencia(k): v for k, v in EQUIVALENCIAS_HASTA.items()}
        self.umbral = umbral_nombre
        self.revision = pd.DataFrame()
        self.seleccion = pd.DataFrame()

    # ------------------------------------------------------------------ #
    def _resolver(self) -> pd.DataFrame:
        """Asigna a cada empresa-año el ticker con el que se descargan sus
        datos (Ticker_datos) y su estado."""
        # Referencia de nombres: la fuente de holdings más reciente (hoy)
        hold = self.df[self.df["Fuente"] != "FactSet"]
        ref_anio = int(hold["Year"].max()) if len(hold) else int(self.df["Year"].max())
        ref = (self.df[self.df["Year"] == ref_anio]
               .sort_values("Weight", ascending=False)
               .drop_duplicates("Ticker_norm").set_index("Ticker_norm")["Company"])

        d = self.df[(self.df["Year"] >= self.anio_inicio) &
                    (self.df["Year"] <= self.anio_fin)].copy()
        tick, estado, sim_l = [], [], []
        for _, r in d.iterrows():
            t, e, s = None, None, None
            clave = clave_equivalencia(r["Company"])
            hasta = self.equiv_hasta.get(clave)
            if hasta is not None and r["Year"] <= hasta[1]:
                encontrado, destino = True, hasta[0]
            elif hasta is None and clave in self.equivalencias:
                encontrado, destino = True, self.equivalencias[clave]
            else:
                encontrado = False
            if encontrado:
                if destino is None:
                    e = "SIN_FUENTE"
                else:
                    t, e = normalizar_ticker(destino), "MANUAL"
            elif r["Fuente"] == "FactSet":
                t, e = r["Ticker_norm"], "FACTSET"
            elif r["Ticker_norm"] in ref.index:
                s = round(similitud_nombres(r["Company"], ref[r["Ticker_norm"]]), 2)
                if s >= self.umbral:
                    t, e = r["Ticker_norm"], "OK"
                else:
                    e = "REVISAR"
            else:
                t, e = r["Ticker_norm"], "FUERA_HOY"
            tick.append(t); estado.append(e); sim_l.append(s)
        d["Ticker_datos"], d["Estado"], d["Similitud_nombre"] = tick, estado, sim_l
        d["Nombre_hoy"] = d["Ticker_datos"].map(ref)

        # Dos empresas del mismo año con el mismo ticker de datos: se queda
        # la de nombre más parecido (MANUAL primero), la otra es DUPLICADA.
        vivos = d["Ticker_datos"].notna()
        orden = d[vivos].assign(
            _m=(d.loc[vivos, "Estado"] == "MANUAL").astype(int),
            _s=d.loc[vivos, "Similitud_nombre"].fillna(1.0),
        ).sort_values(["_m", "_s", "Weight"], ascending=False)
        dup = orden.index[orden.duplicated(["Year", "Ticker_datos"], keep="first")]
        d.loc[dup, "Estado"] = "DUPLICADA"
        d.loc[dup, "Ticker_datos"] = None

        # Identificador de empresa estable entre años
        d["Firma_id"] = d["Ticker_datos"].where(
            d["Ticker_datos"].notna(),
            "SIN_" + d["Company"].map(normalizar_nombre).str.replace(" ", "_"))
        return d

    # ------------------------------------------------------------------ #
    def calculate_rankings(self) -> pd.DataFrame:
        self.revision = self._resolver()
        d = self.revision.copy()

        # Una sola clase de acción por empresa y año (GOOG/GOOGL, FOX/FOXA,
        # NWS/NWSA): se queda la de mayor peso.
        d["_empresa"] = d["Nombre_hoy"].fillna(d["Company"]).map(normalizar_nombre)
        d = (d.sort_values("Weight", ascending=False)
             .drop_duplicates(["Year", "_empresa"]).drop(columns="_empresa"))

        partes = []
        anios_disp = sorted(d["Year"].unique())
        for anio in range(self.anio_inicio, self.anio_fin + 1):
            base = d[d["Year"] == anio]
            nota = ""
            if base.empty:
                previos = [a for a in anios_disp if a < anio]
                if not previos:
                    print(f"  AVISO: no hay composición para {anio} ni para años anteriores; se omite.")
                    continue
                base = d[d["Year"] == previos[-1]].copy()
                nota = f" (sin archivo de {anio}: se repite la selección de {previos[-1]})"
                base["Fuente"] = base["Fuente"] + f" [repetido de {previos[-1]}]"
                base["Year"] = anio
            top = base.sort_values("Weight", ascending=False).head(self.top_n).copy()
            top["Rank"] = range(1, len(top) + 1)
            partes.append(top)
            n_datos = top["Estado"].isin(CON_DATOS).sum()
            print(f"  {anio}: top {len(top)} de {len(base)} empresas del índice; "
                  f"{n_datos} con fuente de datos{nota}")

        self.seleccion = pd.concat(partes, ignore_index=True)[
            ["Year", "Rank", "Firma_id", "Ticker_datos", "Company", "Nombre_hoy",
             "Weight", "Estado", "Similitud_nombre", "Fuente"]
        ].rename(columns={"Ticker_datos": "Ticker"})

        n_emp = self.seleccion["Firma_id"].nunique()
        sin = self.seleccion[~self.seleccion["Estado"].isin(CON_DATOS)]
        print(f"Empresas distintas en el top {self.top_n} de {self.anio_inicio} a "
              f"{self.anio_fin}: {n_emp} ({self.seleccion['Ticker'].nunique()} con ticker para descargar)")
        if len(sin):
            print(f"  {sin['Firma_id'].nunique()} empresas sin fuente de datos por ahora "
                  f"(estados: {sin.drop_duplicates('Firma_id')['Estado'].value_counts().to_dict()}). "
                  "Ver revision_top100_anual.csv")
        return self.seleccion

    # ------------------------------------------------------------------ #
    def get_tickers_list(self) -> list:
        if self.seleccion.empty:
            self.calculate_rankings()
        return sorted(self.seleccion["Ticker"].dropna().unique().tolist())

    def resumen_por_empresa(self) -> pd.DataFrame:
        """Una fila por empresa con ticker (para la hoja Info de cada Excel)."""
        if self.seleccion.empty:
            self.calculate_rankings()
        s = self.seleccion.dropna(subset=["Ticker"]).sort_values("Year")
        return (s.groupby("Ticker")
                .agg(Company=("Company", "last"),
                     Anios_en_top100=("Year", lambda x: ", ".join(str(a) for a in x)),
                     N_anios=("Year", "nunique"),
                     Primer_anio=("Year", "min"),
                     Ultimo_anio=("Year", "max"),
                     Mejor_rank=("Rank", "min"),
                     Estado=("Estado", "last"),
                     Fuente=("Fuente", "last"))
                .reset_index())

    def export_summary_to_csv(self, output_path: str = "seleccion_anual_top100.csv",
                              revision_path: str = "revision_top100_anual.csv"):
        if self.seleccion.empty:
            self.calculate_rankings()
        self.seleccion.to_csv(output_path, index=False, encoding="utf-8-sig")
        orden = {"REVISAR": 0, "SIN_FUENTE": 1, "DUPLICADA": 2, "MANUAL": 3,
                 "FUERA_HOY": 4, "FACTSET": 5, "OK": 6}
        # En la revisión basta una fila por empresa y estado (con sus años)
        rev = (self.revision.groupby(["Company", "Ticker", "Estado"], dropna=False)
               .agg(Ticker_datos=("Ticker_datos", "first"), Nombre_hoy=("Nombre_hoy", "first"),
                    Similitud_nombre=("Similitud_nombre", "first"),
                    Anios=("Year", lambda x: ", ".join(str(a) for a in sorted(x))),
                    Peso_max=("Weight", "max"))
               .reset_index())
        # marca si alguna vez estuvo en el top 100
        en_top = set(zip(self.seleccion["Company"], self.seleccion["Estado"]))
        rev["En_top100"] = [(c, e) in en_top for c, e in zip(rev["Company"], rev["Estado"])]
        (rev.sort_values(["Estado", "En_top100", "Peso_max"],
                         key=lambda s: s.map(orden) if s.name == "Estado" else s,
                         ascending=[True, False, False])
         .to_csv(revision_path, index=False, encoding="utf-8-sig"))
        print(f"Selección anual guardada en: {output_path}")
        print(f"Revisión caso por caso guardada en: {revision_path}")