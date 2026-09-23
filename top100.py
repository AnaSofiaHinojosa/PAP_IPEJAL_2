import pandas as pd

class GlobalTop100Selector:
    """
    Clase para procesar el universo Point-in-Time y filtrar las 100 empresas
    globales más consistentes y de mayor peso histórico.
    """

    def __init__(self, pit_universe_df: pd.DataFrame):
        self.df = pit_universe_df.copy()
        self.global_top_100 = pd.DataFrame()

    def calculate_rankings(self) -> pd.DataFrame:
        summary = (
            self.df.groupby('Ticker')
            .agg(
                Company=('Company', 'last'),
                Frecuencia=('Year', 'nunique'),
                Peso_Promedio=('Weight', 'mean')
            )
            .reset_index()
        )

        # Doble criterio: Frecuencia (Descendente) -> Peso Promedio (Descendente)
        summary_sorted = summary.sort_values(
            by=['Frecuencia', 'Peso_Promedio'],
            ascending=[False, False]
        ).reset_index(drop=True)

        summary_sorted['Rank_Global'] = summary_sorted.index + 1
        self.global_top_100 = summary_sorted.head(100).copy()

        return self.global_top_100

    def get_tickers_list(self) -> list:
        if self.global_top_100.empty:
            self.calculate_rankings()
        return self.global_top_100['Ticker'].tolist()

    def export_summary_to_csv(self, output_path: str):
        if self.global_top_100.empty:
            self.calculate_rankings()
        self.global_top_100.to_csv(output_path, index=False)
        print(f"Listado consolidado Global Top 100 guardado en: {output_path}")

# ====================================================================== #
# Nueva regla (profesor): SOBREVIVIENTES + TOP 100 POR MARKET CAP ACTUAL
# ====================================================================== #
import re
import difflib

# Equivalencias manuales: nombre de la empresa en el AÑO INICIAL (tal como
# viene en el archivo) -> ticker ACTUAL de esa MISMA empresa, o None si la
# empresa desapareció (fue comprada, quebró, se fusionó en otra).
#
# Por qué hace falta: la fuente histórica (Kaggle/SPY) reasigna los tickers
# de empresas compradas al ticker actual del comprador, y a veces a un ticker
# equivocado. Ej. en 2007: United Technologies aparece como "UAL" (United
# Airlines), Kraft Foods como "KHC", Wachovia como "WFC". Comparar solo por
# ticker dejaría pasar como "sobrevivientes" a empresas que no lo son.
#
# Solo están los casos que ya revisamos. Las filas que salgan como REVISAR
# en revision_sobrevivientes.csv se deciden a mano y se agregan aquí.
EQUIVALENCIAS_MANUALES = {
    # ================================================================== #
    # Regla: misma empresa que cambió de nombre/ticker o su sucesora legal
    # -> sobrevive (ticker actual). Empresa que fue COMPRADA por otra ->
    # None (no sobrevive, aunque la fuente le ponga el ticker del comprador).
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

    # --- compradas / desaparecidas: NO sobreviven ----------------------
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
    "SanDisk Corp": None,                         # Western Digital (2016)
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
    "News Corp (Class A)": None,                  # hoy es la línea de Fox; NWS actual = escisión 2013
    # Dell Inc se hizo privada en 2013; Dell Technologies es otra sociedad.
    # Además Yahoo no tiene precios de DELL antes de 2016-08 y 2016-2018 son
    # del tracking stock DVMT (el "split" 1.806 de 2018-12-28 es su canje).
    # Para volver a incluirla, borra esta línea.
    "Dell, Inc": None,
}

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


class SurvivorTop100Selector:
    """
    Regla del profesor:
      1. Sobrevivientes: empresas que están en el S&P 500 en el año INICIAL y
         en el año FINAL de la ventana (no importa si salieron y volvieron).
      2. Top 100: de las sobrevivientes, las 100 con mayor market cap más
         reciente (peso en el SPY del año final; el peso del SPY es
         proporcional al market cap ajustado por flotación).

    Recibe el universo COMPLETO (todas las empresas del índice por año), es
    decir, HoldingsUniverseBuilder(..., top_n=None).build().

    Salidas:
      global_top_100 : las 100 empresas (compatible con el pipeline)
      revision       : TODAS las empresas del año inicial con su estado:
          OK         nombre coincide con el del año final
          MANUAL     resuelto con EQUIVALENCIAS_MANUALES
          REVISAR    mismo ticker pero nombre distinto -> decidir a mano
                     (NO entra al top 100 hasta que se decida)
          DUPLICADA  otra empresa del año inicial con el mismo ticker
                     (típicamente una comprada) -> no entra
          DESCARTADA EQUIVALENCIAS_MANUALES dice que desapareció
          NO_ESTA    su ticker no existe en el año final
    """

    def __init__(self, full_universe_df: pd.DataFrame, start_year: int | None = None,
                 end_year: int | None = None, top_n: int = 100,
                 equivalencias: dict | None = None, umbral_nombre: float = 0.80):
        self.df = full_universe_df.copy()
        self.df["Ticker_norm"] = self.df["Ticker"].map(normalizar_ticker)
        self.start_year = start_year or int(self.df["Year"].min())
        self.end_year = end_year or int(self.df["Year"].max())
        self.top_n = top_n
        self.equivalencias = EQUIVALENCIAS_MANUALES if equivalencias is None else equivalencias
        self.umbral = umbral_nombre
        self.revision = pd.DataFrame()
        self.global_top_100 = pd.DataFrame()

    def _revisar_inicio(self) -> pd.DataFrame:
        ini = self.df[self.df["Year"] == self.start_year]
        fin = (self.df[self.df["Year"] == self.end_year]
               .sort_values("Weight", ascending=False)
               .drop_duplicates("Ticker_norm"))
        fin_por_ticker = fin.set_index("Ticker_norm")

        filas = []
        for _, r in ini.iterrows():
            fila = {"Company_inicio": r["Company"], "Ticker_inicio": r["Ticker"],
                    "Peso_inicio": r["Weight"], "Ticker": None, "Company_final": None,
                    "Peso_final": None, "Similitud_nombre": None, "Estado": None}
            if r["Company"] in self.equivalencias:
                destino = self.equivalencias[r["Company"]]
                if destino is None:
                    fila["Estado"] = "DESCARTADA"
                    filas.append(fila)
                    continue
                destino, manual = normalizar_ticker(destino), True
            else:
                destino, manual = r["Ticker_norm"], False

            if destino not in fin_por_ticker.index:
                fila["Estado"] = "NO_ESTA"
                fila["Ticker"] = destino
                filas.append(fila)
                continue
            f = fin_por_ticker.loc[destino]
            sim = similitud_nombres(r["Company"], f["Company"])
            fila.update(Ticker=f["Ticker"], Company_final=f["Company"],
                        Peso_final=f["Weight"], Similitud_nombre=round(sim, 2),
                        Estado="MANUAL" if manual else ("OK" if sim >= self.umbral else "REVISAR"))
            filas.append(fila)

        rev = pd.DataFrame(filas)
        # Varias empresas del año inicial apuntando al mismo ticker actual:
        # se queda la de nombre más parecido, las demás son DUPLICADA.
        vivos = rev["Estado"].isin(["OK", "MANUAL", "REVISAR"])
        if vivos.any():
            orden = rev[vivos].assign(
                _m=(rev.loc[vivos, "Estado"] == "MANUAL").astype(int)
            ).sort_values(["_m", "Similitud_nombre"], ascending=False)
            duplicadas = orden.index[orden.duplicated("Ticker", keep="first")]
            rev.loc[duplicadas, "Estado"] = "DUPLICADA"
        return rev

    def calculate_rankings(self) -> pd.DataFrame:
        if self.end_year - self.start_year < 20:
            print(f"AVISO: la ventana va de {self.start_year} a {self.end_year} "
                  f"({self.end_year - self.start_year} años). Para 20 años completos "
                  f"agrega el archivo de {self.end_year - 20} a la carpeta de datos.")
        self.revision = self._revisar_inicio()
        sobrev = self.revision[self.revision["Estado"].isin(["OK", "MANUAL"])].copy()

        # Una sola clase por empresa (GOOG/GOOGL, FOX/FOXA, NWS/NWSA): se
        # queda la de mayor peso.
        sobrev["_empresa"] = sobrev["Company_final"].map(normalizar_nombre)
        sobrev = (sobrev.sort_values("Peso_final", ascending=False)
                  .drop_duplicates("_empresa").drop(columns="_empresa"))

        sobrev = sobrev.sort_values("Peso_final", ascending=False).reset_index(drop=True)
        sobrev["Rank_Global"] = sobrev.index + 1
        sobrev["Company"] = sobrev["Company_final"]
        self.n_sobrevivientes = len(sobrev)
        self.global_top_100 = sobrev.head(self.top_n)[
            ["Ticker", "Company", "Rank_Global", "Peso_final", "Ticker_inicio",
             "Company_inicio", "Similitud_nombre", "Estado"]].copy()

        cuenta = self.revision["Estado"].value_counts()
        print(f"Año inicial {self.start_year}: {len(self.revision)} empresas -> "
              f"{self.n_sobrevivientes} sobrevivientes en {self.end_year}. "
              f"Estados: {cuenta.to_dict()}")
        if cuenta.get("REVISAR", 0):
            print(f"  {cuenta['REVISAR']} casos REVISAR (mismo ticker, nombre distinto) "
                  "quedaron FUERA hasta decidirlos en EQUIVALENCIAS_MANUALES.")
        if self.n_sobrevivientes < self.top_n:
            print(f"  AVISO: solo hay {self.n_sobrevivientes} sobrevivientes (< {self.top_n}).")
        return self.global_top_100

    def get_tickers_list(self) -> list:
        if self.global_top_100.empty:
            self.calculate_rankings()
        return self.global_top_100["Ticker"].tolist()

    def export_summary_to_csv(self, output_path: str,
                              revision_path: str = "revision_sobrevivientes.csv"):
        if self.global_top_100.empty:
            self.calculate_rankings()
        self.global_top_100.to_csv(output_path, index=False)
        orden = {"REVISAR": 0, "MANUAL": 1, "DUPLICADA": 2, "DESCARTADA": 3, "OK": 4, "NO_ESTA": 5}
        (self.revision.sort_values("Estado", key=lambda s: s.map(orden))
         .to_csv(revision_path, index=False, encoding="utf-8-sig"))
        print(f"Top {self.top_n} sobrevivientes guardado en: {output_path}")
        print(f"Revisión caso por caso guardada en: {revision_path}")