"""
main.py - Pipeline completo del PAP:
    1. Construye el universo del S&P 500 por año (Datos.py): holdings del SPY
       y, si existen, exportaciones de FactSet Screening (Data/factset/)
    2. Selección ANUAL: cada año, las 100 de mayor market cap del S&P 500 al
       cierre de diciembre anterior, desde 2010 (top100.py)
    3. Descarga precios/volumen y mercado desde Yahoo Finance
    4. Descarga EPS y acciones en circulación desde SEC EDGAR
    5. Guarda UN Excel por empresa + mercado + reportes
    6. Construye los factores y los rendimientos para la regresión
       (construir_factores.py -> salidas/regresion)

Salidas (carpeta "salidas"):
    empresas/<TICKER>.xlsx      una por empresa, con hojas:
        Mensual   fecha, precios, volumen, EPS TTM, P/E, acciones,
                  market cap, rotación (volumen / acciones)
        EPS       EPS trimestral de EDGAR (limpia) y su TTM
        Acciones  acciones en circulación de EDGAR
        Info      empresa, rank, CIK, tags usados, cobertura y errores
    mercado.xlsx                S&P 500 y T-bill mensual
    reporte_errores.csv         todos los errores por ticker y etapa
    reporte_cobertura_edgar.csv qué se encontró en EDGAR para cada ticker
    resumen_universo.xlsx       una fila por empresa (cobertura de todo)

Antes de correr:
    - pip install yfinance requests pandas openpyxl --break-system-packages
    - Cambia USER_AGENT en sec_edgar_fundamentals.py por tu nombre y correo real
"""

from pathlib import Path
import pandas as pd

# Clases de tu amiga (construcción del universo)
from Datos import HoldingsUniverseBuilder, FactSetScreeningLoader, combinar_fuentes
from top100 import AnnualTop100Selector

# Clases propias (descarga de datos para los factores)
from yahoo_factor_data import YahooFactorDataFetcher
import construir_factores
from sec_edgar_fundamentals import (
    SecEdgarFundamentalsFetcher,
    construir_pe_mensual,
    construir_market_cap_mensual,
    eps_ttm_mensual,
    acciones_mensual,
    completar_eps_con_yahoo,
)

CARPETA_SALIDA = Path("salidas")


def normalizar_ticker_yahoo(ticker: str) -> str:
    """Yahoo Finance usa guion en vez de punto para acciones de doble clase,
    ej. BRK.B -> BRK-B."""
    return ticker.replace(".", "-")


def _sin_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Excel se ve mejor con fechas sin hora."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    return df


class PAPFactorPipeline:
    """Toma la lista final de tickers (ya salida del selector de tu amiga) y
    arma los insumos mensuales (precio, volumen, P/E, market cap) para los
    6 factores, combinando Yahoo Finance y SEC EDGAR."""

    def __init__(self, tickers_originales: list[str], info_universo: pd.DataFrame | None = None):
        self.tickers_originales = tickers_originales
        self.tickers_yahoo = [normalizar_ticker_yahoo(t) for t in tickers_originales]
        self.info_universo = (info_universo.set_index("Ticker")
                              if info_universo is not None and "Ticker" in info_universo
                              else pd.DataFrame())

        self.yahoo_fetcher = YahooFactorDataFetcher(self.tickers_yahoo)
        self.edgar_fetcher = SecEdgarFundamentalsFetcher(self.tickers_originales)

        self.tabla_mensual: dict[str, pd.DataFrame] = {}
        self.errores_pe_mcap: dict[str, str] = {}

    def ejecutar(self) -> None:
        print(f"Descargando precios/volumen de Yahoo para {len(self.tickers_yahoo)} tickers...")
        self.yahoo_fetcher.descargar_precios_mensuales()
        self.yahoo_fetcher.descargar_mercado_y_libre_riesgo()

        # Los splits de Yahoo se pasan a EDGAR para poner EPS y acciones en la
        # misma base de acciones que los precios.
        self.edgar_fetcher.splits = {
            t_orig: self.yahoo_fetcher.splits.get(t_yahoo)
            for t_orig, t_yahoo in zip(self.tickers_originales, self.tickers_yahoo)
        }

        print(f"Descargando fundamentales de SEC EDGAR para {len(self.tickers_originales)} tickers...")
        self.edgar_fetcher.descargar_fundamentales()

        # Trimestres recientes que la SEC todavía no publica en su API -> Yahoo
        print("Completando los trimestres más recientes de EPS con Yahoo...")
        eps_y = self.yahoo_fetcher.descargar_eps_trimestral()
        for t_orig, t_yahoo in zip(self.tickers_originales, self.tickers_yahoo):
            eps_df = self.edgar_fetcher.eps_historico.get(t_orig)
            nuevo, n = completar_eps_con_yahoo(eps_df, eps_y.get(t_yahoo))
            if n:
                self.edgar_fetcher.eps_historico[t_orig] = nuevo
                cob = self.edgar_fetcher.cobertura.get(t_orig, {})
                cob["eps_trimestres_yahoo"] = n
                ttm = nuevo.dropna(subset=["eps_ttm"])
                cob["eps_ttm_hasta"] = ttm["end"].max().date() if len(ttm) else cob.get("eps_ttm_hasta")

        print("Construyendo tabla mensual por empresa (P/E, market cap, rotación)...")
        for t_orig, t_yahoo in zip(self.tickers_originales, self.tickers_yahoo):
            precios = self.yahoo_fetcher.precios.get(t_yahoo)
            if precios is None:
                continue  # ya falló en Yahoo; el error está en el reporte
            tabla = precios.copy()
            eps_df = self.edgar_fetcher.eps_historico.get(t_orig)
            acc_df = self.edgar_fetcher.acciones_historico.get(t_orig)
            try:
                if eps_df is not None:
                    tabla["eps_ttm"] = eps_ttm_mensual(tabla.index, eps_df)
                    tabla["pe"] = construir_pe_mensual(tabla["precio_cierre"], eps_df)
                if acc_df is not None:
                    tabla["acciones_circulacion"] = acciones_mensual(tabla.index, acc_df)
                    tabla["market_cap"] = construir_market_cap_mensual(tabla["precio_cierre"], acc_df)
                    tabla["rotacion"] = tabla["volumen"] / tabla["acciones_circulacion"]
            except Exception as e:
                self.errores_pe_mcap[t_orig] = str(e)
            self.tabla_mensual[t_orig] = tabla

    # ------------------------------------------------------------------ #
    # Reportes
    # ------------------------------------------------------------------ #
    def reporte_errores_completo(self) -> pd.DataFrame:
        """Junta los errores de Yahoo, EDGAR y construcción de P/E/market cap."""
        filas = []
        for t, e in self.yahoo_fetcher.errores.items():
            filas.append({"ticker": t, "etapa": "yahoo_precios", "error": e})
        for t, e in self.edgar_fetcher.errores.items():
            filas.append({"ticker": t, "etapa": "sec_edgar", "error": e})
        for t, e in self.errores_pe_mcap.items():
            filas.append({"ticker": t, "etapa": "pe_mcap", "error": e})
        return pd.DataFrame(filas, columns=["ticker", "etapa", "error"])

    def _info_empresa(self, t_orig: str, t_yahoo: str) -> pd.DataFrame:
        cob = self.edgar_fetcher.cobertura.get(t_orig, {})
        tabla = self.tabla_mensual.get(t_orig)
        uni = self.info_universo.loc[t_orig] if t_orig in self.info_universo.index else {}

        def meses(col):
            return int(tabla[col].notna().sum()) if tabla is not None and col in tabla else 0

        def rango(col):
            if tabla is None or col not in tabla or tabla[col].notna().sum() == 0:
                return ""
            s = tabla[col].dropna()
            return f"{s.index.min():%Y-%m} a {s.index.max():%Y-%m}"

        info = {
            "Ticker": t_orig,
            "Ticker Yahoo": t_yahoo,
            "Empresa (universo)": uni.get("Company", ""),
            "Años en el top 100": uni.get("Anios_en_top100", ""),
            "Número de años": uni.get("N_anios", ""),
            "Mejor rank": uni.get("Mejor_rank", ""),
            "Estado de selección": uni.get("Estado", ""),
            "Fuente del universo": uni.get("Fuente", ""),
            "Empresa según SEC": cob.get("empresa_sec") or "",
            "CIK": cob.get("cik") or "",
            "CIK anteriores usados": cob.get("cik_predecesores") or "",
            "Splits aplicados": cob.get("splits_aplicados") or "",
            "Tag EPS usado": cob.get("eps_tag") or "",
            "Tag acciones usado": cob.get("acciones_tag") or "",
            "Meses con precio": meses("precio_ajustado"),
            "Rango precio": rango("precio_ajustado"),
            "Meses con P/E": meses("pe"),
            "Rango P/E": rango("pe"),
            "Meses con market cap": meses("market_cap"),
            "Rango market cap": rango("market_cap"),
            "Error Yahoo": self.yahoo_fetcher.errores.get(t_yahoo, ""),
            "Error SEC EDGAR": self.edgar_fetcher.errores.get(t_orig, ""),
            "Error P/E - market cap": self.errores_pe_mcap.get(t_orig, ""),
        }
        return pd.DataFrame({"campo": list(info.keys()), "valor": list(info.values())})

    def _escribir_excel_empresa(self, archivo: Path, t_orig: str, info: pd.DataFrame) -> None:
        with pd.ExcelWriter(archivo, engine="openpyxl") as xw:
            tabla = self.tabla_mensual.get(t_orig)
            if tabla is not None:
                _sin_hora(tabla.reset_index()).to_excel(xw, sheet_name="Mensual", index=False)
            else:
                pd.DataFrame({"nota": ["Sin precios de Yahoo; ver hoja Info"]}).to_excel(
                    xw, sheet_name="Mensual", index=False)

            eps = self.edgar_fetcher.eps_historico.get(t_orig)
            if eps is not None:
                cols = ["start", "end", "filed", "form", "fy", "fp", "valor",
                        "origen", "eps_ttm", "filed_ttm", "factor", "tag"]
                _sin_hora(eps[[c for c in cols if c in eps]].rename(
                    columns={"valor": "eps_trimestral"})).to_excel(xw, sheet_name="EPS", index=False)

            acc = self.edgar_fetcher.acciones_historico.get(t_orig)
            if acc is not None:
                _sin_hora(acc.rename(columns={"valor": "acciones"})).to_excel(
                    xw, sheet_name="Acciones", index=False)

            info.to_excel(xw, sheet_name="Info", index=False)

            # anchos de columna legibles
            for hoja in xw.sheets.values():
                for col in hoja.columns:
                    largo = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                    hoja.column_dimensions[col[0].column_letter].width = min(max(12, largo + 2), 60)

    def _guardar_seguro(self, archivo: Path, escribir) -> None:
        """Si el archivo está abierto en Excel (Windows lo bloquea), guarda una
        copia como <nombre>_nuevo y avisa, en vez de tronar y perder la
        descarga completa."""
        try:
            escribir(archivo)
        except PermissionError:
            alterno = archivo.with_name(f"{archivo.stem}_nuevo{archivo.suffix}")
            try:
                escribir(alterno)
                self.archivos_bloqueados.append(f"{archivo.name} -> se guardó como {alterno.name}")
            except PermissionError:
                self.archivos_bloqueados.append(f"{archivo.name} -> NO se pudo guardar")

    def guardar_resultados(self, carpeta: Path = CARPETA_SALIDA) -> None:
        carpeta_emp = carpeta / "empresas"
        carpeta_emp.mkdir(parents=True, exist_ok=True)
        self.archivos_bloqueados = []

        # Excel de corridas anteriores de empresas que YA NO están en el
        # universo -> se mueven a salidas/empresas_anteriores (no se borran)
        # para que validar_salidas.py no los revise como si fueran actuales.
        vigentes = {f"{t.replace('.', '-')}.xlsx" for t in self.tickers_originales}
        viejos = [a for a in carpeta_emp.glob("*.xlsx") if a.name not in vigentes]
        if viejos:
            destino = carpeta / "empresas_anteriores"
            destino.mkdir(exist_ok=True)
            for a in viejos:
                try:
                    a.replace(destino / a.name)
                except PermissionError:
                    self.archivos_bloqueados.append(f"{a.name} (no se pudo mover; ciérralo)")
            print(f"{len(viejos)} archivos de empresas fuera del universo movidos a {destino}")

        resumen = []
        for t_orig, t_yahoo in zip(self.tickers_originales, self.tickers_yahoo):
            info = self._info_empresa(t_orig, t_yahoo)
            archivo = carpeta_emp / f"{t_orig.replace('.', '-')}.xlsx"
            self._guardar_seguro(archivo, lambda a: self._escribir_excel_empresa(a, t_orig, info))
            resumen.append(dict(zip(info["campo"], info["valor"])))

        # Mercado
        if not self.yahoo_fetcher.mercado.empty:
            mercado = _sin_hora(self.yahoo_fetcher.mercado.reset_index())
            self._guardar_seguro(carpeta / "mercado.xlsx",
                                 lambda a: mercado.to_excel(a, index=False))

        # Reportes
        errores = self.reporte_errores_completo()
        cobertura = self.edgar_fetcher.reporte_cobertura()
        self._guardar_seguro(carpeta / "reporte_errores.csv",
                             lambda a: errores.to_csv(a, index=False))
        self._guardar_seguro(carpeta / "reporte_cobertura_edgar.csv",
                             lambda a: cobertura.to_csv(a, index=False))
        self._guardar_seguro(carpeta / "resumen_universo.xlsx",
                             lambda a: pd.DataFrame(resumen).to_excel(a, index=False))

        if self.archivos_bloqueados:
            print("\nAVISO: estos archivos estaban abiertos (ciérralos en Excel antes de correr):")
            for linea in self.archivos_bloqueados:
                print("   ", linea)


def revisar_cobertura_precios(pipeline: "PAPFactorPipeline", resumen: pd.DataFrame,
                              archivo: Path = CARPETA_SALIDA / "revision_precios_vs_top100.csv") -> pd.DataFrame:
    """Compara, por empresa, desde cuándo hay precios contra el primer año en
    que estuvo en el top 100. Si los precios empiezan DESPUÉS, el ticker
    probablemente ya es de otra empresa (tickers reciclados, ej. una empresa
    comprada cuyo ticker hoy usa otra) o Yahoo no tiene su historia."""
    filas = []
    for _, r in resumen.iterrows():
        tabla = pipeline.tabla_mensual.get(r["Ticker"])
        precio = tabla["precio_ajustado"].dropna() if tabla is not None and "precio_ajustado" in tabla else pd.Series(dtype=float)
        desde = precio.index.min() if len(precio) else pd.NaT
        hasta = precio.index.max() if len(precio) else pd.NaT
        inicio_top = pd.Timestamp(f"{int(r['Primer_anio'])}-01-01")
        if pd.isna(desde):
            alerta = "SIN_PRECIOS"
        elif desde > inicio_top + pd.DateOffset(months=2):   # tolera IPOs/escisiones de enero
            alerta = "PRECIOS_EMPIEZAN_TARDE"
        else:
            alerta = ""
        filas.append({"Ticker": r["Ticker"], "Company": r["Company"],
                      "Anios_en_top100": r["Anios_en_top100"],
                      "Precio_desde": desde.date() if pd.notna(desde) else "",
                      "Precio_hasta": hasta.date() if pd.notna(hasta) else "",
                      "Alerta": alerta})
    df = pd.DataFrame(filas).sort_values("Alerta", ascending=False)
    df.to_csv(archivo, index=False, encoding="utf-8-sig")
    n = (df["Alerta"] != "").sum()
    print(f"{n} empresas con alerta de precios en {archivo} (revisar a mano)")
    return df


def main():
    print("--- INICIANDO PIPELINE DE SELECCIÓN ---")

    # 1. Parámetros de entrada
    HISTORICAL_DIR = "Data"
    EXCEL_2026_PATH = "Data/holdings-daily-us-en-spy.xlsx"
    FACTSET_DIR = "Data/factset"   # SP500_AAAA.xlsx de FactSet Screening (opcional)
    TOP_N = 100
    ANIO_INICIO = 2010             # primer año de tenencia (regla del profesor)

    # 2. Universo COMPLETO del S&P 500 por año de tenencia. Si hay exportación
    #    de FactSet para un año, sustituye a los holdings del SPY de ese año.
    print("\n[Paso 1/6] Procesando holdings históricos, SPY 2026 y FactSet...")
    builder = HoldingsUniverseBuilder(
        historical_dir=HISTORICAL_DIR,
        excel_2026_path=EXCEL_2026_PATH,
        top_n=None
    )
    df_hold = builder.build()
    df_fs = FactSetScreeningLoader(FACTSET_DIR).build() if Path(FACTSET_DIR).exists() else None
    df_pit = combinar_fuentes(df_hold, df_fs)
    df_pit.to_csv("universo_sp500_completo.csv", index=False)
    print(df_pit.groupby("Year")["Fuente"].first().to_string())

    # 3. Top 100 por market cap de CADA año (al cierre de diciembre anterior)
    print(f"\n[Paso 2/6] Seleccionando el top {TOP_N} de cada año desde {ANIO_INICIO}...")
    selector = AnnualTop100Selector(df_pit, anio_inicio=ANIO_INICIO, top_n=TOP_N)
    selector.calculate_rankings()
    selector.export_summary_to_csv("seleccion_anual_top100.csv",
                                   revision_path="revision_top100_anual.csv")

    # 4. Lista de tickers: todas las empresas que estuvieron en el top 100
    #    algún año y que tienen fuente de datos
    tickers_list = selector.get_tickers_list()
    resumen = selector.resumen_por_empresa()
    print(f"\nTotal de empresas a descargar: {len(tickers_list)}")

    # 5. Descargar Yahoo Finance + SEC EDGAR para los factores
    print("\n[Paso 3/6] Descargando datos de Yahoo Finance y SEC EDGAR...")
    pipeline = PAPFactorPipeline(tickers_list, info_universo=resumen)
    pipeline.ejecutar()

    print("\n[Paso 4/6] Guardando un Excel por empresa en 'salidas/empresas'...")
    pipeline.guardar_resultados()
    revisar_cobertura_precios(pipeline, resumen)

    print("\n[Paso 5/6] Reporte de errores:")
    errores = pipeline.reporte_errores_completo()
    print(f"{len(errores)} errores registrados en salidas/reporte_errores.csv")
    if not errores.empty:
        print(errores.groupby("etapa").size())
    print("Cobertura de EDGAR por ticker en salidas/reporte_cobertura_edgar.csv")

    # 6. Factores (insumos de la Parte 2: rendimientos.csv, factores.csv, ...)
    print("\n[Paso 6/6] Construyendo factores en 'salidas/regresion'...")
    try:
        construir_factores.main()
    except Exception as e:
        print(f"No se pudieron construir los factores: {e}")

    print("\n--- PIPELINE FINALIZADO CORRECTAMENTE ---")


if __name__ == "__main__":
    main()