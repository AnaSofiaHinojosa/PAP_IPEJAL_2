## Obtención de datos históricos
#import kagglehub
# Download latest version
#path = kagglehub.dataset_download("devaangbarthwal/s-and-p-500-holdings-and-weights-spy-2000-2024")
#print("Path to dataset files:", path)

#2026
# https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy

import os
import glob
import pandas as pd


class HoldingsUniverseBuilder:
    """
    Clase para homogeneizar y filtrar el universo de activos Top 100 del S&P 500
    combinando fuentes históricas (.csv) y reportes de holdings institucionales (.xlsx).
    """

    def __init__(self, historical_dir: str, excel_2026_path: str, top_n: int | None = 100):
        """top_n=None conserva TODAS las empresas del índice en cada año (lo
        que necesita la selección anual: el top 100 se escoge entre todas
        las empresas del índice de cada año)."""
        self.historical_dir = historical_dir
        self.excel_2026_path = excel_2026_path
        self.top_n = top_n
        self.universe_df = pd.DataFrame()

    def _process_csv_historical(self) -> list:
        """
        procesar los archivos históricos
        """
        dfs = []
        csv_files = sorted(glob.glob(os.path.join(self.historical_dir, "*.csv")))

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            try:
                # Extrae el año a partir del nombre del archivo (ej. '2025.csv' -> 2025)
                year = int(os.path.splitext(filename)[0])
            except ValueError:
                continue

            df = pd.read_csv(filepath)
            df = df[['Company', 'Ticker', 'Weight']].copy()
            df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

            # Ordenar por ponderación y (si top_n) tomar las Top N empresas
            df_top = df.dropna(subset=['Ticker']).sort_values(by='Weight', ascending=False)
            if self.top_n:
                df_top = df_top.head(self.top_n)
            df_top = df_top.copy()
            df_top['Year'] = year
            dfs.append(df_top)

        return dfs

    def _process_excel_2026(self) -> pd.DataFrame:
        """
        Procesar el reporte de holdings en Excel de 2026.
        Ajusta los metadatos iniciales omitiendo las primeras 4 filas de encabezado.
        """
        # Se omiten las 4 filas de metadatos iniciales del reporte de State Street / SPY
        df_raw = pd.read_excel(self.excel_2026_path, skiprows=4)

        df = df_raw.rename(columns={
            'Name': 'Company',
            'Ticker': 'Ticker',
            'Weight': 'Weight'
        })[['Company', 'Ticker', 'Weight']].copy()

        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        df = df.dropna(subset=['Ticker', 'Weight'])

        # Ordenar por ponderación y (si top_n) tomar las Top N empresas
        df_top = df.sort_values(by='Weight', ascending=False)
        if self.top_n:
            df_top = df_top.head(self.top_n)
        df_top = df_top.copy()
        df_top['Year'] = 2026
        return df_top

    def build(self) -> pd.DataFrame:
        """
        Ensambla y unifica los datos históricos y de 2026 en un solo Panel Dataframe.
        """
        historical_dfs = self._process_csv_historical()
        df_2026 = self._process_excel_2026()

        all_dfs = historical_dfs + [df_2026]
        self.universe_df = pd.concat(all_dfs, ignore_index=True)

        # Estandarizar orden de columnas
        self.universe_df = self.universe_df[['Year', 'Company', 'Ticker', 'Weight']]
        return self.universe_df

    def export_to_csv(self, output_path: str):
        """
        Exporta el dataset procesado a un archivo CSV.
        """
        if self.universe_df.empty:
            raise ValueError("El universo aún no ha sido construido. Ejecuta el método .build() primero.")
        self.universe_df.to_csv(output_path, index=False)
        print(f"Dataset del Universo guardado correctamente en: {output_path}")


# ====================================================================== #
# Exportaciones de FactSet Universal Screening (opcional)
# ====================================================================== #
import re


class FactSetScreeningLoader:
    """
    Lee las exportaciones de FactSet Universal Screening, una por fecha de
    corte, con el criterio FG_CONSTITUENTS(SP50,<AAAA1231>,CLOSE) y las
    columnas Symbol, Name, MktVal Co (con la misma fecha) y Perm. Sec. ID.

    Nombre de archivo: SP500_AAAA.xlsx, donde AAAA es el año de la fecha de
    corte (31/12/AAAA). Esa composición se usa para el año de tenencia
    AAAA + 1 (se escoge al cierre de diciembre y se mantiene el año siguiente).

    Devuelve las mismas columnas que HoldingsUniverseBuilder:
        Year (año de tenencia), Company, Ticker, Weight (= market cap)
    más Fuente, PermID y Fecha_corte.
    """

    def __init__(self, carpeta: str, patron: str = "SP500_*.xlsx"):
        self.carpeta = carpeta
        self.patron = patron
        self.universe_df = pd.DataFrame()

    @staticmethod
    def _leer(filepath: str) -> pd.DataFrame:
        crudo = pd.read_excel(filepath, header=None)
        # La exportación trae unas filas de título antes del encabezado:
        # se busca la fila cuya primera celda dice "Symbol".
        fila = crudo.index[crudo.iloc[:, 0].astype(str).str.strip() == "Symbol"]
        if len(fila) == 0:
            raise ValueError("no se encontró el encabezado 'Symbol'")
        df = pd.read_excel(filepath, header=int(fila[0]))
        df.columns = [str(c).strip() for c in df.columns]

        def col(*opciones):
            for o in opciones:
                for c in df.columns:
                    if c.lower() == o.lower():
                        return c
            return None

        c_mcap = col("MktVal Co", "Market Value", "Market Cap", "MktVal")
        if c_mcap is None:
            raise ValueError(f"no hay columna de market cap (columnas: {list(df.columns)})")
        out = pd.DataFrame({
            "Company": df[col("Name")],
            "Ticker": df[col("Symbol")],
            "Weight": pd.to_numeric(df[c_mcap], errors="coerce"),
            "PermID": df[col("Perm. Sec. ID", "Permanent Security Identifier")]
            if col("Perm. Sec. ID", "Permanent Security Identifier") else None,
        })
        return out.dropna(subset=["Ticker"])

    def build(self) -> pd.DataFrame:
        dfs = []
        for filepath in sorted(glob.glob(os.path.join(self.carpeta, self.patron))):
            m = re.search(r"(\d{4})", os.path.basename(filepath))
            if not m:
                continue
            anio_corte = int(m.group(1))
            try:
                df = self._leer(filepath)
            except Exception as e:
                print(f"  AVISO: {os.path.basename(filepath)} no se pudo leer ({e})")
                continue
            df["Year"] = anio_corte + 1
            df["Fecha_corte"] = f"{anio_corte}-12-31"
            df["Fuente"] = "FactSet"
            dfs.append(df)
            print(f"  FactSet {os.path.basename(filepath)}: {len(df)} empresas "
                  f"(corte 31/12/{anio_corte} -> año {anio_corte + 1})")
        if dfs:
            self.universe_df = pd.concat(dfs, ignore_index=True)
        return self.universe_df


def combinar_fuentes(holdings: pd.DataFrame, factset: pd.DataFrame | None) -> pd.DataFrame:
    """Para cada año de tenencia usa FactSet si hay exportación de ese año;
    si no, los holdings del SPY (Kaggle / State Street)."""
    holdings = holdings.copy()
    holdings["Fuente"] = holdings.get("Fuente", "Holdings SPY")
    if factset is None or factset.empty:
        return holdings
    anios_fs = set(factset["Year"].unique())
    resto = holdings[~holdings["Year"].isin(anios_fs)]
    return pd.concat([resto, factset], ignore_index=True).sort_values(["Year", "Weight"],
                                                                      ascending=[True, False])