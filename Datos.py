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
        que necesita la regla de sobrevivientes: estar en el S&P 500, no solo
        en el top 100, en el año inicial y en el final)."""
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