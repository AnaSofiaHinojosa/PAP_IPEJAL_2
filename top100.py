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