# Importar las clases definidas en los otros archivos
from Datos import HoldingsUniverseBuilder
from top100 import GlobalTop100Selector

def main():
    print("--- INICIANDO PIPELINE DE SELECCIÓN ---")

    # 1. Parámetros de entrada
    HISTORICAL_DIR = "Data"
    EXCEL_2026_PATH = "Data/holdings-daily-us-en-spy.xlsx"
    TOP_N = 100

    # 2. Construir el universo Point-in-Time
    print("\n[Paso 1/3] Procesando archivos históricos y holdings de 2026...")
    builder = HoldingsUniverseBuilder(
        historical_dir=HISTORICAL_DIR,
        excel_2026_path=EXCEL_2026_PATH,
        top_n=TOP_N
    )
    df_pit = builder.build()
    builder.export_to_csv("universo_sp500.csv")

    # 3. Seleccionar las 100 empresas globales
    print("\n[Paso 2/3] Calculando rankings de frecuencia y peso promedio...")
    selector = GlobalTop100Selector(df_pit)
    df_top100_final = selector.calculate_rankings()
    selector.export_summary_to_csv("universo_final_top100_global.csv")

    # 4. Obtener la lista de Tickers
    tickers_list = selector.get_tickers_list()
    print("\n[Paso 3/3] Universo Top 100 consolidado con éxito.")
    print(f"Total de activos únicos seleccionados: {len(tickers_list)}")

    print("\n--- PIPELINE FINALIZADO CORRECTAMENTE ---")


if __name__ == "__main__":
    main()