from pathlib import Path
import pandas as pd

DATA = Path("Data")
ESTRATEGIA = Path("salidas/estrategia")
SALIDA = Path("salidas/ranking_marketcap")

N_ANIOS = 5


def leer_top20() -> list[str]:
    """Tickers del libro largo (Top N_LARGO) del mes más reciente de posiciones.csv."""
    ruta = ESTRATEGIA / "posiciones.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encontré {ruta}. Corre primero estrategia.py.")
    pos = pd.read_csv(ruta, parse_dates=["fecha"])
    ultima_fecha = pos["fecha"].max()
    largo = pos[(pos["fecha"] == ultima_fecha) & (pos["lado"] == "LARGO")]
    return largo.sort_values("er", ascending=False)["ticker"].tolist()


def anios_disponibles(n_anios: int) -> list[int]:
    """Los `n_anios` más recientes que haya en Data/<AAAA>.csv."""
    anios = []
    for f in DATA.glob("*.csv"):
        try:
            anios.append(int(f.stem))
        except ValueError:
            continue
    return sorted(anios)[-n_anios:]


def rankear_anio(anio: int) -> pd.DataFrame:
    """Ranking 1..N de TODAS las empresas del índice ese año, por Weight
    (proxy de market cap, como ya documenta el README)."""
    ruta = DATA / f"{anio}.csv"
    df = pd.read_csv(ruta)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Weight"]).sort_values(
        "Weight", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["n_empresas_indice"] = len(df)
    # si el ticker se repite (p.ej. dos clases de acción con el mismo
    # ticker), se queda con la fila de mayor peso = mejor rank
    df = df.drop_duplicates(subset="Ticker", keep="first")
    return df.set_index("Ticker")[["rank", "n_empresas_indice", "Weight"]]


def main():
    tickers = leer_top20()
    anios = anios_disponibles(N_ANIOS)
    print(f"Tickers de entrada (top {len(tickers)}): {', '.join(tickers)}")
    print(f"Años evaluados: {anios}")

    rankings_por_anio = {}
    for anio in anios:
        ruta = DATA / f"{anio}.csv"
        if not ruta.exists():
            print(f"  AVISO: no encontré {ruta}, se omite {anio}.")
            continue
        rankings_por_anio[anio] = rankear_anio(anio)

    filas_rank, filas_weight = {}, {}
    for t in tickers:
        fila_r, fila_w = {}, {}
        for anio, tabla in rankings_por_anio.items():
            if t in tabla.index:
                fila_r[anio] = int(tabla.loc[t, "rank"])
                fila_w[anio] = float(tabla.loc[t, "Weight"])
            else:
                fila_r[anio] = pd.NA   # no estaba en el índice ese año
                fila_w[anio] = pd.NA
        filas_rank[t] = fila_r
        filas_weight[t] = fila_w

    tabla_rank = pd.DataFrame(filas_rank).T.rename_axis("ticker")
    tabla_weight = pd.DataFrame(filas_weight).T.rename_axis("ticker")
    tabla_rank["rank_promedio"] = tabla_rank[anios].mean(axis=1, skipna=True)
    tabla_rank = tabla_rank.sort_values("rank_promedio")

    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla_rank.to_csv(SALIDA / "ranking_anual_top20.csv")
    tabla_weight.to_csv(SALIDA / "weight_anual_top20.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"\nRanking anual dentro del índice completo (~500), últimos {N_ANIOS} años\n{'=' * 90}")
    print(tabla_rank.round(1).to_string())
    print(f"\nResultados completos en {SALIDA}")


if __name__ == "__main__":
    main()