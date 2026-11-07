"""
descargar_sectores.py - Sector de cada empresa, para la atribución BHB.

Lee los tickers de salidas/empresas/ (y salidas/empresas_anteriores/), pide a
Yahoo Finance el sector y la industria de cada uno y escribe:

    salidas/sectores/sectores.csv            ticker, sector, industria, nombre, fuente
    salidas/sectores/sectores_faltantes.csv  tickers sin sector (revisar a mano)

CORRECCIONES MANUALES
  Si un ticker sale sin sector, se puede completar en
  sectores_manual.csv (en la raíz del repo) con columnas: ticker, sector
  Ese archivo SIEMPRE manda sobre lo que diga Yahoo. Vuelve a correr el script
  y se aplica.

REANUDACIÓN
  Si sectores.csv ya existe, no vuelve a pedir los tickers que ya tienen
  sector (usa REINTENTAR_TODO = True para pedirlos de nuevo).

LIMITACIÓN (decirlo en el reporte)
  Yahoo da el sector ACTUAL de cada empresa; se aplica a todo el periodo
  (2010 en adelante). Algunas empresas cambiaron de sector en ese lapso.

Requiere: pandas, yfinance, openpyxl (los mismos que el resto del proyecto).
Correr parado en la raíz del repo.
"""

from pathlib import Path
import time

import pandas as pd

CARPETAS_EMPRESAS = [Path("salidas/empresas"), Path("salidas/empresas_anteriores")]
SALIDA = Path("salidas/sectores")
MANUAL = Path("sectores_manual.csv")      # ticker, sector (opcional)

PAUSA_SEG = 0.5          # espera entre tickers, para no saturar a Yahoo
REINTENTOS = 3
REINTENTAR_TODO = False  # True = ignora lo ya descargado


def listar_tickers() -> list[str]:
    tickers = set()
    for carpeta in CARPETAS_EMPRESAS:
        if carpeta.exists():
            # Los archivos "~$XXX.xlsx" son bloqueos temporales de Excel (aparecen
            # cuando el Excel está abierto), no empresas: se ignoran.
            tickers.update(a.stem for a in carpeta.glob("*.xlsx")
                           if not a.name.startswith("~$"))
    return sorted(tickers)


def pedir_a_yahoo(ticker: str) -> dict:
    import yfinance as yf
    ultimo_error = None
    for intento in range(1, REINTENTOS + 1):
        try:
            info = yf.Ticker(ticker).info or {}
            return {
                "sector": info.get("sector"),
                "industria": info.get("industry"),
                "nombre": info.get("longName") or info.get("shortName"),
            }
        except Exception as e:                # red, límite de tasa, ticker raro
            ultimo_error = e
            time.sleep(2 * intento)
    print(f"  {ticker}: falló tras {REINTENTOS} intentos ({ultimo_error})")
    return {"sector": None, "industria": None, "nombre": None}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / "sectores.csv"

    tickers = listar_tickers()
    if not tickers:
        raise FileNotFoundError(
            "No encontré Excel en salidas/empresas/. Corre primero main.py.")
    print(f"{len(tickers)} tickers en las carpetas de empresas.")

    previo = pd.DataFrame(columns=["ticker", "sector", "industria", "nombre", "fuente"])
    if archivo.exists() and not REINTENTAR_TODO:
        previo = pd.read_csv(archivo)
    hechos = set(previo.loc[previo["sector"].notna(), "ticker"])
    pendientes = [t for t in tickers if t not in hechos]
    print(f"  ya con sector: {len(hechos & set(tickers))}; por pedir a Yahoo: {len(pendientes)}")

    filas = previo[previo["ticker"].isin(hechos)].to_dict("records")
    for i, t in enumerate(pendientes, 1):
        datos = pedir_a_yahoo(t)
        filas.append({"ticker": t, **datos, "fuente": "yahoo"})
        if i % 20 == 0 or i == len(pendientes):
            print(f"  [{i}/{len(pendientes)}] {t}: {datos['sector']}")
            pd.DataFrame(filas).to_csv(archivo, index=False)   # guardado parcial
        time.sleep(PAUSA_SEG)

    df = pd.DataFrame(filas).drop_duplicates("ticker", keep="last")

    # Correcciones manuales (mandan sobre Yahoo)
    if MANUAL.exists():
        man = pd.read_csv(MANUAL).dropna(subset=["ticker", "sector"])
        for _, f in man.iterrows():
            m = df["ticker"] == f["ticker"]
            if m.any():
                df.loc[m, ["sector", "fuente"]] = [f["sector"], "manual"]
            else:
                df = pd.concat([df, pd.DataFrame([{
                    "ticker": f["ticker"], "sector": f["sector"], "industria": None,
                    "nombre": None, "fuente": "manual"}])], ignore_index=True)
        print(f"  {len(man)} correcciones manuales aplicadas desde {MANUAL}")

    df = df[df["ticker"].isin(tickers)].sort_values("ticker").reset_index(drop=True)
    df.to_csv(archivo, index=False)

    faltan = df[df["sector"].isna()]
    faltan[["ticker", "nombre"]].to_csv(SALIDA / "sectores_faltantes.csv", index=False)

    print(f"\nSectores guardados en {archivo}")
    print(f"  con sector: {df['sector'].notna().sum()} / {len(df)}")
    print("\nEmpresas por sector:")
    print(df["sector"].fillna("(sin sector)").value_counts().to_string())
    if len(faltan):
        print(f"\nAVISO: {len(faltan)} tickers sin sector: "
              f"{', '.join(faltan['ticker'].head(20))}{' ...' if len(faltan) > 20 else ''}")
        print(f"  Complétalos en {MANUAL} (columnas: ticker,sector) y vuelve a correr. "
              "Usa los mismos nombres de sector que ya salieron arriba.")


if __name__ == "__main__":
    main()