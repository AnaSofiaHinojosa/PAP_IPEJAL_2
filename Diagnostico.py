"""
diagnostico_xbrl.py - Lista qué datos de ACCIONES, EPS y UTILIDAD trae el API
de company facts de la SEC para unas empresas, con su primer y último
periodo. Sirve para ver por qué una empresa se queda sin datos (BRK.B y V).

Uso (en la carpeta del proyecto):
    python diagnostico_xbrl.py BRK.B V

Salida:
    consola: tabla por empresa
    salidas/diagnostico_xbrl.csv
"""

import sys
import re
from pathlib import Path
import pandas as pd

from sec_edgar_fundamentals import SecEdgarFundamentalsFetcher, FORMAS_VALIDAS

PALABRAS = re.compile(r"share|earnings|netincome|profitloss|stock", re.I)


def diagnosticar(tickers: list[str]) -> pd.DataFrame:
    f = SecEdgarFundamentalsFetcher(tickers)
    f.cargar_mapeo_ticker_cik()
    filas = []
    for t in tickers:
        cik = f._buscar_cik(t)
        if cik is None:
            print(f"{t}: sin CIK")
            continue
        facts = f._descargar_company_facts(cik)
        for tax, tags in facts.get("facts", {}).items():
            for tag, nodo in tags.items():
                if not PALABRAS.search(tag):
                    continue
                for unidad, regs in nodo.get("units", {}).items():
                    regs = [r for r in regs if r.get("form") in FORMAS_VALIDAS]
                    if not regs:
                        continue
                    d = pd.DataFrame(regs).sort_values(["end", "filed"])
                    filas.append({
                        "ticker": t, "tag": f"{tax}:{tag}", "unidad": unidad,
                        "registros": len(d),
                        "primer_fin": d["end"].min(), "ultimo_fin": d["end"].max(),
                        "ultima_publicacion": d["filed"].max(),
                        "ultimo_valor": d.iloc[-1]["val"],
                    })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ["BRK.B", "V"]
    df = diagnosticar(tickers)
    Path("salidas").mkdir(exist_ok=True)
    df.to_csv("salidas/diagnostico_xbrl.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 70)
    for t, g in df.groupby("ticker"):
        print(f"\n===== {t} ({len(g)} tags) =====")
        print(g.drop(columns="ticker").sort_values("ultimo_fin", ascending=False)
              .to_string(index=False))
    print("\nDetalle en salidas/diagnostico_xbrl.csv")