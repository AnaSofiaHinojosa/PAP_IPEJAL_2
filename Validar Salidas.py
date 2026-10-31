"""
validar_salidas.py - Revisa los Excel de salidas/empresas y marca datos
sospechosos ANTES de usarlos para construir factores.

Qué busca (por empresa):
  ACCIONES
    - escala: una observación a >5x o <0.2x de la mediana de la empresa
      (típico de XBRL mal etiquetado: acciones x1,000 o x1,000,000)
    - salto: cambio trimestral de más de +-50% que no coincide con un split
  EPS
    - escala: EPS trimestral 1,000 veces menor a la mediana (|EPS| casi cero)
    - extremo: |EPS| trimestral >5x la mediana de sus 8 vecinos y >3x
      el mismo trimestre de años vecinos (estacionalidad) (puede ser real:
      ganancias/pérdidas extraordinarias; hay que revisarlo a mano)
    - factores: factores de ajuste distintos dentro de la misma fecha de
      publicación (señal de split mal aplicado)
  MENSUAL
    - market cap fuera de [1e9, 1e13] USD
    - rotación mensual (volumen / acciones) < 0.0005 (ALTA) o > 2 (MEDIA)
    - P/E > 200 (informativo)
    - huecos en medio de la serie de P/E o market cap

Uso:
    python validar_salidas.py                  # lee salidas/empresas
    python validar_salidas.py otra/carpeta

Salida: salidas/validacion.csv  (una fila por alerta) y un resumen en consola.
Severidad: ALTA = casi seguro error de datos; MEDIA = revisar; INFO = contexto.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def _hoja(archivo: Path, nombre: str) -> pd.DataFrame:
    try:
        return pd.read_excel(archivo, sheet_name=nombre)
    except ValueError:  # la hoja no existe
        return pd.DataFrame()


def _alerta(lista, ticker, hoja, sev, tipo, fecha, detalle):
    lista.append({"ticker": ticker, "hoja": hoja, "severidad": sev, "tipo": tipo,
                  "fecha": pd.Timestamp(fecha).date() if pd.notna(fecha) else None,
                  "detalle": detalle})


def validar_acciones(t, acc, alertas):
    if acc.empty or "acciones" not in acc:
        return
    a = acc.sort_values("end").reset_index(drop=True)
    med = a["acciones"].median()
    for _, r in a.iterrows():
        ratio = r["acciones"] / med
        if ratio > 5 or ratio < 0.2:
            _alerta(alertas, t, "Acciones", "ALTA", "escala", r["end"],
                    f"{r['acciones']:,.0f} acciones = {ratio:,.3g}x la mediana "
                    f"({med:,.0f}); filed {pd.Timestamp(r['filed']).date()}")
    cambio = a["acciones"].pct_change()
    for i in cambio.index[cambio.abs() > 0.5]:
        # si ya es alerta de escala, no duplicar
        ratio = a.loc[i, "acciones"] / med
        if 0.2 <= ratio <= 5:
            _alerta(alertas, t, "Acciones", "MEDIA", "salto", a.loc[i, "end"],
                    f"cambio de {cambio[i]:+.0%} vs trimestre anterior "
                    "(¿fusión, escisión o split mal aplicado?)")


def validar_eps(t, eps, alertas):
    if eps.empty or "eps_trimestral" not in eps:
        return
    e = eps.sort_values("end").reset_index(drop=True)
    med_abs = e["eps_trimestral"].abs().median()
    # mediana móvil de |EPS| en los 8 trimestres vecinos (así no marca
    # empresas estacionales como INTU ni el crecimiento de largo plazo)
    absx = e["eps_trimestral"].abs()
    vecinos = absx.rolling(9, center=True, min_periods=4).median()
    # mismo trimestre fiscal en años vecinos (t-8, t-4, t+4, t+8)
    mismo_q = pd.concat([absx.shift(k) for k in (-8, -4, 4, 8)], axis=1).median(axis=1)
    if med_abs and med_abs > 0:
        for i, r in e.iterrows():
            x = abs(r["eps_trimestral"])
            ref = vecinos[i] if pd.notna(vecinos[i]) and vecinos[i] > 0 else med_abs
            if 0 < x < med_abs / 1000:
                _alerta(alertas, t, "EPS", "ALTA", "escala", r["end"],
                        f"EPS {r['eps_trimestral']:.3g} ~0 vs mediana |EPS| {med_abs:.3g} "
                        "(acciones o utilidad con escala errónea)")
            elif x > 5 * ref and not (pd.notna(mismo_q[i]) and x <= 3 * mismo_q[i]):
                _alerta(alertas, t, "EPS", "ALTA" if x > 50 * ref else "MEDIA", "extremo", r["end"],
                        f"EPS {r['eps_trimestral']:.3g} = {x / ref:.0f}x la mediana de sus "
                        "trimestres vecinos (¿evento extraordinario real o error?)")
    if "factor" in e and "filed" in e:
        n = e.groupby("filed")["factor"].nunique()
        for f in n.index[n > 1]:
            _alerta(alertas, t, "EPS", "ALTA", "factor", f,
                    "factores de split distintos en la misma fecha de publicación")


def validar_mensual(t, m, alertas):
    if m.empty or "fecha" not in m:
        return
    m = m.sort_values("fecha")
    if "market_cap" in m:
        mc = m.dropna(subset=["market_cap"])
        malos = mc[(mc["market_cap"] > 1e13) | (mc["market_cap"] < 1e9)]
        if len(malos):
            _alerta(alertas, t, "Mensual", "ALTA", "market_cap", malos["fecha"].iloc[0],
                    f"{len(malos)} meses con market cap fuera de [1e9, 1e13] "
                    f"(de {malos['fecha'].min():%Y-%m} a {malos['fecha'].max():%Y-%m}; "
                    f"máx {malos['market_cap'].max():.3g})")
    if "rotacion" in m:
        ro = m.dropna(subset=["rotacion"])
        bajos = ro[ro["rotacion"] < 0.0005]   # típico de acciones con escala x1000+
        if len(bajos):
            _alerta(alertas, t, "Mensual", "ALTA", "rotacion", bajos["fecha"].iloc[0],
                    f"{len(bajos)} meses con rotación < 0.0005 "
                    f"(de {bajos['fecha'].min():%Y-%m} a {bajos['fecha'].max():%Y-%m})")
        altos = ro[ro["rotacion"] > 2]
        if len(altos):
            _alerta(alertas, t, "Mensual", "MEDIA", "rotacion", altos["fecha"].iloc[0],
                    f"{len(altos)} meses con rotación > 2 (más de 2 veces las acciones "
                    f"en un mes; de {altos['fecha'].min():%Y-%m} a {altos['fecha'].max():%Y-%m})")
    if "pe" in m:
        altos = m[m["pe"] > 200]
        if len(altos):
            _alerta(alertas, t, "Mensual", "INFO", "pe_alto", altos["fecha"].iloc[0],
                    f"{len(altos)} meses con P/E > 200 (máx {altos['pe'].max():.0f})")
    for col in ("pe", "market_cap"):
        if col not in m or m[col].notna().sum() == 0:
            continue
        s = m.set_index("fecha")[col]
        tramo = s.loc[s.first_valid_index():s.last_valid_index()]
        huecos = int(tramo.isna().sum())
        if huecos:
            # EPS TTM <= 0 deja P/E vacío a propósito; no es error
            nota = " (incluye meses con EPS TTM <= 0)" if col == "pe" else ""
            _alerta(alertas, t, "Mensual", "INFO", f"huecos_{col}", tramo.index[tramo.isna()][0],
                    f"{huecos} meses vacíos en medio de la serie{nota}")


def main(carpeta: str = "salidas/empresas"):
    carpeta = Path(carpeta)
    archivos = sorted(carpeta.glob("*.xlsx"))
    if not archivos:
        print(f"No hay archivos .xlsx en {carpeta}")
        return
    alertas = []
    for arch in archivos:
        t = arch.stem
        validar_acciones(t, _hoja(arch, "Acciones"), alertas)
        validar_eps(t, _hoja(arch, "EPS"), alertas)
        validar_mensual(t, _hoja(arch, "Mensual"), alertas)

    df = pd.DataFrame(alertas, columns=["ticker", "hoja", "severidad", "tipo", "fecha", "detalle"])
    orden = {"ALTA": 0, "MEDIA": 1, "INFO": 2}
    df = df.sort_values(["severidad", "ticker", "fecha"], key=lambda s: s.map(orden) if s.name == "severidad" else s)
    salida = carpeta.parent / "validacion.csv"
    df.to_csv(salida, index=False, encoding="utf-8-sig")

    print(f"Revisé {len(archivos)} empresas -> {len(df)} alertas en {salida}")
    if not df.empty:
        print(df.groupby(["severidad", "tipo"]).size().to_string())
        altas = df[df["severidad"] == "ALTA"]["ticker"].unique()
        print(f"\nEmpresas con alertas ALTAS ({len(altas)}): {', '.join(altas) or 'ninguna'}")


if __name__ == "__main__":
    main(*sys.argv[1:])