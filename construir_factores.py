"""
construir_factores.py - Parte 1 (final): convierte las características de las
100 empresas en SERIES DE TIEMPO DE FACTORES, listas para la Parte 2
(regresión de cada acción contra los factores, con suma de betas = 1).

Lee:
    salidas/empresas/<TICKER>.xlsx   (hoja Mensual)
    salidas/mercado.xlsx             (S&P 500 y T-bill 13 semanas)
    seleccion_anual_top100.csv       qué empresas están en el top 100 cada año
Escribe en salidas/regresion/:
    rendimientos.csv     fecha x ticker   rendimiento mensual (precio ajustado),
                                          todos los meses con precio
    rendimientos_miembros.csv             lo mismo, pero solo los meses en que
                                          la empresa estaba en el top 100
    membresia.csv        fecha x ticker   1 = en el top 100 ese mes
    factores.csv         fecha x factor   rf, MERCADO, VALUE, TAMANO, MOMENTUM,
                                          VOLATILIDAD, LIQUIDEZ
    caracteristicas.csv  formato largo    fecha, ticker, rendimiento, pe,
                                          market_cap, momentum_12_1,
                                          volatilidad_12m, rotacion_3m
    insumos_regresion.xlsx  lo mismo en un Excel + hoja "Definiciones"

CÓMO SE CONSTRUYE CADA FACTOR (estilo Fama-French)
  - Cada fin de mes t-1 se ordenan las empresas por su característica
    (con datos conocidos en t-1) y se arman dos portafolios: el 30% de un
    extremo y el 30% del otro. El factor del mes t es el rendimiento del
    primero menos el del segundo. Así no hay look-ahead.
  - UNIVERSO ANUAL: el mes t solo entran las empresas del top 100 del año
    de t (escogidas al cierre de diciembre anterior). Las características
    (momentum, volatilidad) sí usan la historia previa de la empresa aunque
    todavía no estuviera en el top 100.
  - Mínimo MIN_EMPRESAS con dato ese mes; si no, el factor queda vacío.
  - Ponderación: PONDERACION = "igual" o "market_cap" (market cap de t-1).

  MERCADO     rendimiento del S&P 500 (^GSPC) - rf
  VALUE       P/E BAJO - P/E ALTO   (solo empresas con EPS TTM > 0)
  TAMANO      market cap CHICO - GRANDE
  MOMENTUM    rendimiento t-12..t-2 ALTO - BAJO (se salta el último mes)
  VOLATILIDAD volatilidad 12m BAJA - ALTA
  LIQUIDEZ    rotación 3m BAJA (ilíquidas) - ALTA
  rf          T-bill 13 semanas (^IRX, % anual) / 12, conocido al inicio del mes

Todos los parámetros están arriba para que el equipo los ajuste.
Requiere: pandas, openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd

CARPETA = Path("salidas")
SALIDA = CARPETA / "regresion"
SELECCION = Path("seleccion_anual_top100.csv")   # salida de top100.py
FECHA_INICIO = "2010-01-01"   # primer mes de los factores (regla del profesor)

CORTE = 0.30            # 30% de cada extremo (terciles)
MIN_EMPRESAS = 15       # empresas con dato para calcular el factor ese mes
PONDERACION = "igual"   # "igual" o "market_cap"
MESES_MOMENTUM = 12     # ventana t-12..t-2
MESES_VOL = 12          # volatilidad de 12 meses
MIN_OBS_VOL = 10
MESES_ROTACION = 3

# Factores muy correlacionados vuelven inestables las betas de la regresión.
# Si se activa, el factor de la izquierda se reemplaza por su residuo contra
# el de la derecha (lo que tiene de información PROPIA), conservando su media.
# Ej.: ORTOGONALIZAR = {"LIQUIDEZ": "VOLATILIDAD"}
ORTOGONALIZAR: dict[str, str] = {"LIQUIDEZ": "VOLATILIDAD"}
UMBRAL_CORRELACION = 0.80   # avisa si dos factores pasan este nivel

# (nombre, característica, extremo largo): "bajo" = largo en los valores bajos
FACTORES = [
    ("VALUE", "pe", "bajo"),
    ("TAMANO", "market_cap", "bajo"),
    ("MOMENTUM", "momentum_12_1", "alto"),
    ("VOLATILIDAD", "volatilidad_12m", "bajo"),
    ("LIQUIDEZ", "rotacion_3m", "bajo"),
]


def cargar_empresas(carpeta: Path) -> dict[str, pd.DataFrame]:
    """{campo: DataFrame fecha x ticker}"""
    campos = {"precio_ajustado": {}, "pe": {}, "market_cap": {}, "rotacion": {}}
    for arch in sorted((carpeta / "empresas").glob("*.xlsx")):
        try:
            m = pd.read_excel(arch, sheet_name="Mensual")
        except Exception as e:
            print(f"  {arch.name}: no se pudo leer ({e})")
            continue
        if "fecha" not in m or "precio_ajustado" not in m:
            print(f"  {arch.name}: sin precios, se omite")
            continue
        m["fecha"] = pd.to_datetime(m["fecha"])
        m = m.set_index("fecha").sort_index()
        for c in campos:
            if c in m:
                campos[c][arch.stem] = m[c]
    return {c: pd.DataFrame(v).sort_index() for c, v in campos.items()}


def cargar_mercado(carpeta: Path) -> pd.DataFrame:
    m = pd.read_excel(carpeta / "mercado.xlsx")
    m["fecha"] = pd.to_datetime(m["fecha"])
    return m.set_index("fecha").sort_index()


def cargar_membresia(indice: pd.Index, columnas: pd.Index,
                     archivo: Path = SELECCION) -> pd.DataFrame:
    """fecha x ticker, True si la empresa está en el top 100 del año de esa
    fecha. Los tickers se escriben igual que los Excel (BRK.B -> BRK-B)."""
    if not archivo.exists():
        print(f"  AVISO: no existe {archivo}; se usan TODAS las empresas todos los meses.")
        return pd.DataFrame(True, index=indice, columns=columnas)
    sel = pd.read_csv(archivo).dropna(subset=["Ticker"])
    sel["col"] = sel["Ticker"].astype(str).str.replace(".", "-", regex=False)
    miembro = pd.DataFrame(False, index=indice, columns=columnas)
    anios = pd.Index(indice).year
    for anio, grupo in sel.groupby("Year"):
        cols = [c for c in grupo["col"] if c in miembro.columns]
        miembro.loc[anios == anio, cols] = True
    faltan = sorted(set(sel["col"]) - set(columnas))
    if faltan:
        print(f"  {len(faltan)} tickers de la selección sin Excel de precios "
              f"(no entran a los factores): {', '.join(faltan[:15])}{' ...' if len(faltan) > 15 else ''}")
    return miembro


def caracteristicas(datos: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    precio = datos["precio_ajustado"]
    ret = precio.pct_change(fill_method=None)
    return {
        "rendimiento": ret,
        "pe": datos["pe"].where(datos["pe"] > 0),
        "market_cap": datos["market_cap"],
        # en la fecha d: P(d-1) / P(d-12) - 1  -> rendimiento t-12..t-2 visto en t-1
        "momentum_12_1": precio.shift(1) / precio.shift(MESES_MOMENTUM) - 1,
        "volatilidad_12m": ret.rolling(MESES_VOL, min_periods=MIN_OBS_VOL).std(),
        "rotacion_3m": datos["rotacion"].rolling(MESES_ROTACION, min_periods=2).mean(),
    }


def factor_long_short(ret: pd.DataFrame, carac: pd.DataFrame, largo: str,
                      pesos: pd.DataFrame | None,
                      miembro: pd.DataFrame | None = None) -> pd.Series:
    """Rendimiento mensual del portafolio largo-corto. La característica y los
    pesos se toman de t-1 (shift) y el rendimiento de t. Solo entran las
    empresas del top 100 del año de t (miembro)."""
    x_prev = carac.shift(1).reindex(ret.index)
    w_prev = pesos.shift(1).reindex(ret.index) if pesos is not None else None
    salida = pd.Series(np.nan, index=ret.index)
    for fecha in ret.index:
        x, r = x_prev.loc[fecha], ret.loc[fecha]
        ok = x.notna() & r.notna()
        if miembro is not None:
            ok &= miembro.loc[fecha]
        if w_prev is not None:
            ok &= w_prev.loc[fecha].notna() & (w_prev.loc[fecha] > 0)
        if ok.sum() < MIN_EMPRESAS:
            continue
        x, r = x[ok], r[ok]
        bajo = x <= x.quantile(CORTE)
        alto = x >= x.quantile(1 - CORTE)

        def prom(mask):
            if w_prev is None:
                return r[mask].mean()
            w = w_prev.loc[fecha][ok][mask]
            return (r[mask] * w).sum() / w.sum()

        rb, ra = prom(bajo), prom(alto)
        salida[fecha] = (rb - ra) if largo == "bajo" else (ra - rb)
    return salida


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("Leyendo Excel de empresas...")
    datos = cargar_empresas(CARPETA)
    n = datos["precio_ajustado"].shape[1]
    print(f"  {n} empresas")
    car = caracteristicas(datos)
    ret = car["rendimiento"]
    miembro = cargar_membresia(ret.index, ret.columns)
    n_mes = miembro.sum(axis=1)
    print(f"  Empresas del top 100 con precio por mes: mín {n_mes[n_mes > 0].min() if (n_mes > 0).any() else 0}, "
          f"máx {n_mes.max()}")

    mercado = cargar_mercado(CARPETA)
    rf = (mercado["tbill_13w"] / 100 / 12).shift(1).reindex(ret.index)   # conocido al inicio del mes
    ret_sp = mercado["sp500"].pct_change(fill_method=None).reindex(ret.index)

    factores = pd.DataFrame(index=ret.index)
    factores["rf"] = rf
    factores["MERCADO"] = ret_sp - rf
    pesos = car["market_cap"] if PONDERACION == "market_cap" else None
    for nombre, c, largo in FACTORES:
        factores[nombre] = factor_long_short(ret, car[c], largo, pesos, miembro)
    for f, base in ORTOGONALIZAR.items():
        par = factores[[f, base]].dropna()
        if len(par) > 24:
            b = np.polyfit(par[base], par[f], 1)[0]
            factores[f] = factores[f] - b * (factores[base] - par[base].mean())
            print(f"  {f} ortogonalizado contra {base} (pendiente {b:.2f})")
    factores = factores.iloc[1:]           # el primer mes no tiene rendimiento
    ret = ret.iloc[1:]
    # Ventana de la regla del profesor
    factores = factores[factores.index >= FECHA_INICIO]
    ret = ret[ret.index >= FECHA_INICIO]
    miembro = miembro.reindex(ret.index).fillna(False).astype(bool)
    ret = ret.loc[:, miembro.any()]        # solo empresas que estuvieron en el top 100
    miembro = miembro[ret.columns]
    ret_miembros = ret.where(miembro)

    # formato largo de características (respaldo / otras especificaciones)
    car_m = {k: v.reindex(index=ret.index, columns=ret.columns) for k, v in car.items()}
    car_m["miembro"] = miembro.astype(int)
    largo = pd.concat({k: v.stack(future_stack=True) for k, v in car_m.items()}, axis=1)
    largo.index.names = ["fecha", "ticker"]
    largo = largo.reset_index().dropna(subset=["rendimiento"])

    fmt = lambda df: df.rename_axis("fecha").reset_index().assign(
        fecha=lambda d: pd.to_datetime(d["fecha"]).dt.date)
    fmt(ret).to_csv(SALIDA / "rendimientos.csv", index=False)
    fmt(ret_miembros).to_csv(SALIDA / "rendimientos_miembros.csv", index=False)
    fmt(miembro.astype(int)).to_csv(SALIDA / "membresia.csv", index=False)
    fmt(factores).to_csv(SALIDA / "factores.csv", index=False)
    largo.assign(fecha=largo["fecha"].dt.date).to_csv(SALIDA / "caracteristicas.csv", index=False)

    definiciones = pd.DataFrame([
        ("rf", "T-bill 13 semanas (^IRX) / 100 / 12, del cierre del mes anterior"),
        ("MERCADO", "Rendimiento mensual del S&P 500 (^GSPC, índice de precio) menos rf"),
        ("VALUE", "Portafolio 30% P/E más bajo menos 30% P/E más alto (EPS TTM > 0)"),
        ("TAMANO", "30% market cap más chico menos 30% más grande"),
        ("MOMENTUM", "30% mayor rendimiento t-12..t-2 menos 30% menor"),
        ("VOLATILIDAD", "30% menor volatilidad (12 meses) menos 30% mayor"),
        ("LIQUIDEZ", "30% menor rotación (volumen/acciones, prom. 3 meses) menos 30% mayor"),
        ("Formación", "Orden con datos al cierre de t-1; rendimiento del mes t (sin look-ahead)"),
        ("Universo", "Cada mes t: top 100 por market cap del S&P 500 al 31/12 del año anterior "
                     "(seleccion_anual_top100.csv); ver membresia.csv"),
        ("Ventana", f"Desde {FECHA_INICIO[:7]}"),
        ("Ponderación", PONDERACION),
        ("Ortogonalización", ", ".join(f"{k} vs {v}" for k, v in ORTOGONALIZAR.items()) or "ninguna"),
        ("Mínimo de empresas", f"{MIN_EMPRESAS} con dato; si no, el factor queda vacío ese mes"),
        ("rendimientos.csv", "Rendimiento simple mensual con precio ajustado (incluye dividendos), todos los meses"),
        ("rendimientos_miembros.csv", "Igual, pero vacío en los meses en que la empresa no estaba en el top 100"),
    ], columns=["concepto", "definición"])
    with pd.ExcelWriter(SALIDA / "insumos_regresion.xlsx", engine="openpyxl") as xw:
        fmt(ret).to_excel(xw, sheet_name="Rendimientos", index=False)
        fmt(factores).to_excel(xw, sheet_name="Factores", index=False)
        fmt(ret_miembros).to_excel(xw, sheet_name="Rendimientos_miembros", index=False)
        fmt(miembro.astype(int)).to_excel(xw, sheet_name="Membresia", index=False)
        definiciones.to_excel(xw, sheet_name="Definiciones", index=False)

    print("\nCobertura de cada factor (primer y último mes con dato):")
    for c in factores.columns:
        s = factores[c].dropna()
        if len(s):
            print(f"  {c:<12} {s.index.min():%Y-%m} a {s.index.max():%Y-%m}  "
                  f"({len(s)} meses)  media {s.mean():+.4f}  desv {s.std():.4f}")
        else:
            print(f"  {c:<12} SIN DATOS")
    print("\nCorrelación entre factores:")
    corr = factores.drop(columns="rf").corr()
    print(corr.round(2).to_string())
    altos = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i + 1:]
             if abs(corr.loc[a, b]) > UMBRAL_CORRELACION]
    for a, b, c in altos:
        print(f"  AVISO: {a} y {b} tienen correlación {c:.2f}: en la regresión sus betas "
              f"serán inestables. Considera ORTOGONALIZAR o quitar uno.")
    comunes = factores.dropna()
    if len(comunes):
        print(f"\nMeses con TODOS los factores: {comunes.index.min():%Y-%m} a "
              f"{comunes.index.max():%Y-%m} ({len(comunes)} meses) -> ventana de la regresión")
    print(f"\nArchivos en {SALIDA}")


if __name__ == "__main__":
    main()