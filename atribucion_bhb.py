"""
atribucion_bhb.py - Atribución de Brinson-Hood-Beebower (BHB) del largo-corto.

PREGUNTA QUE RESPONDE
  ¿La estrategia le gana al benchmark por los SECTORES en los que carga el
  portafolio (asignación) o por las ACCIONES que escoge dentro de cada sector
  (selección)?

CÓMO SE ADAPTA AL LARGO-CORTO
  BHB necesita un portafolio con pesos que suman 1 y un benchmark también.
  Cada pierna cumple (N acciones con 1/N cada una), así que se hace BHB de
  cada pierna contra el MISMO benchmark y se combinan:

      R_largo - R_corto = (R_largo - R_bench) - (R_corto - R_bench)

  El benchmark se cancela. Entonces:
      efecto del largo-corto = efecto de la pierna larga - efecto de la corta
  (estar corto invierte el signo: que las cortas rindan MENOS que el benchmark
  es bueno para la estrategia). En las tablas, la pierna corta se muestra ya
  como APORTE al largo-corto (signo invertido), así que
      aporte larga + aporte corta = rendimiento bruto del largo-corto.

BENCHMARK
  Top 100 igual ponderado del universo de cada mes: las acciones con
  rendimiento esperado y rendimiento real ese mes (el mismo universo del que
  el modelo escoge y el de ret_ew_top100 en estrategia_cuantilica.py).

FÓRMULAS (por sector s y por mes; w = peso, R = rendimiento en el sector)
  asignación   = (w_p - w_b) * R_b
  selección    = w_b * (R_p - R_b)
  interacción  = (w_p - w_b) * (R_p - R_b)
  La suma sobre sectores de los tres es EXACTAMENTE R_portafolio - R_benchmark.
  Si el portafolio no tiene acciones en un sector, R_p = R_b (selección e
  interacción = 0).

Lee:
    salidas/calibracion/finalistas_12_mismas_fechas.csv   (modelo final)
    salidas/calibracion/modeloN/rendimiento_esperado_X.csv
    salidas/estrategia/modeloN_X_mejor_global/posiciones.csv
    salidas/estrategia/modeloN_X_mejor_global/retornos_mensuales.csv (cuadre y costos)
    salidas/regresion/rendimientos_miembros.csv
    salidas/sectores/sectores.csv                          (descargar_sectores.py)
Escribe en salidas/atribucion_bhb/:
    resumen_bhb.csv          efectos anualizados por pierna y total (+ t-stats)
    por_sector.csv           aporte de cada sector al largo-corto
    efectos_mensuales.csv    efectos de cada mes (larga, corta, largo-corto)
    cuadre.csv               chequeo contra estrategia_cuantilica.py
    pesos_netos.csv          peso promedio de cada sector en la larga, la corta y neto
    pesos_netos_por_anio.csv peso neto (larga - corta) de cada sector, por año
    subperiodos_resumen.csv  resumen BHB en primera y segunda mitad
    subperiodos_sector.csv   aporte por sector en primera y segunda mitad
    atribucion_bhb.xlsx      todo lo anterior + definiciones
    atribucion_bhb.png       efectos acumulados y aporte por sector

NOTA SOBRE LA ASIGNACIÓN POR SECTOR
  La asignación BHB clásica multiplica por el rendimiento ABSOLUTO del sector.
  Como casi todos los sectores suben en un periodo largo, el valor de cada
  sector por separado depende del nivel general del mercado (el TOTAL no).
  Por eso por_sector.csv trae también la asignación Brinson-Fachler
  (asignacion_bf_anual), que usa R_b del sector menos R_b total y se lee mejor
  sector por sector. La suma sobre sectores de las dos asignaciones es igual.

Correr parado en la raíz del repo, después de estrategia_cuantilica.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from optimizar_modelos import cargar_rendimientos
from metricas_desempeno import t_stat_media, MESES_ANIO

REGRESION = Path("salidas/regresion")
CALIBRACION = Path("salidas/calibracion")
ESTRATEGIA = Path("salidas/estrategia")
SECTORES = Path("salidas/sectores/sectores.csv")
SALIDA = Path("salidas/atribucion_bhb")

# Si se deja en None, se usa el mejor_global de finalistas_12_mismas_fechas.csv.
# Si se pone (num, version), p. ej. (1, "A"), se usa ese modelo.
MODELO_MANUAL = None
SIN_SECTOR = "Sin sector"
TOLERANCIA_CUADRE = 1e-9

EFECTOS = ["asignacion", "seleccion", "interaccion"]


# Insumos
def leer_modelo() -> tuple[int, str]:
    if MODELO_MANUAL is not None:
        return int(MODELO_MANUAL[0]), str(MODELO_MANUAL[1])
    archivo = CALIBRACION / "finalistas_12_mismas_fechas.csv"
    if not archivo.exists():
        raise FileNotFoundError(
            f"No encontré {archivo}. Corre primero calibrar_modelos.py.")
    df = pd.read_csv(archivo)
    fila = df[df["mejor_global"].astype(str).str.lower().eq("true")].iloc[0]
    return int(fila["modelo"]), str(fila["version"])


def cargar_sectores() -> pd.Series:
    if not SECTORES.exists():
        raise FileNotFoundError(
            f"No encontré {SECTORES}. Corre primero descargar_sectores.py.")
    s = pd.read_csv(SECTORES).drop_duplicates("ticker", keep="last")
    return s.set_index("ticker")["sector"].fillna(SIN_SECTOR)


# Núcleo BHB de UNA pierna en UN mes
def bhb_un_mes(pos: pd.DataFrame, universo: pd.Series, sector: pd.Series) -> pd.DataFrame:
    """pos: DataFrame con columnas ticker, peso (magnitud, suma 1), ret_real.
    universo: rendimientos reales del universo del mes (índice = ticker).
    Regresa un DataFrame por sector con w_p, w_b, R_p, R_b y los tres efectos."""
    sec_u = universo.index.map(lambda t: sector.get(t, SIN_SECTOR))
    sec_p = pos["ticker"].map(lambda t: sector.get(t, SIN_SECTOR))
    sectores = sorted(set(sec_u) | set(sec_p))

    w_b = (universo.groupby(sec_u).size() / len(universo)).reindex(sectores, fill_value=0.0)
    r_b = universo.groupby(sec_u).mean().reindex(sectores).fillna(0.0)

    p = pos.assign(sector=sec_p.values, x=pos["peso"] * pos["ret_real"])
    peso_s = p.groupby("sector")["peso"].sum()
    w_p = peso_s.reindex(sectores, fill_value=0.0)
    r_p = (p.groupby("sector")["x"].sum() / peso_s).reindex(sectores)
    r_p = r_p.fillna(r_b)                     # sin acciones en el sector: R_p = R_b

    out = pd.DataFrame({"w_p": w_p, "w_b": w_b, "r_p": r_p, "r_b": r_b})
    out["asignacion"] = (out["w_p"] - out["w_b"]) * out["r_b"]
    out["seleccion"] = out["w_b"] * (out["r_p"] - out["r_b"])
    out["interaccion"] = (out["w_p"] - out["w_b"]) * (out["r_p"] - out["r_b"])
    # Brinson-Fachler: asignación respecto al rendimiento TOTAL del benchmark.
    # Suma lo mismo que la BHB sobre todos los sectores (porque sum(w_p - w_b) = 0).
    r_bench = float((out["w_b"] * out["r_b"]).sum())
    out["asignacion_bf"] = (out["w_p"] - out["w_b"]) * (out["r_b"] - r_bench)
    out.index.name = "sector"
    return out


def calcular(posiciones: pd.DataFrame, er: pd.DataFrame, real: pd.DataFrame,
             sector: pd.Series):
    """Regresa (efectos mensuales por pierna, detalle mensual por sector y pierna,
    rendimientos de cuadre por mes)."""
    mensual, detalle, cuadre = [], [], []
    omitidos = 0
    for fecha, pos_mes in posiciones.groupby("fecha"):
        if fecha not in er.index or fecha not in real.index:
            omitidos += 1
            continue
        u = pd.concat([er.loc[fecha].rename("er"), real.loc[fecha].rename("real")],
                      axis=1).dropna()
        universo = u["real"]
        r_bench = float(universo.mean())
        r_pierna = {}
        for pierna, lado in (("LARGA", "LARGO"), ("CORTA", "CORTO")):
            p = pos_mes[pos_mes["lado"] == lado][["ticker", "peso", "ret_real"]].copy()
            p["peso"] = p["peso"].abs()
            faltan = set(p["ticker"]) - set(universo.index)
            if faltan:
                raise ValueError(f"{fecha:%Y-%m}: tickers de {lado} fuera del universo "
                                 f"reconstruido: {sorted(faltan)[:5]}. ¿Coincide er con el "
                                 f"que usó estrategia_cuantilica.py?")
            p["peso"] = p["peso"] / p["peso"].sum()
            d = bhb_un_mes(p, universo, sector)
            r_pierna[pierna] = float((d["w_p"] * d["r_p"]).sum())
            # signo del aporte al largo-corto: +1 la larga, -1 la corta
            signo = 1.0 if pierna == "LARGA" else -1.0
            fila = {"fecha": fecha, "pierna": pierna}
            for e in EFECTOS:
                fila[e] = signo * float(d[e].sum())
            mensual.append(fila)
            dd = d.reset_index()
            dd.insert(0, "pierna", pierna)
            dd.insert(0, "fecha", fecha)
            for e in EFECTOS + ["asignacion_bf"]:
                dd[e] = signo * dd[e]
            detalle.append(dd)
        cuadre.append({"fecha": fecha, "bench": r_bench,
                       "ret_largo": r_pierna["LARGA"], "ret_corto": r_pierna["CORTA"]})
    if omitidos:
        print(f"  AVISO: {omitidos} meses de posiciones.csv sin E[R] o rendimiento real; se omiten.")
    if not mensual:
        raise RuntimeError("No hubo ningún mes para calcular la atribución.")
    return (pd.DataFrame(mensual), pd.concat(detalle, ignore_index=True),
            pd.DataFrame(cuadre).set_index("fecha"))


# Resúmenes
def con_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total"] = df[EFECTOS].sum(axis=1)
    return df


def resumen_bhb(mensual: pd.DataFrame, costos: pd.Series | None) -> pd.DataFrame:
    m = con_total(mensual)
    larga = m[m["pierna"] == "LARGA"].set_index("fecha")[EFECTOS + ["total"]]
    corta = m[m["pierna"] == "CORTA"].set_index("fecha")[EFECTOS + ["total"]]
    series = {
        "Pierna larga (aporte)": larga,
        "Pierna corta (aporte, signo invertido)": corta,
        "Largo-corto bruto": larga + corta,
    }
    filas = []
    for nombre, s in series.items():
        fila = {"concepto": nombre}
        for c in EFECTOS + ["total"]:
            fila[f"{c}_anual"] = float(s[c].mean() * MESES_ANIO)
            fila[f"t_{c}"] = t_stat_media(s[c])
        total = fila["total_anual"]
        for c in EFECTOS:
            fila[f"pct_{c}"] = fila[f"{c}_anual"] / total if total else np.nan
        fila["meses"] = len(s)
        filas.append(fila)
    if costos is not None:
        bruto = series["Largo-corto bruto"]["total"]
        cs = costos.reindex(bruto.index)
        filas.append({"concepto": "(-) Costos de operar", "total_anual": float(-cs.mean() * MESES_ANIO),
                      "meses": int(cs.notna().sum())})
        filas.append({"concepto": "Largo-corto neto (aritmético)",
                      "total_anual": float((bruto - cs).mean() * MESES_ANIO),
                      "t_total": t_stat_media((bruto - cs).dropna()),
                      "meses": int(cs.notna().sum())})
    return pd.DataFrame(filas)


def tabla_por_sector(detalle: pd.DataFrame, n_meses: int) -> pd.DataFrame:
    """Aporte promedio anualizado de cada sector, por pierna y combinado.
    Los meses en que un sector no aparece cuentan como 0 (no hubo efecto).
    total_anual = BHB (asignación BHB + selección + interacción);
    total_bf_anual = lo mismo con la asignación Brinson-Fachler."""
    d = con_total(detalle)
    anuales = EFECTOS + ["total", "asignacion_bf"]
    filas = []
    for pierna, g in d.groupby("pierna"):
        s = g.groupby("sector")[["w_p", "w_b"] + anuales].sum() / n_meses
        for c in anuales:
            s[c] = s[c] * MESES_ANIO
        s["sobreponderacion"] = s["w_p"] - s["w_b"]
        s = s.rename(columns={"w_p": "peso_prom_portafolio", "w_b": "peso_prom_benchmark",
                              **{c: f"{c}_anual" for c in anuales}})
        s["total_bf_anual"] = (s["total_anual"] - s["asignacion_anual"] + s["asignacion_bf_anual"])
        s.insert(0, "pierna", pierna)
        filas.append(s.reset_index())
    por_pierna = pd.concat(filas, ignore_index=True)

    suma = ["asignacion_anual", "seleccion_anual", "interaccion_anual", "total_anual",
            "asignacion_bf_anual", "total_bf_anual"]
    comb = por_pierna.groupby("sector")[suma].sum()
    comb.insert(0, "pierna", "LARGO-CORTO")
    pp = por_pierna.pivot(index="sector", columns="pierna", values="peso_prom_portafolio").fillna(0.0)
    comb["peso_neto"] = pp.get("LARGA", 0.0) - pp.get("CORTA", 0.0)      # larga - corta
    comb["peso_prom_benchmark"] = (por_pierna[por_pierna["pierna"] == "LARGA"]
                                   .set_index("sector")["peso_prom_benchmark"])
    comb = comb.reset_index().sort_values("total_anual", ascending=False)
    return pd.concat([comb, por_pierna.sort_values(["pierna", "total_anual"],
                                                   ascending=[True, False])], ignore_index=True)


def pesos_netos(detalle: pd.DataFrame, n_meses: int) -> pd.DataFrame:
    """Peso promedio de cada sector en la pierna larga, en la corta y neto
    (larga - corta), junto al del benchmark."""
    pp = (detalle.pivot_table(index="sector", columns="pierna", values="w_p", aggfunc="sum")
          .fillna(0.0) / n_meses)
    wb = (detalle[detalle["pierna"] == "LARGA"].groupby("sector")["w_b"].sum() / n_meses)
    t = pd.DataFrame({"peso_prom_larga": pp.get("LARGA", 0.0),
                      "peso_prom_corta": pp.get("CORTA", 0.0),
                      "peso_prom_benchmark": wb})
    t["peso_neto"] = t["peso_prom_larga"] - t["peso_prom_corta"]
    return t.sort_values("peso_neto", ascending=False).rename_axis("sector").reset_index()


def pesos_netos_por_anio(detalle: pd.DataFrame) -> pd.DataFrame:
    """Peso neto (larga - corta) promedio de cada sector en cada año calendario:
    sirve para ver si el sesgo de sectores es estable o cambia en el tiempo."""
    wp = (detalle.pivot_table(index=["fecha", "sector"], columns="pierna",
                              values="w_p", aggfunc="sum").fillna(0.0))
    neto = (wp.get("LARGA", 0.0) - wp.get("CORTA", 0.0)).unstack("sector").fillna(0.0)
    return neto.groupby(neto.index.year).mean().rename_axis("anio")


def por_subperiodos(mensual: pd.DataFrame, detalle: pd.DataFrame,
                    costos: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resumen BHB y aporte por sector (largo-corto) en la primera y la segunda
    mitad de los meses (mismo corte que estrategia_cuantilica.py)."""
    fechas = pd.DatetimeIndex(sorted(mensual["fecha"].unique()))
    corte = fechas[len(fechas) // 2]
    partes = (
        (f"Primera mitad ({fechas[0]:%Y-%m} a {fechas[len(fechas) // 2 - 1]:%Y-%m})",
         fechas[fechas < corte]),
        (f"Segunda mitad ({corte:%Y-%m} a {fechas[-1]:%Y-%m})", fechas[fechas >= corte]),
    )
    res, sec = [], []
    for etiqueta, f in partes:
        r = resumen_bhb(mensual[mensual["fecha"].isin(f)], costos)
        r.insert(0, "subperiodo", etiqueta)
        res.append(r)
        s = tabla_por_sector(detalle[detalle["fecha"].isin(f)], len(f))
        s = s[s["pierna"] == "LARGO-CORTO"].copy()
        s.insert(0, "subperiodo", etiqueta)
        sec.append(s)
    return pd.concat(res, ignore_index=True), pd.concat(sec, ignore_index=True)


def tabla_cuadre(cuadre: pd.DataFrame, rm: pd.DataFrame | None) -> pd.DataFrame:
    c = cuadre.copy()
    c["ls_bruto_bhb"] = c["ret_largo"] - c["ret_corto"]
    if rm is not None:
        rm = rm.reindex(c.index)
        c["dif_ret_largo"] = c["ret_largo"] - rm["ret_largo"]
        c["dif_ret_corto"] = c["ret_corto"] - rm["ret_corto"]
        c["dif_ls_bruto"] = c["ls_bruto_bhb"] - rm["ls_bruto"]
        c["dif_benchmark"] = c["bench"] - rm["ret_ew_top100"]
    return c


# Gráfico
def graficar(mensual: pd.DataFrame, sectores: pd.DataFrame, ruta: Path, titulo: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado; se omite el gráfico)")
        return
    m = con_total(mensual).groupby("fecha")[EFECTOS + ["total"]].sum().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for c, nombre in (("asignacion", "Asignación"), ("seleccion", "Selección"),
                      ("interaccion", "Interacción"), ("total", "Total largo-corto")):
        axes[0].plot(m[c].cumsum(), label=nombre, lw=2 if c == "total" else 1.2)
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_title("Efectos BHB acumulados (suma de efectos mensuales)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    s = sectores[sectores["pierna"] == "LARGO-CORTO"].sort_values("total_anual")
    y = np.arange(len(s))
    izq = np.zeros(len(s))
    for c, nombre in (("asignacion_anual", "Asignación"), ("seleccion_anual", "Selección"),
                      ("interaccion_anual", "Interacción")):
        v = s[c].to_numpy()
        axes[1].barh(y, v, left=izq, label=nombre)
        izq = izq + v        # apilado simple (aproximado si hay signos mezclados)
    axes[1].set_yticks(y, s["sector"])
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[1].set_title("Aporte anual de cada sector al largo-corto")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="x")
    fig.suptitle(f"Atribución BHB  |  {titulo}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def definiciones() -> pd.DataFrame:
    return pd.DataFrame([
        ("Benchmark", "Top 100 igual ponderado del universo de cada mes (acciones con E[R] y rendimiento "
                      "real); es el mismo universo del que el modelo escoge"),
        ("Pierna larga / corta", "Cada pierna es un portafolio de N acciones con pesos iguales que suman 1. "
                                 "Se hace BHB de cada una contra el mismo benchmark"),
        ("Largo-corto", "R_largo - R_corto = (R_largo - R_bench) - (R_corto - R_bench). Los efectos de la "
                        "pierna corta se muestran con signo invertido (aporte al largo-corto), de modo que "
                        "aporte larga + aporte corta = largo-corto bruto"),
        ("asignacion", "(w_p - w_b) * R_b por sector: premia sobreponderar sectores cuyo rendimiento en el "
                       "benchmark fue positivo (y subponderar los negativos). Es la versión BHB clásica: "
                       "mide el rendimiento absoluto del sector, no el relativo al benchmark total "
                       "(eso es Brinson-Fachler)"),
        ("seleccion", "w_b * (R_p - R_b) por sector: las acciones escogidas dentro del sector rindieron más "
                      "(o menos) que el promedio del sector en el benchmark"),
        ("interaccion", "(w_p - w_b) * (R_p - R_b): efecto conjunto de asignación y selección"),
        ("Sin residuo", "La suma de los tres efectos sobre todos los sectores es exactamente "
                        "R_portafolio - R_benchmark de cada pierna (ver cuadre.csv)"),
        ("asignacion_bf_anual", "Asignación Brinson-Fachler: (w_p - w_b) * (R_b del sector - R_b total del "
                                "benchmark). Premia sobreponderar sectores que rindieron MÁS que el "
                                "benchmark total. Suma sobre sectores lo mismo que la asignación BHB; "
                                "solo cambia cómo se reparte entre sectores. total_bf_anual = esa "
                                "asignación + selección + interacción"),
        ("peso_neto", "Peso promedio del sector en la pierna larga menos su peso en la corta (suma 0 sobre "
                      "todos los sectores). Positivo = el modelo carga largo en ese sector"),
        ("pesos_netos_por_anio", "Peso neto de cada sector promediado por año calendario: muestra si el "
                                 "sesgo de sectores es estable o cambia con el tiempo"),
        ("Subperiodos", "Primera y segunda mitad de los meses (mismo corte que estrategia_cuantilica.py). "
                        "Sirven para ver si los efectos y el sesgo de sectores se sostienen en ambos"),
        ("Sector sin acciones del portafolio", "Se toma R_p = R_b: selección e interacción = 0 y el efecto "
                                               "es solo de asignación"),
        ("_anual", "Promedio mensual x 12 (aritmético). Los efectos mensuales no se componen entre sí, así "
                   "que no suman al rendimiento compuesto (retorno_anualizado de otras tablas)"),
        ("t_*", "t-stat de la media mensual del efecto (distinto de cero si |t| > 2, aprox.)"),
        ("pct_*", "Fracción del total del largo-corto bruto que explica cada efecto"),
        ("Costos", "BHB trabaja con rendimiento bruto; los costos (COSTO_BPS de estrategia_cuantilica.py) "
                   "se muestran como una línea aparte"),
        ("Sector", "Sector ACTUAL de Yahoo Finance aplicado a todo el periodo (algunas empresas cambiaron "
                   "de sector). Tickers sin sector quedan en 'Sin sector'"),
        ("Ruido", "Con N acciones por pierna repartidas en ~11 sectores, muchos sectores tienen 0-2 acciones; "
                  "el efecto de selección de un mes es ruidoso. Es más confiable el promedio del periodo"),
        ("Circularidad", "El modelo no conoce los sectores; las diferencias de peso por sector son un "
                         "subproducto de lo que prefieren los factores"),
        ("Sesgo", "El modelo se eligió con el mismo periodo que se usa aquí (en-muestra)"),
    ], columns=["concepto", "definición"])


# Programa principal
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    num, version = leer_modelo()
    carpeta = ESTRATEGIA / f"modelo{num}_{version}_mejor_global"
    titulo = f"Modelo {num}, versión {version}"
    print(f"{titulo}\n  estrategia: {carpeta}")
    for ruta in (carpeta / "posiciones.csv", CALIBRACION / f"modelo{num}" /
                 f"rendimiento_esperado_{version}.csv"):
        if not ruta.exists():
            raise FileNotFoundError(f"No encontré {ruta}. Corre estrategia_cuantilica.py / "
                                    "calibrar_modelos.py.")

    posiciones = pd.read_csv(carpeta / "posiciones.csv", parse_dates=["fecha"])
    er = pd.read_csv(CALIBRACION / f"modelo{num}" / f"rendimiento_esperado_{version}.csv",
                     index_col=0, parse_dates=True).sort_index()
    real = cargar_rendimientos(REGRESION / "rendimientos_miembros.csv")
    sector = cargar_sectores()

    rm, costos = None, None
    ruta_rm = carpeta / "retornos_mensuales.csv"
    if ruta_rm.exists():
        rm = pd.read_csv(ruta_rm, index_col=0, parse_dates=True)
        costos = rm["costo_largo"] + rm["costo_corto"]

    tickers = set(posiciones["ticker"])
    sin = sorted(t for t in tickers if t not in sector.index or sector[t] == SIN_SECTOR)
    print(f"  {len(posiciones['fecha'].unique())} meses, {len(tickers)} tickers en el portafolio; "
          f"{len(sin)} sin sector" + (f": {', '.join(sin[:10])}" if sin else ""))

    mensual, detalle, cuadre = calcular(posiciones, er, real, sector)
    n_meses = mensual["fecha"].nunique()

    resumen = resumen_bhb(mensual, costos)
    por_sector = tabla_por_sector(detalle, n_meses)
    chequeo = tabla_cuadre(cuadre, rm)
    efectos_m = con_total(mensual)
    netos = pesos_netos(detalle, n_meses)
    netos_anio = pesos_netos_por_anio(detalle)
    sub_res, sub_sec = por_subperiodos(mensual, detalle, costos)

    # ---- Cuadre ----
    if "dif_ls_bruto" in chequeo.columns:
        peor = chequeo[["dif_ret_largo", "dif_ret_corto", "dif_ls_bruto", "dif_benchmark"]] \
            .abs().max().max()
        estado = "OK" if peor < TOLERANCIA_CUADRE else "REVISAR"
        print(f"\nCuadre contra estrategia_cuantilica.py: diferencia máxima = {peor:.2e}  [{estado}]")
        if estado == "REVISAR":
            print("  Los rendimientos reconstruidos no coinciden con retornos_mensuales.csv. "
                  "Revisa que posiciones.csv y el E[R] sean de la misma corrida.")
    # identidad BHB: suma de efectos = R_pierna - R_bench
    m = efectos_m.pivot(index="fecha", columns="pierna", values="total")
    id_larga = (m["LARGA"] - (cuadre["ret_largo"] - cuadre["bench"])).abs().max()
    id_corta = (m["CORTA"] + (cuadre["ret_corto"] - cuadre["bench"])).abs().max()
    print(f"Identidad BHB (suma de efectos = rendimiento activo): "
          f"larga {id_larga:.2e}, corta {id_corta:.2e}")
    dif_bf = (detalle.groupby(["fecha", "pierna"])[["asignacion", "asignacion_bf"]].sum()
              .diff(axis=1)["asignacion_bf"].abs().max())
    print(f"Asignación BHB vs Brinson-Fachler (el total mensual debe ser igual): {dif_bf:.2e}")
    print(f"Pesos netos: suma sobre sectores = {netos['peso_neto'].sum():.2e} (debe ser ~0)")

    # ---- Guardar ----
    resumen.to_csv(SALIDA / "resumen_bhb.csv", index=False)
    por_sector.to_csv(SALIDA / "por_sector.csv", index=False)
    efectos_m.to_csv(SALIDA / "efectos_mensuales.csv", index=False)
    chequeo.to_csv(SALIDA / "cuadre.csv")
    netos.to_csv(SALIDA / "pesos_netos.csv", index=False)
    netos_anio.to_csv(SALIDA / "pesos_netos_por_anio.csv")
    sub_res.to_csv(SALIDA / "subperiodos_resumen.csv", index=False)
    sub_sec.to_csv(SALIDA / "subperiodos_sector.csv", index=False)
    with pd.ExcelWriter(SALIDA / "atribucion_bhb.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"modelo": [titulo], "carpeta_estrategia": [str(carpeta)],
                      "meses": [n_meses]}).to_excel(xw, sheet_name="Modelo_usado", index=False)
        resumen.to_excel(xw, sheet_name="Resumen", index=False)
        por_sector.to_excel(xw, sheet_name="Por_sector", index=False)
        netos.to_excel(xw, sheet_name="Pesos_netos", index=False)
        netos_anio.to_excel(xw, sheet_name="Pesos_netos_por_anio")
        sub_res.to_excel(xw, sheet_name="Subperiodos_resumen", index=False)
        sub_sec.to_excel(xw, sheet_name="Subperiodos_sector", index=False)
        efectos_m.to_excel(xw, sheet_name="Efectos_mensuales", index=False)
        chequeo.reset_index().to_excel(xw, sheet_name="Cuadre", index=False)
        definiciones().to_excel(xw, sheet_name="Definiciones", index=False)
    graficar(mensual, por_sector, SALIDA / "atribucion_bhb.png", titulo)

    # ---- Consola ----
    pct = lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    t = resumen[["concepto", "asignacion_anual", "seleccion_anual", "interaccion_anual",
                 "total_anual"]].copy()
    for c in t.columns[1:]:
        t[c] = pct(t[c])
    print(f"\n{'=' * 90}\nEfectos BHB anualizados (promedio mensual x 12), {n_meses} meses\n{'=' * 90}")
    print(t.to_string(index=False))
    print(f"\nAporte por sector al largo-corto (anual):")
    s = por_sector[por_sector["pierna"] == "LARGO-CORTO"][
        ["sector", "peso_neto", "asignacion_anual", "asignacion_bf_anual", "seleccion_anual",
         "interaccion_anual", "total_anual", "total_bf_anual"]].copy()
    for c in s.columns[1:]:
        s[c] = pct(s[c])
    print(s.to_string(index=False))
    print("  (peso_neto = larga - corta; asignacion_anual = BHB, asignacion_bf_anual = "
          "Brinson-Fachler; total_bf_anual usa la BF)")

    print(f"\n{'=' * 90}\nSubperiodos: largo-corto bruto\n{'=' * 90}")
    sr = sub_res[sub_res["concepto"] == "Largo-corto bruto"][
        ["subperiodo", "asignacion_anual", "t_asignacion", "seleccion_anual", "t_seleccion",
         "total_anual", "t_total", "meses"]].copy()
    for c in ("asignacion_anual", "seleccion_anual", "total_anual"):
        sr[c] = pct(sr[c])
    print(sr.round(2).to_string(index=False))
    print("\nPeso neto por sector en cada subperiodo (larga - corta):")
    sn = sub_sec.pivot(index="sector", columns="subperiodo", values="peso_neto")
    sn = sn.reindex(netos["sector"]).map(lambda v: f"{v:+.1%}")
    print(sn.to_string())
    print(f"\nResultados completos en {SALIDA}")


if __name__ == "__main__":
    main()