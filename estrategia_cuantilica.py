from pathlib import Path
import numpy as np
import pandas as pd

from optimizar_modelos import cargar_rendimientos, cargar_factores
from metricas_desempeno import (
    resumen_metricas, comparar_pareado, sharpe_clasico, sharpe_movil,
    curva_capital, serie_drawdown, t_stat_media, treynor, MESES_ANIO,
)

# holaaa

REGRESION = Path("salidas/regresion")
CALIBRACION = Path("salidas/calibracion")
SALIDA_RAIZ = Path("salidas/estrategia")

# Parámetros ajustables
N_LARGO = 20                 # acciones largas (mayor rendimiento esperado)
N_CORTO = 20                 # acciones cortas (menor rendimiento esperado)
N_CUANTILES = 5              # quintiles para la tabla por cuantiles
# costo por cada 100% operado, en puntos base (0 = sin costos)
COSTO_BPS = 10.0
VENTANA_SHARPE_MOVIL = 36    # meses
# Si se deja en None, se corren TODOS los modelos de finalistas_12_mismas_fechas.csv.
# Si se pone (num, version), se corre SOLO ese modelo (útil para pruebas rápidas).
MODELO_MANUAL = None

# Nombres de las estrategias en las tablas
LS_BRUTO = "Largo-Corto (bruto)"
LS_NETO = "Largo-Corto (neto de costos)"
SOLO_LARGO = "Solo largo, top N (neto)"
SOLO_CORTO = "Solo corto, bottom N (neto)"
PASIVA_EW = "Pasiva: Top 100 igual ponderado"
PASIVA_SP = "Pasiva: S&P 500"


# Selección del/los modelo(s)

def leer_modelos_a_correr() -> list[tuple[int, str, str, bool]]:
    """Lista de (número de modelo, versión, descripción, es_mejor_global) a correr.

    Si MODELO_MANUAL está definido, regresa solo ese. Si no, regresa TODAS las
    filas de finalistas_12_mismas_fechas.csv (no solo la marcada mejor_global)."""
    if MODELO_MANUAL is not None:
        num, version = MODELO_MANUAL
        return [(int(num), str(version), "(forzado con MODELO_MANUAL)", True)]
    archivo = CALIBRACION / "finalistas_12_mismas_fechas.csv"
    if not archivo.exists():
        raise FileNotFoundError(
            f"No encontré {archivo}. Corre primero calibrar_modelos.py.")
    df = pd.read_csv(archivo)
    es_mejor = df["mejor_global"].astype(str).str.lower().eq("true")
    return [
        (int(fila["modelo"]), str(fila["version"]), str(fila["parametros"]), bool(mejor))
        for (_, fila), mejor in zip(df.iterrows(), es_mejor)
    ]


def cargar_rendimiento_esperado(num: int, version: str) -> pd.DataFrame:
    ruta = CALIBRACION / f"modelo{num}" / f"rendimiento_esperado_{version}.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encontré {ruta}. Solo se guardan A, mejor B y mejor C de cada modelo; "
            f"revisa MODELO_MANUAL o vuelve a correr calibrar_modelos.py.")
    er = pd.read_csv(ruta, index_col=0, parse_dates=True)
    return er.sort_index()

# Construcción de los portafolios mes a mes


def construir_portafolios(er: pd.DataFrame, real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regresa (retornos mensuales, posiciones).

    Universo de cada mes: acciones con rendimiento esperado y rendimiento real
    (miembros del Top 100 ese mes), igual que en el cálculo del IC."""
    costo = COSTO_BPS / 10_000.0
    fechas = er.index.intersection(real.index)
    filas, posiciones = [], []
    w_largo_ant = pd.Series(dtype=float)
    w_corto_ant = pd.Series(dtype=float)

    for fecha in fechas:
        u = pd.concat([er.loc[fecha].rename("er"),
                      real.loc[fecha].rename("real")], axis=1).dropna()
        if len(u) < N_LARGO + N_CORTO:
            continue
        u = u.sort_values("er", ascending=False, kind="stable")
        largo, corto = u.head(N_LARGO), u.tail(N_CORTO)

        w_largo = pd.Series(1.0 / N_LARGO, index=largo.index)
        # magnitud; la posición es -w
        w_corto = pd.Series(1.0 / N_CORTO, index=corto.index)
        oper_largo = w_largo.sub(w_largo_ant, fill_value=0.0).abs().sum()
        oper_corto = w_corto.sub(w_corto_ant, fill_value=0.0).abs().sum()
        w_largo_ant, w_corto_ant = w_largo, w_corto

        ret_largo, ret_corto = largo["real"].mean(), corto["real"].mean()
        filas.append({
            "fecha": fecha,
            "n_universo": len(u),
            "er_largo": largo["er"].mean(),
            "er_corto": corto["er"].mean(),
            # rendimiento esperado
            "er_largo_corto": largo["er"].mean() - corto["er"].mean(),
            "ret_largo": ret_largo,
            "ret_corto": ret_corto,
            "ls_bruto": ret_largo - ret_corto,
            "operado_largo": oper_largo,
            "operado_corto": oper_corto,
            "costo_largo": oper_largo * costo,
            "costo_corto": oper_corto * costo,
            "ret_ew_top100": float(u["real"].mean()),
        })
        for t, fila in largo.iterrows():
            posiciones.append({"fecha": fecha, "ticker": t, "lado": "LARGO",
                               "peso": 1.0 / N_LARGO, "er": fila["er"], "ret_real": fila["real"]})
        for t, fila in corto.iterrows():
            posiciones.append({"fecha": fecha, "ticker": t, "lado": "CORTO",
                               "peso": -1.0 / N_CORTO, "er": fila["er"], "ret_real": fila["real"]})

    if not filas:
        raise RuntimeError(
            "No hubo ningún mes con suficientes acciones para armar el portafolio.")
    ret = pd.DataFrame(filas).set_index("fecha").sort_index()
    ret["ls_neto"] = ret["ls_bruto"] - ret["costo_largo"] - ret["costo_corto"]
    ret["largo_neto"] = ret["ret_largo"] - ret["costo_largo"]
    ret["corto_neto"] = -ret["ret_corto"] - \
        ret["costo_corto"]       # P&L de estar corto
    return ret, pd.DataFrame(posiciones)

# Tablas


def tabla_cuantiles(er: pd.DataFrame, real: pd.DataFrame, n_q: int) -> pd.DataFrame:
    """Rendimiento real promedio por cuantil del rendimiento esperado
    (cuantil 1 = menor esperado, cuantil n_q = mayor esperado)."""
    filas = {}
    for fecha in er.index.intersection(real.index):
        u = pd.concat([er.loc[fecha].rename("er"),
                      real.loc[fecha].rename("real")], axis=1).dropna()
        if len(u) < n_q * 2:
            continue
        q = pd.qcut(u["er"].rank(method="first"), n_q, labels=False) + 1
        filas[fecha] = u.groupby(q)["real"].mean()
    m = pd.DataFrame(filas).T.sort_index()
    spread = m[n_q] - m[1]
    tabla = pd.DataFrame({
        "retorno_medio_mensual": m.mean(),
        "retorno_anualizado": (1 + m).prod() ** (MESES_ANIO / len(m)) - 1,
        "vol_anual": m.std(ddof=1) * np.sqrt(MESES_ANIO),
    })
    tabla.index = [f"Q{i}" + (" (menor E[R])" if i == 1 else " (mayor E[R])" if i == n_q else "")
                   for i in tabla.index]
    tabla.loc[f"Q{n_q} - Q1"] = pd.Series({
        "retorno_medio_mensual": spread.mean(),
        "retorno_anualizado": (1 + spread).prod() ** (MESES_ANIO / len(spread)) - 1,
        "vol_anual": spread.std(ddof=1) * np.sqrt(MESES_ANIO),
    })
    tabla["t_stat_media"] = [t_stat_media(
        m[c]) for c in m.columns] + [t_stat_media(spread)]
    tabla["meses"] = len(m)
    return tabla


def por_anio(retornos: pd.DataFrame) -> pd.DataFrame:
    """Rendimiento compuesto de cada estrategia en cada año calendario."""
    return retornos.groupby(retornos.index.year).apply(lambda g: (1 + g).prod() - 1) \
                   .rename_axis("anio")


def metricas_de_todas(estrategias: pd.DataFrame, excesos: pd.DataFrame, mercado_exceso) -> pd.DataFrame:
    filas = {}
    for nombre in estrategias.columns:
        filas[nombre] = resumen_metricas(
            estrategias[nombre], excesos[nombre], mercado_exceso)
    df = pd.DataFrame(filas).T.rename_axis("estrategia")
    if "beta_mercado" in df.columns:
        df["treynor"] = [
            treynor(excesos[nombre], df.loc[nombre, "beta_mercado"])
            for nombre in df.index
        ]
    else:
        df["treynor"] = np.nan
    return df


def comparacion_vs_pasivas(estrategias: pd.DataFrame, excesos: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for activa in (LS_NETO, SOLO_LARGO):
        for pasiva in (PASIVA_EW, PASIVA_SP):
            if pasiva not in estrategias.columns:
                continue
            c = comparar_pareado(estrategias[activa], estrategias[pasiva])
            filas.append({
                "estrategia": activa, "contra": pasiva,
                "sharpe_estrategia": sharpe_clasico(excesos[activa]),
                "sharpe_pasiva": sharpe_clasico(excesos[pasiva]),
                "dif_sharpe": sharpe_clasico(excesos[activa]) - sharpe_clasico(excesos[pasiva]),
                **c,
            })
    return pd.DataFrame(filas)


# Gráfico

def graficar(estrategias: pd.DataFrame, excesos: pd.DataFrame, ruta: Path, titulo: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado; se omite el gráfico)")
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    for nombre in (LS_NETO, SOLO_LARGO, PASIVA_EW, PASIVA_SP):
        if nombre in estrategias.columns:
            axes[0].plot(curva_capital(estrategias[nombre]), label=nombre)
    axes[0].set_title(f"Valor de $1 invertido  |  {titulo}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for nombre in (LS_NETO, PASIVA_EW, PASIVA_SP):
        if nombre in estrategias.columns:
            axes[1].plot(serie_drawdown(estrategias[nombre]), label=nombre)
    axes[1].set_title("Drawdown")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for nombre in (LS_NETO, PASIVA_EW, PASIVA_SP):
        if nombre in excesos.columns:
            axes[2].plot(sharpe_movil(excesos[nombre],
                         VENTANA_SHARPE_MOVIL), label=nombre)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_title(f"Sharpe clásico móvil de {VENTANA_SHARPE_MOVIL} meses")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def definiciones() -> pd.DataFrame:
    return pd.DataFrame([
        ("Estrategia", f"Cada mes: largo en las {N_LARGO} acciones con mayor E[R] y corto en las {N_CORTO} "
         "con menor E[R] del Top 100, pesos iguales (+1/N y -1/N). 100% largo, 100% corto"),
        ("Rendimiento esperado de la estrategia", "Promedio del E[R] de las largas menos promedio del E[R] de "
                                                  "las cortas (ex-ante), mes a mes"),
        ("Largo-Corto (bruto / neto)", f"Neto resta {COSTO_BPS:g} pb por cada 100% operado (compras y ventas, "
         "ambas piernas). No incluye costo de préstamo de acciones"),
        ("Pasiva: Top 100 igual ponderado", "Comprar y mantener el promedio simple de las acciones del Top 100 "
                                            "de cada mes (no usa el modelo)"),
        ("Pasiva: S&P 500",
         "Índice de precio (^GSPC), sin dividendos: MERCADO + rf de factores.csv"),
        ("Sharpe clásico", "(media del rendimiento en exceso / desviación) * sqrt(12). Al largo-corto no se "
                           "le resta rf porque se autofinancia; a las demás sí"),
        ("alpha_anual / beta_mercado",
         "Regresión del exceso de la estrategia contra el exceso del S&P 500"),
        ("treynor",
         "Rendimiento en exceso medio anualizado / beta_mercado"),
        ("dif_media_anualizada / t_stat_dif",
         "Diferencia mensual contra la pasiva (x12) y su t pareada"),
        ("Cuantiles",
         "Rendimiento real promedio por quintil del rendimiento esperado; Q1 = menor E[R]"),
        ("Sesgo", "El modelo se eligió con el mismo periodo que se usa aquí; el desempeño es en-muestra "
                  "respecto a esa elección"),
    ], columns=["concepto", "definición"])


# Corre la estrategia completa para UN modelo

def correr_modelo(num: int, version: str, descr: str, es_mejor_global: bool,
                   real: pd.DataFrame, factores: pd.DataFrame) -> dict:
    nombre_modelo = f"Modelo {num}, versión {version}"
    etiqueta_carpeta = f"modelo{num}_{version}" + \
        ("_mejor_global" if es_mejor_global else "")
    salida = SALIDA_RAIZ / etiqueta_carpeta
    salida.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 100}\n{nombre_modelo}  ({descr})"
          f"{'  <-- mejor_global' if es_mejor_global else ''}\n{'-' * 100}")

    er = cargar_rendimiento_esperado(num, version)
    print(f"  rendimiento esperado: {er.index.min():%Y-%m} a {er.index.max():%Y-%m}, "
          f"{er.shape[1]} acciones")

    # Portafolios mes a mes
    ret, pos = construir_portafolios(er, real)
    print(f"  {len(ret)} meses con estrategia ({ret.index.min():%Y-%m} a {ret.index.max():%Y-%m}), "
          f"universo medio = {ret['n_universo'].mean():.0f} acciones")

    rf = factores["rf"].reindex(
        ret.index) if "rf" in factores.columns else pd.Series(0.0, index=ret.index)
    if "rf" not in factores.columns:
        print("  AVISO: factores.csv no tiene rf; el Sharpe usa rf = 0.")
    mercado_exceso = factores["MERCADO"].reindex(
        ret.index) if "MERCADO" in factores.columns else None

    # Series de rendimiento y de exceso
    estrategias = pd.DataFrame({
        LS_BRUTO: ret["ls_bruto"],
        LS_NETO: ret["ls_neto"],
        SOLO_LARGO: ret["largo_neto"],
        SOLO_CORTO: ret["corto_neto"],
        PASIVA_EW: ret["ret_ew_top100"],
    })
    excesos = pd.DataFrame({
        LS_BRUTO: ret["ls_bruto"],
        LS_NETO: ret["ls_neto"],
        SOLO_LARGO: ret["largo_neto"] - rf,
        SOLO_CORTO: ret["corto_neto"],
        PASIVA_EW: ret["ret_ew_top100"] - rf,
    })
    if mercado_exceso is not None:
        estrategias[PASIVA_SP] = mercado_exceso + rf
        excesos[PASIVA_SP] = mercado_exceso
    else:
        print("  AVISO: factores.csv no tiene MERCADO; se omite la pasiva S&P 500.")
    estrategias = estrategias.dropna(how="all")
    excesos = excesos.reindex(estrategias.index)

    # Métricas
    metricas = metricas_de_todas(estrategias, excesos, mercado_exceso)
    comparacion = comparacion_vs_pasivas(estrategias, excesos)
    anual = por_anio(estrategias)
    cuantiles = tabla_cuantiles(er, real, N_CUANTILES)

    # Subperiodos (look-back en varios periodos): primera y segunda mitad
    corte = estrategias.index[len(estrategias) // 2]
    sub = {}
    for etiqueta, mascara in (
        (f"Primera mitad ({estrategias.index[0]:%Y-%m} a {estrategias.index[len(estrategias)//2 - 1]:%Y-%m})",
         estrategias.index < corte),
        (f"Segunda mitad ({corte:%Y-%m} a {estrategias.index[-1]:%Y-%m})",
         estrategias.index >= corte),
    ):
        m = metricas_de_todas(estrategias[mascara], excesos[mascara],
                              mercado_exceso.reindex(
                                  estrategias.index)[mascara]
                              if mercado_exceso is not None else None)
        m.insert(0, "subperiodo", etiqueta)
        sub[etiqueta] = m.reset_index()
    subperiodos = pd.concat(sub.values(), ignore_index=True)

    # Rendimiento esperado vs realizado de la estrategia
    esperado = pd.DataFrame({
        "er_largo_corto_medio_mensual": [ret["er_largo_corto"].mean()],
        "realizado_bruto_medio_mensual": [ret["ls_bruto"].mean()],
        "er_largo_corto_anualizado": [ret["er_largo_corto"].mean() * MESES_ANIO],
        "realizado_bruto_anualizado": [ret["ls_bruto"].mean() * MESES_ANIO],
        "correlacion_esperado_vs_realizado": [ret["er_largo_corto"].corr(ret["ls_bruto"])],
        "pct_meses_largo_corto_positivo": [(ret["ls_bruto"] > 0).mean()],
        "operado_promedio_mensual_largo_corto": [(ret["operado_largo"] + ret["operado_corto"]).mean()],
    }, index=[nombre_modelo])

    # Guardar
    salida_ret = ret.copy()
    salida_ret["rf"] = rf
    if PASIVA_SP in estrategias.columns:
        salida_ret["pasiva_sp500"] = estrategias[PASIVA_SP]
    salida_ret.rename_axis("fecha").to_csv(salida / "retornos_mensuales.csv")
    pos.to_csv(salida / "posiciones.csv", index=False)
    metricas.to_csv(salida / "resumen_metricas.csv")
    comparacion.to_csv(salida / "comparacion_vs_pasivas.csv", index=False)
    anual.to_csv(salida / "por_anio.csv")
    subperiodos.to_csv(salida / "subperiodos.csv", index=False)
    cuantiles.to_csv(salida / "cuantiles.csv")
    pd.DataFrame({n: sharpe_movil(excesos[n], VENTANA_SHARPE_MOVIL) for n in excesos.columns}) \
        .rename_axis("fecha").to_csv(salida / "sharpe_movil_36m.csv")

    with pd.ExcelWriter(salida / "estrategia.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"modelo": [nombre_modelo], "configuracion": [descr], "N_largo": [N_LARGO],
                      "N_corto": [N_CORTO], "costo_bps": [COSTO_BPS]}).to_excel(xw, sheet_name="Modelo_usado", index=False)
        metricas.to_excel(xw, sheet_name="Metricas")
        comparacion.to_excel(
            xw, sheet_name="Comparacion_vs_pasivas", index=False)
        esperado.to_excel(xw, sheet_name="Esperado_vs_real")
        cuantiles.to_excel(xw, sheet_name="Cuantiles")
        anual.to_excel(xw, sheet_name="Por_anio")
        subperiodos.to_excel(xw, sheet_name="Subperiodos", index=False)
        salida_ret.rename_axis("fecha").to_excel(
            xw, sheet_name="Retornos_mensuales")
        definiciones().to_excel(xw, sheet_name="Definiciones", index=False)

    graficar(estrategias, excesos, salida /
             "estrategia_cuantilica.png", nombre_modelo)

    ls = metricas.loc[LS_NETO]
    print(f"  Largo-Corto neto: Sharpe clásico = {ls['sharpe_clasico']:.2f}, "
          f"rendimiento anualizado = {ls['retorno_anualizado']:.2%}, "
          f"max drawdown = {ls['max_drawdown']:.2%}, "
          f"Treynor = {ls['treynor']:.4f}")
    print(f"  Resultados completos en {salida}")

    fila_resumen = {
        "modelo": num, "version": version, "parametros": descr,
        "mejor_global": es_mejor_global, "carpeta": str(salida),
    }
    fila_resumen.update({f"{LS_NETO}__{c}": metricas.loc[LS_NETO, c]
                          for c in metricas.columns})
    return fila_resumen


# Programa principal
def main():
    SALIDA_RAIZ.mkdir(parents=True, exist_ok=True)

    real = cargar_rendimientos(REGRESION / "rendimientos_miembros.csv")
    factores = cargar_factores(REGRESION / "factores.csv")

    modelos = leer_modelos_a_correr()
    print(f"Se van a correr {len(modelos)} modelo(s).")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    resumenes = []
    for num, version, descr, es_mejor_global in modelos:
        try:
            resumenes.append(correr_modelo(
                num, version, descr, es_mejor_global, real, factores))
        except FileNotFoundError as e:
            print(f"  AVISO: se omite Modelo {num} versión {version}: {e}")

    if not resumenes:
        raise RuntimeError("No se pudo correr ningún modelo.")

    comparacion_modelos = pd.DataFrame(resumenes).sort_values(
        f"{LS_NETO}__sharpe_clasico", ascending=False)
    comparacion_modelos.to_csv(
        SALIDA_RAIZ / "resumen_todos_los_modelos.csv", index=False)

    cols = ["modelo", "version", "mejor_global",
            f"{LS_NETO}__sharpe_clasico", f"{LS_NETO}__treynor",
            f"{LS_NETO}__retorno_anualizado", f"{LS_NETO}__max_drawdown"]
    cols = [c for c in cols if c in comparacion_modelos.columns]
    print(f"\n{'=' * 100}\nComparación de todos los modelos (Largo-Corto neto, ordenado por Sharpe)\n{'=' * 100}")
    print(comparacion_modelos[cols].round(4).to_string(index=False))
    print(f"\nResumen completo en {SALIDA_RAIZ / 'resumen_todos_los_modelos.csv'}")


if __name__ == "__main__":
    main()