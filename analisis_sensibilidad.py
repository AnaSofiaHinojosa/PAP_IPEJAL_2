from pathlib import Path
import numpy as np
import pandas as pd

from backtesting import calcular_ic_serie
from optimizar_modelos import cargar_rendimientos, cargar_factores, resumir_ic, serie_ic
from ajustes import pesos_por_ic
from backtesting_ajustado import (
    construir_insumos,
    rendimiento_esperado,
    ic_por_factor,
    encoger_betas_por_corte,
)
from calibrar_modelos import (
    MODELOS,
    FACTORES,
    FRECUENCIA_REESTIMACION,
    REZAGO_FACTOR,
    MIN_ACCIONES_IC,
    MIN_PERIODOS_IC,
    ENTRENAR_CON_HISTORIAL,
    CONFIG_DEFAULT,
    leer_config_base,
)
from estrategia_cuantilica import N_LARGO, N_CORTO, COSTO_BPS
from metricas_desempeno import resumen_metricas, curva_capital, treynor, MESES_ANIO

REGRESION = Path("salidas/regresion")
CALIBRACION = Path("salidas/calibracion")
SALIDA = Path("salidas/sensibilidad")

# Parámetros ajustables
# Ventanas de estimación de los betas (meses): 3, 5 y 10 años.
# Se puede agregar None para incluir la ventana expansiva (histórico completo).
VENTANAS_BETA = [36, 60, 120]
# Cada cuántos meses se rebalancea el portafolio (1 = mensual, 3 = trimestral).
# El rebalanceo trimestral se hace al inicio de cada trimestre calendario
# (enero, abril, julio y octubre); entre rebalanceos los pesos se dejan correr.
FRECUENCIAS_REBALANCEO = {"Mensual": 1, "Trimestral": 3}
# Si se deja en None, se usa el mejor_global de finalistas_12_mismas_fechas.csv.
# Si se pone (num, version), se usa ese modelo, p. ej. (1, "A").
MODELO_MANUAL = None

# Estrategias y pasivas (mismos nombres que estrategia_cuantilica.py).
# Solo se analiza la estrategia largo-corto.
LS_BRUTO = "Largo-Corto (bruto)"
LS_NETO = "Largo-Corto (neto de costos)"
PASIVA_EW = "Pasiva: Top 100 igual ponderado"
PASIVA_SP = "Pasiva: S&P 500"
ESTRATEGIA_GRAFICA = LS_NETO     # la que se grafica


def etiqueta_ventana(v: int | None) -> str:
    return "completo" if v is None else f"{v}m ({v // 12} años)"


# Selección del modelo
def leer_modelo() -> tuple[int, str, str, str]:
    """(número de modelo, versión, candidato, parámetros) del modelo a analizar."""
    if MODELO_MANUAL is not None:
        num, version = int(MODELO_MANUAL[0]), str(MODELO_MANUAL[1])
    else:
        archivo = CALIBRACION / "finalistas_12_mismas_fechas.csv"
        if not archivo.exists():
            raise FileNotFoundError(
                f"No encontré {archivo}. Corre primero calibrar_modelos.py.")
        df = pd.read_csv(archivo)
        fila = df[df["mejor_global"].astype(
            str).str.lower().eq("true")].iloc[0]
        num, version = int(fila["modelo"]), str(fila["version"])

    finalistas = pd.read_csv(CALIBRACION / f"modelo{num}" / "finalistas.csv")
    fila = finalistas[finalistas["version"] == version]
    if fila.empty:
        raise ValueError(
            f"El modelo {num} no tiene versión {version} en su finalistas.csv")
    fila = fila.iloc[0]
    return num, version, str(fila["candidato"]), str(fila["parametros"])


def rendimiento_esperado_version(ins, version: str, candidato: str,
                                 rendimiento_real: pd.DataFrame,
                                 factores_df: pd.DataFrame) -> pd.DataFrame:
    """Rendimiento esperado de la versión A, B o C, con los mismos ajustes que
    usó calibrar_modelos.py para ese candidato."""
    if version == "A":
        return rendimiento_esperado(ins)
    if version == "B":
        # candidato = "B_<modo>_<ventana>", p. ej. "B_ic_ratio_60m" o "B_ic_var_exp"
        modo, etq = candidato[2:].rsplit("_", 1)
        v = None if etq == "exp" else int(etq.replace("m", ""))
        ic_fac = ic_por_factor(ins, rendimiento_real, MIN_ACCIONES_IC)
        pesos = (pesos_por_ic(ic_fac, ventana=v, min_periodos=MIN_PERIODOS_IC, modo=modo)
                 .reindex(factores_df.index).fillna(1.0))
        return rendimiento_esperado(ins, pesos=pesos)
    if version == "C":
        # candidato = "C_kappa<k>", p. ej. "C_kappa1"
        kappa = float(candidato.split("kappa")[1])
        betas_enc, _ = encoger_betas_por_corte(ins, kappa)
        return rendimiento_esperado(ins, betas_por_corte=betas_enc)
    raise ValueError(f"Versión desconocida: {version}")


# Portafolios con frecuencia de rebalanceo
def construir_portafolios_rebalanceo(
    er: pd.DataFrame,
    real_miembros: pd.DataFrame,
    real_total: pd.DataFrame,
    fechas: pd.DatetimeIndex,
    frecuencia: int,
) -> pd.DataFrame:
    """Mismo portafolio que estrategia_cuantilica.construir_portafolios (largo en
    las N_LARGO acciones con mayor E[R], corto en las N_CORTO con menor E[R],
    pesos iguales), pero solo se rebalancea en los meses de rebalanceo.

    - Mes de rebalanceo: se eligen las acciones con el E[R] de ese mes, entre
      los miembros del Top 100 (igual que en estrategia_cuantilica.py), y se
      vuelve a pesos iguales.
    - Meses intermedios: se mantienen las mismas acciones y los pesos se dejan
      correr con su rendimiento (comprar y mantener). El rendimiento se toma de
      rendimientos.csv para no perder una acción si sale del Top 100 a la
      mitad del trimestre.
    - Lo operado (turnover) se mide contra los pesos que ya se movieron con el
      mercado, así que en los meses sin rebalanceo es 0."""
    costo = COSTO_BPS / 10_000.0
    w_largo = pd.Series(dtype=float)
    w_corto = pd.Series(dtype=float)
    filas = []

    for fecha in fechas:
        es_mes_rebalanceo = (fecha.month - 1) % frecuencia == 0 or w_largo.empty
        oper_largo = oper_corto = 0.0
        rebalanceo = False
        if es_mes_rebalanceo and fecha in er.index and fecha in real_miembros.index:
            u = pd.concat([er.loc[fecha].rename("er"),
                           real_miembros.loc[fecha].rename("real")], axis=1).dropna()
            if len(u) >= N_LARGO + N_CORTO:
                u = u.sort_values("er", ascending=False, kind="stable")
                nuevo_largo = pd.Series(1.0 / N_LARGO, index=u.head(N_LARGO).index)
                nuevo_corto = pd.Series(1.0 / N_CORTO, index=u.tail(N_CORTO).index)
                oper_largo = nuevo_largo.sub(w_largo, fill_value=0.0).abs().sum()
                oper_corto = nuevo_corto.sub(w_corto, fill_value=0.0).abs().sum()
                w_largo, w_corto = nuevo_largo, nuevo_corto
                rebalanceo = True
        if w_largo.empty:
            continue  # todavía no se ha podido armar el primer portafolio

        r_largo = real_total.loc[fecha].reindex(w_largo.index)
        r_corto = real_total.loc[fecha].reindex(w_corto.index)
        # Si a una acción le falta el rendimiento, su peso se reparte entre las demás
        w_l = w_largo[r_largo.notna()] / w_largo[r_largo.notna()].sum()
        w_c = w_corto[r_corto.notna()] / w_corto[r_corto.notna()].sum()
        ret_largo = float((w_l * r_largo.dropna()).sum())
        ret_corto = float((w_c * r_corto.dropna()).sum())

        filas.append({
            "fecha": fecha,
            "rebalanceo": rebalanceo,
            "ret_largo": ret_largo,
            "ret_corto": ret_corto,
            "ls_bruto": ret_largo - ret_corto,
            "operado_largo": oper_largo,
            "operado_corto": oper_corto,
            "costo_largo": oper_largo * costo,
            "costo_corto": oper_corto * costo,
        })

        # Los pesos se mueven con el rendimiento de cada acción
        w_largo = w_l * (1.0 + r_largo.dropna())
        w_largo = w_largo / w_largo.sum()
        w_corto = w_c * (1.0 + r_corto.dropna())
        w_corto = w_corto / w_corto.sum()

    if not filas:
        raise RuntimeError(
            "No hubo ningún mes con suficientes acciones para armar el portafolio.")
    ret = pd.DataFrame(filas).set_index("fecha").sort_index()
    ret["ls_neto"] = ret["ls_bruto"] - ret["costo_largo"] - ret["costo_corto"]
    return ret


# Métricas
def information_ratio(r: pd.Series, benchmark: pd.Series) -> dict:
    """IR = rendimiento activo medio anualizado / tracking error anualizado."""
    activo = (r - benchmark).dropna()
    if len(activo) < 2:
        return {"rend_activo_anual": np.nan, "tracking_error": np.nan, "ir": np.nan}
    te = activo.std(ddof=1) * np.sqrt(MESES_ANIO)
    media = activo.mean() * MESES_ANIO
    return {"rend_activo_anual": float(media), "tracking_error": float(te),
            "ir": float(media / te) if te > 0 else np.nan}


def metricas_escenario(ret: pd.DataFrame, pasivas: pd.DataFrame,
                       mercado_exceso: pd.Series) -> list[dict]:
    """Métricas del largo-corto (bruto y neto) de un escenario (ventana x rebalanceo).
    El largo-corto se autofinancia, así que su rendimiento en exceso es el mismo
    rendimiento (no se le resta rf), igual que en estrategia_cuantilica.py."""
    series = {LS_BRUTO: ret["ls_bruto"], LS_NETO: ret["ls_neto"]}
    operado = ret["operado_largo"] + ret["operado_corto"]
    cero = pd.Series(0.0, index=ret.index)
    filas = []
    for nombre, r in series.items():
        m = resumen_metricas(r, r, mercado_exceso)
        ir_0 = information_ratio(r, cero)
        ir_ew = information_ratio(r, pasivas[PASIVA_EW])
        ir_sp = information_ratio(r, pasivas[PASIVA_SP]) if PASIVA_SP in pasivas else {}
        filas.append({
            "estrategia": nombre,
            "retorno_anualizado": m["retorno_anualizado"],
            "vol_anual": m["vol_anual"],
            "sharpe_clasico": m["sharpe_clasico"],
            "ir_vs_cero": ir_0["ir"],
            "ir_vs_top100_ew": ir_ew["ir"],
            "tracking_error_vs_top100_ew": ir_ew["tracking_error"],
            "rend_activo_vs_top100_ew": ir_ew["rend_activo_anual"],
            "ir_vs_sp500": ir_sp.get("ir", np.nan),
            "tracking_error_vs_sp500": ir_sp.get("tracking_error", np.nan),
            "treynor": treynor(r.dropna(), m.get("beta_mercado", np.nan)),
            "beta_mercado": m.get("beta_mercado", np.nan),
            "r2_mercado": m.get("r2_mercado", np.nan),
            "alpha_anual": m.get("alpha_anual", np.nan),
            "t_alpha": m.get("t_alpha", np.nan),
            "max_drawdown": m["max_drawdown"],
            "sortino": m["sortino"],
            "operado_promedio_mensual": float(operado.mean()),
            "operado_anual": float(operado.mean() * MESES_ANIO),
            "n_rebalanceos": int(ret["rebalanceo"].sum()),
            "n_meses": m["n_meses"],
        })
    return filas


def metricas_pasivas(pasivas: pd.DataFrame, rf: pd.Series,
                     mercado_exceso: pd.Series) -> pd.DataFrame:
    filas = {}
    for nombre in pasivas.columns:
        m = resumen_metricas(pasivas[nombre], pasivas[nombre] - rf, mercado_exceso)
        filas[nombre] = {k: m[k] for k in ("retorno_anualizado", "vol_anual",
                                          "sharpe_clasico", "max_drawdown", "n_meses")}
    return pd.DataFrame(filas).T.rename_axis("pasiva")


# Gráfico
def graficar(resumen: pd.DataFrame, curvas: dict, ruta: Path, titulo: str):
    """Barras ventana x rebalanceo de cada métrica del largo-corto neto, más la
    curva de capital de los 6 escenarios."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado; se omite el gráfico)")
        return
    # (columna, título, formato porcentaje)
    metricas = [("retorno_anualizado", "Retorno anualizado", True),
                ("vol_anual", "Volatilidad anual", True),
                ("sharpe_clasico", "Sharpe clásico", False),
                ("ir_vs_cero", "IR vs cero (benchmark = no invertir)", False),
                ("ir_vs_top100_ew", "IR vs Top 100 igual ponderado", False),
                ("treynor", "Treynor (exceso anual / beta)", True),
                ("beta_mercado", "Beta de mercado (vs S&P 500)", False)]
    ventanas = list(dict.fromkeys(resumen["ventana_beta"]))
    frecuencias = list(FRECUENCIAS_REBALANCEO)
    sub = resumen[resumen["estrategia"] == ESTRATEGIA_GRAFICA]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    ancho = 0.8 / len(frecuencias)
    x = np.arange(len(ventanas))
    for ax, (metrica, nombre, pct) in zip(axes.ravel(), metricas):
        for j, frec in enumerate(frecuencias):
            vals = [sub[(sub["ventana_beta"] == v) & (sub["rebalanceo"] == frec)][metrica].iloc[0]
                    for v in ventanas]
            barras = ax.bar(x + (j - (len(frecuencias) - 1) / 2) * ancho, vals,
                            ancho, label=frec)
            ax.bar_label(barras, fontsize=7,
                         labels=[f"{v:.1%}" if pct else f"{v:.2f}" for v in vals])
        ax.set_xticks(x, [v.split(" ")[0] for v in ventanas])
        ax.set_xlabel("Ventana de estimación de betas", fontsize=8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(nombre, fontsize=10)
        if pct:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8, title="Rebalanceo", title_fontsize=8)
    if "r2_mercado" in sub.columns:
        ax_beta = axes.ravel()[len(metricas) - 1]
        ax_beta.set_title(f"Beta de mercado (vs S&P 500)\nR² con el mercado: "
                          f"{sub['r2_mercado'].min():.2f} a {sub['r2_mercado'].max():.2f}",
                          fontsize=10)

    # Último panel: valor de $1 invertido en cada escenario
    ax = axes.ravel()[-1]
    for (v, frec), serie in curvas.items():
        ax.plot(curva_capital(serie), lw=1, ls="-" if frec == "Mensual" else "--",
                label=f"{v.split(' ')[0]}, {frec.lower()}")
    ax.set_title("Valor de $1 invertido", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle(f"Análisis de sensibilidad del {ESTRATEGIA_GRAFICA}  |  {titulo}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def definiciones(cfg: dict) -> pd.DataFrame:
    return pd.DataFrame([
        ("Escenario", "Combinación de ventana de estimación de betas x frecuencia de rebalanceo. "
                      "Todo lo demás (versión del modelo, min_obs, ventana de factor, N, costos) se "
                      f"deja fijo: min_obs={cfg['min_obs']}, ventana de factor={cfg['ventana_factor']}m, "
                      f"N_largo={N_LARGO}, N_corto={N_CORTO}, costo={COSTO_BPS:g} pb"),
        ("Ventana de estimación", "Meses de historia con los que se reestiman los betas en cada corte "
                                  "(walk-forward, cada 12 meses): 36 = 3 años, 60 = 5 años, 120 = 10 años"),
        ("Rebalanceo mensual", "Cada mes se eligen de nuevo las acciones con el E[R] de ese mes y se "
                               "vuelve a pesos iguales (misma regla que estrategia_cuantilica.py)"),
        ("Rebalanceo trimestral", "Solo en enero, abril, julio y octubre se eligen acciones y se vuelve a "
                                  "pesos iguales; los otros dos meses se mantienen las mismas acciones y los "
                                  "pesos se mueven con el mercado"),
        ("Mismas fechas", "Los 6 escenarios se evalúan exactamente en los mismos meses, para que la "
                          "comparación sea justa"),
        ("retorno_anualizado", "Rendimiento geométrico anual (compuesto)"),
        ("vol_anual", "Desviación estándar mensual x sqrt(12)"),
        ("Estrategia", f"Largo-corto: cada vez que se rebalancea, largo en las {N_LARGO} acciones con mayor "
                       f"E[R] y corto en las {N_CORTO} con menor E[R] del Top 100, pesos iguales. Bruto = "
                       f"sin costos; neto = menos {COSTO_BPS:g} pb por cada 100% operado"),
        ("sharpe_clasico", "(media del rendimiento en exceso / desviación) x sqrt(12). Al largo-corto no "
                           "se le resta rf porque se autofinancia"),
        ("ir_vs_cero", "Information Ratio con benchmark = no invertir (el benchmark natural de un "
                       "largo-corto): rendimiento medio anualizado / volatilidad anualizada. Como al "
                       "largo-corto no se le resta rf, coincide con el sharpe_clasico"),
        ("ir_vs_top100_ew", "Information Ratio = rendimiento activo medio anualizado / tracking error "
                            "anualizado, contra el Top 100 igual ponderado (rebalanceado cada mes). Sale "
                            "negativo porque el largo-corto tiene poca exposición al mercado y el Top 100 "
                            "es 100% largo: la diferencia es casi todo el rendimiento del mercado"),
        ("ir_vs_sp500", "Mismo IR contra el S&P 500 (^GSPC). Ojo: el índice no incluye dividendos"),
        ("tracking_error", "Desviación estándar anualizada del rendimiento activo (estrategia - benchmark)"),
        ("beta_mercado / r2_mercado", "Regresión del rendimiento del largo-corto contra el exceso del "
                                      "S&P 500 (MERCADO de factores.csv): pendiente y R². Una beta cercana a "
                                      "0 significa que el portafolio casi no depende del mercado"),
        ("treynor", "Rendimiento en exceso medio anualizado / beta_mercado (misma función que "
                    "estrategia_cuantilica.py). Mide el rendimiento por unidad de riesgo de mercado. Como la "
                    "beta del largo-corto es chica, el Treynor sale grande y cambia mucho con pequeños "
                    "cambios en la beta: hay que leerlo junto con la gráfica de la beta"),
        ("alpha_anual / t_alpha", "Constante de esa misma regresión, anualizada, y su t-stat"),
        ("operado_anual", "Suma de |cambio de peso| por año (compras y ventas de ambas piernas). Con "
                          "rebalanceo trimestral solo se opera 4 veces al año"),
        ("Pasiva: Top 100 igual ponderado", "Promedio simple del rendimiento de todos los miembros del "
                                            "Top 100 en cada mes (no usa el modelo); solo se usa como "
                                            "benchmark del IR"),
        ("Sesgo", "El modelo se eligió con el mismo periodo que se usa aquí; el desempeño es en-muestra "
                  "respecto a esa elección"),
    ], columns=["concepto", "definición"])


# Programa principal
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("Leyendo rendimientos y factores...")
    real_miembros = cargar_rendimientos(REGRESION / "rendimientos_miembros.csv")
    real_total = cargar_rendimientos(REGRESION / "rendimientos.csv")
    factores_df = cargar_factores(REGRESION / "factores.csv")
    rendimientos = real_total if ENTRENAR_CON_HISTORIAL else real_miembros

    # Modelo a analizar y su configuración base
    num, version, candidato, parametros = leer_modelo()
    _, nombre_modelo, carpeta_bt, flags = next(m for m in MODELOS if m[0] == num)
    cfg = leer_config_base(carpeta_bt)
    if cfg is None:
        cfg = {**CONFIG_DEFAULT, "modelo_id": "(configuración por defecto)"}
    titulo = f"Modelo {num}, versión {version} ({candidato})"
    print(f"\n{'=' * 90}\n{titulo}\n  {nombre_modelo}; {parametros}")
    print(f"  configuración base: beta={etiqueta_ventana(cfg['ventana_beta'])}, "
          f"min_obs={cfg['min_obs']}, factor={cfg['ventana_factor']}m")
    print(f"  ventanas a probar: {[etiqueta_ventana(v) for v in VENTANAS_BETA]}")
    print(f"  rebalanceos a probar: {list(FRECUENCIAS_REBALANCEO)}\n{'=' * 90}")

    # 1) Rendimiento esperado e IC con cada ventana de estimación
    er_por_ventana, filas_ic = {}, []
    for ventana in VENTANAS_BETA:
        etq = etiqueta_ventana(ventana)
        print(f"\nVentana de betas = {etq}...")
        ins = construir_insumos(
            rendimientos, factores_df, FACTORES,
            ventana_beta=ventana, min_obs=cfg["min_obs"],
            frecuencia_meses=FRECUENCIA_REESTIMACION, ventana_factor=cfg["ventana_factor"],
            rezago_factor=REZAGO_FACTOR, **flags,
        )
        er = rendimiento_esperado_version(ins, version, candidato, real_miembros, factores_df)
        er_por_ventana[etq] = er
        ic = serie_ic(calcular_ic_serie(er, real_miembros, min_acciones=MIN_ACCIONES_IC))
        filas_ic.append({"ventana_beta": etq, "ic_serie": ic})
        print(f"  {len(ins.cortes)} cortes, E[R] de {er.index.min():%Y-%m} a {er.index.max():%Y-%m}")

        # Verificación: con la ventana base debe salir el mismo E[R] que calibrar_modelos.py
        if ventana == cfg["ventana_beta"]:
            ruta = CALIBRACION / f"modelo{num}" / f"rendimiento_esperado_{version}.csv"
            if ruta.exists():
                guardado = pd.read_csv(ruta, index_col=0, parse_dates=True)
                dif = (er - guardado).abs().max().max()
                print(f"  verificación contra {ruta.name}: diferencia máxima = {dif:.2e}")

    # 2) Fechas comunes: meses en los que todas las ventanas tienen E[R], desde el
    #    primer inicio de trimestre (para que mensual y trimestral arranquen igual)
    fechas = real_miembros.index
    for er in er_por_ventana.values():
        fechas = fechas.intersection(er.dropna(how="all").index)
    fechas = fechas.sort_values()
    inicio = next(f for f in fechas if (f.month - 1) % max(FRECUENCIAS_REBALANCEO.values()) == 0)
    fechas = fechas[fechas >= inicio]
    print(f"\nTodos los escenarios se evalúan en las MISMAS {len(fechas)} fechas: "
          f"{fechas[0]:%Y-%m} a {fechas[-1]:%Y-%m}")

    rf = factores_df["rf"].reindex(fechas)
    mercado_exceso = factores_df["MERCADO"].reindex(fechas)
    pasivas = pd.DataFrame({
        PASIVA_EW: real_miembros.loc[fechas].mean(axis=1),
        PASIVA_SP: mercado_exceso + rf,
    })

    # IC de cada ventana en las fechas comunes
    tabla_ic = pd.DataFrame([
        {"ventana_beta": d["ventana_beta"], **resumir_ic(d["ic_serie"].reindex(fechas).dropna())}
        for d in filas_ic
    ])

    # 3) Portafolios y métricas de cada escenario
    filas, curvas, retornos = [], {}, {}
    for etq, er in er_por_ventana.items():
        for nombre_frec, frec in FRECUENCIAS_REBALANCEO.items():
            ret = construir_portafolios_rebalanceo(er, real_miembros, real_total, fechas, frec)
            ret = ret.reindex(fechas)
            for fila in metricas_escenario(ret, pasivas, mercado_exceso):
                filas.append({"ventana_beta": etq, "rebalanceo": nombre_frec, **fila})
            curvas[(etq, nombre_frec)] = ret["ls_neto"] if ESTRATEGIA_GRAFICA == LS_NETO \
                else ret["ls_bruto"]
            for col, nombre in (("ls_bruto", LS_BRUTO), ("ls_neto", LS_NETO)):
                retornos[f"{nombre} | {etq} | {nombre_frec}"] = ret[col]
    resumen = pd.DataFrame(filas)

    # Tablas ventana x rebalanceo para cada métrica principal
    tablas = {}
    for metrica, hoja in (("retorno_anualizado", "Retorno"), ("vol_anual", "Volatilidad"),
                          ("sharpe_clasico", "Sharpe"), ("ir_vs_cero", "IR_vs_cero"),
                          ("ir_vs_top100_ew", "IR_vs_Top100"), ("ir_vs_sp500", "IR_vs_SP500"),
                          ("treynor", "Treynor"), ("beta_mercado", "Beta_mercado"),
                          ("operado_anual", "Operado_anual")):
        tablas[hoja] = resumen.pivot_table(index="ventana_beta", columns=["estrategia", "rebalanceo"],
                                           values=metrica, sort=False)
    pasivas_tabla = metricas_pasivas(pasivas, rf, mercado_exceso)
    retornos_df = pd.DataFrame(retornos)
    retornos_df[PASIVA_EW] = pasivas[PASIVA_EW]
    retornos_df[PASIVA_SP] = pasivas[PASIVA_SP]
    retornos_df["rf"] = rf

    # 4) Guardar
    resumen.to_csv(SALIDA / "resumen_sensibilidad.csv", index=False)
    tabla_ic.to_csv(SALIDA / "ic_por_ventana.csv", index=False)
    pasivas_tabla.to_csv(SALIDA / "pasivas.csv")
    retornos_df.rename_axis("fecha").to_csv(SALIDA / "retornos_mensuales.csv")

    with pd.ExcelWriter(SALIDA / "sensibilidad.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({
            "modelo": [titulo], "configuracion": [parametros],
            "min_obs": [cfg["min_obs"]], "ventana_factor_meses": [cfg["ventana_factor"]],
            "ventana_beta_base": [etiqueta_ventana(cfg["ventana_beta"])],
            "N_largo": [N_LARGO], "N_corto": [N_CORTO], "costo_bps": [COSTO_BPS],
            "fechas": [f"{fechas[0]:%Y-%m} a {fechas[-1]:%Y-%m} ({len(fechas)} meses)"],
        }).to_excel(xw, sheet_name="Modelo_usado", index=False)
        resumen.to_excel(xw, sheet_name="Resumen", index=False)
        for hoja, tabla in tablas.items():
            tabla.to_excel(xw, sheet_name=hoja)
        tabla_ic.to_excel(xw, sheet_name="IC_por_ventana", index=False)
        pasivas_tabla.to_excel(xw, sheet_name="Pasivas")
        retornos_df.rename_axis("fecha").to_excel(xw, sheet_name="Retornos_mensuales")
        definiciones(cfg).to_excel(xw, sheet_name="Definiciones", index=False)

    graficar(resumen, curvas, SALIDA / "sensibilidad.png", titulo)

    # 5) Consola
    cols = ["ventana_beta", "rebalanceo", "retorno_anualizado", "vol_anual", "sharpe_clasico",
            "ir_vs_cero", "ir_vs_top100_ew", "treynor", "beta_mercado", "operado_anual"]
    for estrategia in (LS_BRUTO, LS_NETO):
        print(f"\n{'=' * 90}\n{estrategia}\n{'=' * 90}")
        print(resumen.loc[resumen["estrategia"] == estrategia, cols].round(4).to_string(index=False))
    print(f"\n{'=' * 90}\nIC del modelo con cada ventana (mismas fechas)\n{'=' * 90}")
    print(tabla_ic[["ventana_beta", "ic_medio", "ic_std", "ic_ratio", "t_stat", "hit_rate",
                    "n_periodos"]].round(4).to_string(index=False))
    print(f"\n{'=' * 90}\nPasivas (mismas fechas)\n{'=' * 90}")
    print(pasivas_tabla.round(4).to_string())
    print(f"\nResultados completos en {SALIDA}")


if __name__ == "__main__":
    main()
