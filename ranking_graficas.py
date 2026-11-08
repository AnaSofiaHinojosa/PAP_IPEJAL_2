from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("Data")
ESTRATEGIA = Path("salidas/estrategia")
SALIDA = Path("salidas/ranking_marketcap")

N_ANIOS = 5          # años más recientes de Data/<AAAA>.csv
N_EMPRESAS = 20      # cuántas largas (top) y cuántas cortas (bottom)


# Selección de las empresas

def ruta_posiciones() -> Path:
    """posiciones.csv del mejor modelo global (carpeta ..._mejor_global).
    Si no existe, usa el posiciones.csv suelto de salidas/estrategia."""
    candidatas = sorted(ESTRATEGIA.glob("modelo*_mejor_global/posiciones.csv"))
    if candidatas:
        return candidatas[0]
    suelto = ESTRATEGIA / "posiciones.csv"
    if suelto.exists():
        return suelto
    raise FileNotFoundError(
        f"No encontré posiciones.csv en {ESTRATEGIA}. Corre primero estrategia_cuantilica.py.")


def leer_top_y_bottom(ruta: Path) -> tuple[list[str], list[str], pd.Timestamp]:
    """Top = libro largo (mayor E[R]) y bottom = libro corto (menor E[R])
    del mes más reciente de posiciones.csv."""
    pos = pd.read_csv(ruta, parse_dates=["fecha"])
    ultima = pos["fecha"].max()
    mes = pos[pos["fecha"] == ultima]
    top = (mes[mes["lado"] == "LARGO"].sort_values("er", ascending=False)
           ["ticker"].head(N_EMPRESAS).tolist())
    bottom = (mes[mes["lado"] == "CORTO"].sort_values("er", ascending=True)
              ["ticker"].head(N_EMPRESAS).tolist())
    return top, bottom, ultima


# Ranking por año dentro del índice completo

def anios_disponibles(n_anios: int) -> list[int]:
    anios = []
    for f in DATA.glob("*.csv"):
        try:
            anios.append(int(f.stem))
        except ValueError:
            continue
    return sorted(anios)[-n_anios:]


def rankear_anio(anio: int) -> pd.Series:
    """Rank 1..N de cada ticker del índice ese año, por Weight (proxy de market cap)."""
    df = pd.read_csv(DATA / f"{anio}.csv")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Weight"]).sort_values(
        "Weight", ascending=False)
    df = df.drop_duplicates(subset="Ticker", keep="first")
    return pd.Series(range(1, len(df) + 1), index=df["Ticker"].to_numpy())


def tabla_ranking(tickers: list[str], anios: list[int]) -> pd.DataFrame:
    """ticker x año con el rank (NaN si no estaba en el índice ese año)."""
    por_anio = {a: rankear_anio(a) for a in anios}
    # posiciones.csv usa estilo Yahoo (BRK-B); Data/ usa BRK.B
    return pd.DataFrame({a: pd.Series({t: por_anio[a].get(t.replace("-", "."), np.nan)
                                       for t in tickers})
                         for a in anios}).reindex(tickers)


# Gráficas

def grafica_evolucion(top: pd.DataFrame, bottom: pd.DataFrame, ruta: Path, fecha):
    """Rank de cada empresa año por año (1 = la más grande del S&P 500)."""
    import matplotlib.pyplot as plt
    colores = plt.cm.tab20(np.linspace(0, 1, 20))
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    for ax, tabla, titulo in ((axes[0], top, f"Top {len(top)} (largas, mayor E[R])"),
                              (axes[1], bottom, f"Bottom {len(bottom)} (cortas, menor E[R])")):
        for color, (t, fila) in zip(colores, tabla.iterrows()):
            ax.plot(tabla.columns, fila.to_numpy(dtype=float), marker="o", ms=4,
                    lw=1.5, color=color, label=t)
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks(list(tabla.columns))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), frameon=False)
    peor = int(np.nanmax(np.concatenate([top.to_numpy(float).ravel(),
                                         bottom.to_numpy(float).ravel()])))
    axes[0].set_ylim(peor + 15, 1)   # rank 1 arriba
    axes[0].set_yticks([1] + list(range(100, peor + 15, 100)))
    axes[0].set_ylabel("Ranking por market cap en el S&P 500 (1 = mayor)")
    fig.suptitle(f"Ranking anual dentro del S&P 500 | empresas del mes {fecha:%Y-%m}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafica_promedio(top: pd.DataFrame, bottom: pd.DataFrame, ruta: Path, anios: list[int]):
    """Rank promedio en la ventana (barras horizontales, mejor rank arriba)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    for ax, tabla, titulo, color in ((axes[0], top, f"Top {len(top)} (largas)", "#2a7f62"),
                                     (axes[1], bottom, f"Bottom {len(bottom)} (cortas)", "#b5483a")):
        prom = tabla.mean(axis=1, skipna=True).sort_values()
        n = tabla.notna().sum(axis=1).reindex(prom.index)
        barras = ax.barh(prom.index, prom.to_numpy(dtype=float), color=color)
        ax.bar_label(barras, labels=[f"{v:.0f}  ({k}/{len(anios)} años)"
                                     for v, k in zip(prom, n)], fontsize=8, padding=3)
        ax.invert_yaxis()
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("Ranking promedio (1 = mayor market cap)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Ranking promedio en el S&P 500, {anios[0]}-{anios[-1]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafica_top_bottom_por_anio(ruta_pos: Path, n_anios: int, ruta_png: Path):
    """Para cada año, las N_EMPRESAS largas (top) y las N_EMPRESAS cortas (bottom)
    del último mes de ese año (en el año en curso, el último mes disponible).
    Cada celda es un ticker; el color indica en qué fracción de los meses del
    año estuvo en ese libro (más oscuro = más constante)."""
    import matplotlib.pyplot as plt
    pos = pd.read_csv(ruta_pos, parse_dates=["fecha"])
    pos["anio"] = pos["fecha"].dt.year
    anios = sorted(pos["anio"].unique())[-n_anios:]

    fig, axes = plt.subplots(1, 2, figsize=(2.3 * len(anios) * 2, 0.42 * N_EMPRESAS + 2.5))
    for ax, lado, cmap, titulo, ascendente in (
            (axes[0], "LARGO", "Greens", f"Top {N_EMPRESAS} (largas, mayor E[R])", False),
            (axes[1], "CORTO", "Reds", f"Bottom {N_EMPRESAS} (cortas, menor E[R])", True)):
        matriz = np.full((N_EMPRESAS, len(anios)), np.nan)
        textos = [[""] * len(anios) for _ in range(N_EMPRESAS)]
        etiquetas_x = []
        for j, anio in enumerate(anios):
            sub = pos[(pos["anio"] == anio) & (pos["lado"] == lado)]
            ultimo = sub["fecha"].max()
            meses_anio = sub["fecha"].nunique()
            meses_en_libro = sub.groupby("ticker")["fecha"].nunique()
            cierre = (sub[sub["fecha"] == ultimo]
                      .sort_values("er", ascending=ascendente).head(N_EMPRESAS))
            for i, t in enumerate(cierre["ticker"]):
                matriz[i, j] = meses_en_libro[t] / meses_anio
                textos[i][j] = t
            etiquetas_x.append(f"{anio}\n(cierre {ultimo:%b})")
        im = ax.imshow(matriz, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(N_EMPRESAS):
            for j in range(len(anios)):
                if textos[i][j]:
                    ax.text(j, i, textos[i][j], ha="center", va="center", fontsize=8,
                            color="white" if matriz[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(anios)), etiquetas_x, fontsize=8)
        ax.set_yticks(range(N_EMPRESAS), range(1, N_EMPRESAS + 1), fontsize=8)
        ax.set_ylabel("Posición dentro del libro (1 = más extrema)", fontsize=8)
        ax.set_title(titulo, fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Fracción de meses del año en el libro", fontsize=8)
    fig.suptitle("Top y bottom por año (portafolio al cierre de cada año)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        raise SystemExit(
            "Falta matplotlib, necesario para generar las imágenes.\n"
            "Instálalo con:  pip install matplotlib")
    if not DATA.exists():
        raise SystemExit(
            f"No encontré la carpeta {DATA.resolve()}. "
            "Corre el script parado en la raíz del repo (PAP_IPEJAL_2).")

    ruta = ruta_posiciones()
    top_t, bottom_t, fecha = leer_top_y_bottom(ruta)
    anios = anios_disponibles(N_ANIOS)
    print(f"Posiciones de: {ruta}  (mes {fecha:%Y-%m})")
    print(f"Top {len(top_t)}:    {', '.join(top_t)}")
    print(f"Bottom {len(bottom_t)}: {', '.join(bottom_t)}")
    print(f"Años evaluados: {anios}")

    top = tabla_ranking(top_t, anios)
    bottom = tabla_ranking(bottom_t, anios)

    SALIDA.mkdir(parents=True, exist_ok=True)
    png_evolucion = SALIDA / "ranking_evolucion_top_bottom.png"
    png_promedio = SALIDA / "ranking_promedio_top_bottom.png"
    grafica_evolucion(top, bottom, png_evolucion, fecha)
    grafica_promedio(top, bottom, png_promedio, anios)
    png_por_anio = SALIDA / "top_bottom_por_anio.png"
    grafica_top_bottom_por_anio(ruta, N_ANIOS, png_por_anio)
    for png in (png_evolucion, png_promedio, png_por_anio):
        if not png.exists():
            raise RuntimeError(f"No se generó {png}")
        print(f"Imagen generada: {png.resolve()}")


if __name__ == "__main__":
    main()