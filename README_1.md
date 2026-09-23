# PAP 4J05 – IPEJAL · Parte 1: Datos y factores

Pipeline en Python que:

1. Selecciona el universo: las **100 empresas de mayor market cap del S&P 500 que sobrevivieron la ventana** (estaban en el índice en el año inicial y en el final, sin importar si salieron y volvieron a entrar).
2. Descarga **precios mensuales** (Yahoo Finance) y **fundamentales** (EPS y acciones en circulación, SEC EDGAR).
3. Calcula por empresa: P/E, market cap y rotación.
4. Construye las **series de tiempo de los factores**: Mercado, Value, Tamaño, Momentum, Volatilidad y Liquidez.
5. Entrega los insumos para la **Parte 2** (matriz de exposiciones): `rendimientos.csv` y `factores.csv`.

---

## Cómo correrlo

```bash
pip install yfinance requests pandas numpy openpyxl --break-system-packages
```

1. En `sec_edgar_fundamentals.py`, cambia `USER_AGENT` por un nombre y correo reales (la SEC lo exige).
2. Pon en la carpeta `Data/` los holdings históricos (`2007.csv`, `2008.csv`, …) y el Excel de holdings del SPY de 2026 (`holdings-daily-us-en-spy.xlsx`).
3. Corre:

```bash
python main.py                 # todo: universo -> descarga -> Excel por empresa -> factores
python validar_salidas.py      # revisión de calidad de datos
python comparar_con_yahoo.py   # prueba independiente contra Yahoo
```

Si solo cambias parámetros de los factores, basta con `python construir_factores.py` (no vuelve a descargar nada).

> Los archivos deben llamarse exactamente así (sin espacios), porque `main.py` importa `Datos`, `top100`, `yahoo_factor_data`, `sec_edgar_fundamentals` y `construir_factores` por nombre.

---

## Archivos de código

| Archivo | Estado | Qué hace |
|---|---|---|
| `main.py` | Modificado | Orquesta todo el proceso (6 pasos). |
| `Datos.py` | Modificado | Lee los holdings por año. Ahora puede conservar **todas** las empresas del índice (`top_n=None`). |
| `top100.py` | Modificado | Nueva clase `SurvivorTop100Selector`: sobrevivientes + top 100 por market cap. |
| `yahoo_factor_data.py` | Modificado | Precios, volumen, splits, S&P 500, T-bill y EPS trimestral reciente de Yahoo. |
| `sec_edgar_fundamentals.py` | Modificado (mucho) | EPS y acciones históricas de la SEC, con limpieza de datos. |
| `construir_factores.py` | Nuevo | Construye los factores y los archivos para la regresión. |
| `validar_salidas.py` | Nuevo | Marca datos sospechosos en los Excel de salida. |
| `comparar_con_yahoo.py` | Nuevo | Compara el último market cap y EPS contra Yahoo. |
| `explorar_xbrl.py` | Nuevo (opcional) | Lista los datos XBRL de un 10-Q/10-K. Sirve para investigar empresas con tags propios. |

### `main.py`
Pasos:
1. Arma el universo completo del S&P 500 por año (`Datos.py`).
2. Selecciona sobrevivientes y el top 100 (`top100.py`).
3. Descarga Yahoo y EDGAR, y completa con Yahoo los trimestres de EPS más recientes que la SEC aún no publica.
4. Guarda un Excel por empresa en `salidas/empresas/`. Los Excel de empresas que ya no están en el universo se mueven a `salidas/empresas_anteriores/`, no se borran.
5. Guarda los reportes de errores y cobertura.
6. Construye los factores (`construir_factores.py`).

Parámetros: `HISTORICAL_DIR`, `EXCEL_2026_PATH`, `TOP_N = 100` y `START_YEAR` (`None` = el año más antiguo disponible).

### `Datos.py` (`HoldingsUniverseBuilder`)
- Lee los CSV históricos (un archivo por año) y el Excel del SPY de 2026.
- **Cambio:** con `top_n=None` conserva todas las empresas del índice, no solo el top 100. La regla de sobrevivientes lo necesita, porque se mide la pertenencia al S&P 500, no al top 100.

### `top100.py` (`SurvivorTop100Selector`)
Regla del profesor:
1. **Sobrevivientes:** empresas que están en el índice en el año inicial **y** en el final.
2. **Top 100:** de las sobrevivientes, las 100 con mayor peso en el SPY del año final. El peso es proporcional al market cap ajustado por flotación.

Puntos importantes:
- **La fuente histórica reasigna tickers.** Por ejemplo, en 2007 United Technologies aparece como "UAL", Kraft Foods como "KHC" y Wachovia como "WFC". Por eso el código compara también el **nombre** de la empresa en el año inicial contra el del año final, y cada empresa queda con un estado:
  - `OK`: el nombre coincide.
  - `MANUAL`: resuelta en `EQUIVALENCIAS_MANUALES`.
  - `REVISAR`: mismo ticker pero nombre distinto. **Queda fuera** hasta que se decida a mano.
  - `DUPLICADA`: otra empresa del año inicial con el mismo ticker, típicamente una que fue comprada.
  - `DESCARTADA`: fue comprada o desapareció.
  - `NO_ESTA`: su ticker no aparece en el año final.
- **`EQUIVALENCIAS_MANUALES`** usa esta regla: si la empresa solo cambió de nombre o es la sucesora legal, sobrevive; si fue comprada por otra, no sobrevive. Ejemplos:
  - Google → GOOGL
  - WellPoint → ELV
  - UTX → RTX
  - Kraft Foods → MDLZ
  - Thermo Electron → TMO
  - Ingersoll-Rand → TT
  - BK → BNY
  - Tyco → JCI (Tyco es la sucesora legal de la fusión con Johnson Controls)
- **Casos de criterio:**
  - **DELL:** excluida. Dell Inc. se hizo privada en 2013, y Yahoo no tiene precios de Dell antes de 2016; los de 2016 a 2018 son del tracking stock DVMT. Para volver a incluirla, se borra su línea.
  - **LIN:** incluida como sucesora de Praxair.
- Salidas: `universo_final_top100_sobrevivientes.csv` y `revision_sobrevivientes.csv` (todas las empresas del año inicial con su estado).

### `yahoo_factor_data.py` (`YahooFactorDataFetcher`)
- **Precio ajustado** (por splits y dividendos): para rendimientos, volatilidad y momentum.
- **Precio de cierre** (ajustado solo por splits): para P/E y market cap.
- **Volumen mensual**, para la rotación.
- **Splits** de cada ticker, que se pasan a EDGAR para poner todo en la base de acciones de hoy.
- **S&P 500 (^GSPC) y T-bill de 13 semanas (^IRX).**
- **Nuevo:** `descargar_eps_trimestral()` trae el EPS diluido de los últimos ~5 trimestres, solo para rellenar lo que la SEC aún no publica.

### `sec_edgar_fundamentals.py` (`SecEdgarFundamentalsFetcher`)
Descarga el API de *company facts* de la SEC y limpia los datos:
- **EPS trimestral:** solo periodos de 3 meses, y se usa la **primera publicación** de cada trimestre (sin datos corregidos después, para no usar información del futuro). El Q4 se calcula como anual menos los tres trimestres; si el anual viene en otro tag de EPS, también se usa (caso FCX).
- **EPS calculada** (utilidad neta entre acciones promedio): rellena trimestres donde la reportada falta o se publicó tarde, **solo si coincide** con la reportada en los trimestres comunes.
- **EPS imposibles:** se descartan valores 50 veces mayores que la mediana de sus trimestres vecinos. Los eventos reales más extremos de la muestra llegan a unas 35 veces.
- **Acciones:** se prueban varios tags y se usa el más actualizado; sus huecos se rellenan con los otros tags (casos SCHW y GOOGL). Se descartan registros con escala imposible (más de 10 veces o menos de 0.1 veces la mediana) y datos aislados de un solo trimestre (WFC 2023).
- **Splits:** EPS y acciones se convierten a la base de acciones de hoy, igual que los precios de Yahoo.
- **Cambios de CIK:** para XOM, DIS, MDT, CI, GOOGL, LIN y DELL se pega la historia del identificador anterior. En LIN se usa una fecha de corte.
- **BRK.B:** conversión de clase A a clase B (1 A = 1,500 B). Hoy BRK.B no está en el universo.
- **Nuevo:** `completar_eps_con_yahoo()` agrega los trimestres recientes que faltan, solo si Yahoo coincide con EDGAR (±10%) en los trimestres comunes. Esos trimestres quedan marcados como "Yahoo" en la hoja EPS.
- Todo lo que se descarta queda listado en `reporte_cobertura_edgar.csv`, columna `descartes_escala`.

### `construir_factores.py`
Cada fin de mes t−1 ordena las empresas por la característica y arma dos portafolios con el 30% de cada extremo. El factor del mes t es el rendimiento de uno menos el del otro. Como se forma con datos de t−1, no usa información del futuro.

| Factor | Definición |
|---|---|
| `rf` | T-bill 13 semanas / 100 / 12, del cierre del mes anterior |
| `MERCADO` | Rendimiento del S&P 500 − rf |
| `VALUE` | P/E bajo − P/E alto (solo EPS de 12 meses > 0) |
| `TAMANO` | Market cap chico − grande |
| `MOMENTUM` | Rendimiento de t−12 a t−2 alto − bajo |
| `VOLATILIDAD` | Volatilidad de 12 meses baja − alta |
| `LIQUIDEZ` | Rotación de 3 meses baja − alta |

Parámetros (arriba del archivo):
- `CORTE = 0.30`: tamaño de cada extremo (terciles).
- `PONDERACION = "igual"` o `"market_cap"`.
- `MIN_EMPRESAS = 15`: empresas con dato necesarias para calcular el factor ese mes.
- Ventanas: `MESES_MOMENTUM`, `MESES_VOL`, `MESES_ROTACION`.
- `ORTOGONALIZAR`: ver "Notas para la Parte 2".
- `UMBRAL_CORRELACION`: el script avisa si dos factores pasan esta correlación.

### `validar_salidas.py`
Revisa `salidas/empresas/*.xlsx` y genera `salidas/validacion.csv`:
- `ALTA`: casi seguro error de datos (escalas imposibles, EPS absurdos, factores de split inconsistentes).
- `MEDIA`: revisar a mano. Suelen ser eventos reales: pérdidas extraordinarias, fusiones, escisiones.
- `INFO`: contexto (P/E mayor a 200, huecos por EPS negativo).

### `comparar_con_yahoo.py`
Prueba independiente: compara el **último market cap** (tolerancia ±10%) y el **último EPS de 12 meses** (±15%) de nuestros datos contra lo que Yahoo calcula por su cuenta. Resultado más reciente: market cap 100/100 y EPS 99/100. La única diferencia es JCI, y es de definición: nuestro EPS incluye operaciones discontinuadas (la venta de su negocio residencial en 2025) y el de Yahoo no.

---

## Salidas

```
universo_sp500_completo.csv                 todas las empresas del índice por año
universo_final_top100_sobrevivientes.csv    las 100 seleccionadas
revision_sobrevivientes.csv                 estado de cada empresa del año inicial
salidas/
  empresas/<TICKER>.xlsx     hojas Mensual, EPS, Acciones, Info
  mercado.xlsx               S&P 500 y T-bill
  reporte_errores.csv
  reporte_cobertura_edgar.csv
  validacion.csv             (validar_salidas.py)
  comparacion_yahoo.csv      (comparar_con_yahoo.py)
  regresion/
    rendimientos.csv         fecha x ticker  -> "y" de cada regresión
    factores.csv             fecha x factor  -> "X" (las mismas para todas)
    caracteristicas.csv      formato largo (respaldo)
    insumos_regresion.xlsx   lo mismo + hoja Definiciones
```

---

## Notas para la Parte 2 (matriz de exposiciones)

- Para cada empresa: `rendimientos[ticker]` contra las columnas de `factores.csv`, uniendo por `fecha`. Se sugiere usar el rendimiento en exceso (`rendimiento − rf`), porque `MERCADO` ya viene en exceso.
- **Ventana:** Value empieza en diciembre de 2010, así que la regresión con todos los factores va de **2010-12 en adelante**.
- **Correlaciones:** Volatilidad y Liquidez tenían una correlación de ~0.90, así que se activó `ORTOGONALIZAR = {"LIQUIDEZ": "VOLATILIDAD"}`. **LIQUIDEZ es el efecto de la rotación que NO explica la volatilidad** (correlación con Volatilidad = 0.00; desviación 1.7% mensual). Las demás correlaciones quedan por debajo de 0.65 en valor absoluto (las más altas: Value–Momentum −0.62 y Mercado–Volatilidad −0.61).
- La restricción de que las betas sumen 1 y la función de exposiciones corresponden a la Parte 2.

## Limitaciones (para el reporte)

- **Fundamentales:** los datos XBRL de la SEC empiezan en 2009–2010. Antes de eso no hay P/E ni market cap.
- **Ventana del universo:** 2007–2026 (19 años), porque no hay archivo de 2006.
- **Sesgo de supervivencia:** el universo está condicionado a sobrevivir y a ser grande en 2026. Por eso las empresas que eran "chicas" en 2011 y hoy están en el top 100 (NVDA, AMD, MU, LLY) sesgan hacia arriba el factor Tamaño, y en menor medida Volatilidad y Liquidez. Ver Grinold y Kahn, cap. 20, y Brown, Goetzmann, Ibbotson y Ross (1992).
- **Decisiones de criterio:** las equivalencias manuales (DELL, LIN, JCI), el Q4 calculado como anual menos tres trimestres, y el relleno de los últimos trimestres de EPS con Yahoo.
- **Mercado:** se usa el S&P 500 como índice de precio, sin dividendos.
