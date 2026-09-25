# PAP_IPEJAL_2 — Modelo de factores para el Top 100 del S&P 500

Pipeline que arma un universo histórico del S&P 500, descarga precios y
fundamentales, construye factores tipo Fama-French, estima la exposición
(beta) de cada acción a esos factores, y con eso calcula un rendimiento
esperado mensual por acción.

El proyecto está dividido en 3 partes que se corren **en orden**, porque cada
una consume las salidas de la anterior:

| Parte | Qué hace | Script principal |
|---|---|---|
| 1. Datos | Universo S&P 500, precios, fundamentales y factores | `main.py` |
| 2. Matriz de exposiciones | Beta de cada acción a cada factor (con restricción ∑β=1) | `generar_matriz.py` |
| 3. Rendimiento esperado | E[R] mensual de cada acción a partir de los betas | `generar_rendimientos_esperados.py` |

---

## 1. Requirements

Crea un entorno virtual e instala dependencias:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` debe tener (el que trae el repo tiene errores — `os` y
`glob` son de la librería estándar de Python, no se instalan con pip; y
faltan `numpy`, `yfinance` y `requests`, que sí se usan):

```
pandas
numpy
openpyxl
yfinance
requests
```

Python 3.10+ (el código usa type hints tipo `list[str] | None`).

### Configuración antes de correr

Abre **`sec_edgar_fundamentals.py`** y cambia esta línea por tu nombre y
correo real (SEC EDGAR exige un `User-Agent` identificable; si lo dejas con
un correo falso el script revienta con `RuntimeError` a propósito):

```python
USER_AGENT = "Tu Nombre tu_correo@ejemplo.com"
```

---

## 2. Estructura de carpetas

```
PAP_IPEJAL_2/
├── Data/                          insumos crudos (NO se modifica, solo se lee)
│   ├── 2007.csv ... 2025.csv      holdings anuales del S&P 500
│   └── holdings-daily-us-en-spy.xlsx
│
├── Datos.py                       Parte 1 - arma el universo S&P 500 histórico
├── top100.py                      Parte 1 - selecciona el Top 100 por año
├── yahoo_factor_data.py           Parte 1 - descarga precios/volumen (Yahoo)
├── sec_edgar_fundamentals.py      Parte 1 - descarga EPS y acciones en circulación (SEC EDGAR)
├── construir_factores.py          Parte 1 - construye los factores mensuales
├── main.py                        Parte 1 - orquesta todo lo anterior
│
├── estimar_exposiciones.py            Parte 2 - función de regresión con restricción
├── generar_matriz.py                  Parte 2 - orquesta la Parte 2
│
├── estimar_rendimiento_esperado.py    Parte 3 - factor esperado + rendimiento esperado
├── generar_rendimientos_esperados.py  Parte 3 - orquesta la Parte 3
│
├── universo_final_top100_global.csv   (insumo/referencia de universo)
├── universo_sp500.csv                 (insumo/referencia de universo)
├── requirements.txt
└── salidas/                       TODAS las salidas del pipeline (se genera sola)
    ├── empresas/<TICKER>.xlsx         precios, EPS, acciones, info (Parte 1)
    ├── empresas_anteriores/           tickers que salieron del universo
    ├── mercado.xlsx
    ├── reporte_errores.csv
    ├── reporte_cobertura_edgar.csv
    ├── resumen_universo.xlsx
    ├── revision_precios_vs_top100.csv
    ├── regresion/                     salidas de la Parte 1 que usa la Parte 2/3
    │   ├── rendimientos.csv               fecha x ticker, TODAS las empresas
    │   ├── rendimientos_miembros.csv       fecha x ticker, solo meses en el top 100
    │   ├── membresia.csv                   fecha x ticker, 1/0
    │   ├── factores.csv                    fecha x factor (rf, MERCADO, VALUE, TAMANO, MOMENTUM, VOLATILIDAD, LIQUIDEZ)
    │   ├── caracteristicas.csv
    │   └── insumos_regresion.xlsx
    ├── exposiciones/                  salidas de la Parte 2
    │   ├── matriz_exposiciones.csv        una fila por empresa: alpha, beta por factor, r2, n_obs, suma_beta
    │   └── matriz_exposiciones.xlsx
    └── rendimientos_esperados/        salidas de la Parte 3
        ├── rendimientos_esperados_ventana3.csv
        ├── rendimientos_esperados_ventana6.csv
        ├── rendimientos_esperados_ventana12.csv
        └── rendimientos_esperados.xlsx
```

`Data/` nunca se escribe — es solo de entrada. Todo lo que el pipeline
produce vive en `salidas/`, que se crea sola la primera vez que corres algo.

Importante: todos los scripts usan rutas **relativas** (`Path("salidas/...")`),
así que **siempre corre los scripts parado en la raíz del repo**
(`PAP_IPEJAL_2/`), nunca desde otra carpeta.

---

## 3. Orden de ejecución, desde cero

```bash
cd PAP_IPEJAL_2

# Parte 1 — universo, precios, fundamentales y factores
python main.py

# Parte 2 — matriz de exposiciones (betas)
python generar_matriz.py

# Parte 3 — rendimiento esperado por acción
python generar_rendimientos_esperados.py
```

Cada script imprime en consola un resumen de lo que hizo y avisos de
cobertura/calidad (empresas sin datos suficientes, R² bajo, etc.) — vale la
pena leer la salida de la terminal, no solo abrir los archivos.

### Qué hace cada parte, a detalle

**Parte 1 — `main.py`**
1. Construye el universo S&P 500 por año combinando los CSVs de `Data/` y el
   Excel de holdings 2026 (`Datos.py`).
2. Selecciona el Top 100 por market cap de cada año desde 2010 (`top100.py`).
3. Descarga precios/volumen (`yahoo_factor_data.py`) y EPS/acciones en
   circulación vía SEC EDGAR (`sec_edgar_fundamentals.py`) para cada empresa
   seleccionada.
4. Guarda un Excel por empresa en `salidas/empresas/`.
5. Revisa cobertura de precios vs. el año de entrada al top 100.
6. Construye los factores mensuales (`construir_factores.py`) hacia
   `salidas/regresion/`.

**Parte 2 — `generar_matriz.py`** (usa `estimar_exposiciones.py`)
Para cada acción, regresión de su rendimiento (en exceso de `rf`) contra los
6 factores, con la restricción ∑β = 1 impuesta de forma exacta (se despeja
el beta de un factor base y se sustituye en la ecuación — sin optimización
numérica). Guarda `matriz_exposiciones.csv/.xlsx` con alpha, un beta por
factor, R², n_obs y la suma de betas (para validar la restricción).

**Parte 3 — `generar_rendimientos_esperados.py`** (usa
`estimar_rendimiento_esperado.py`)
Para cada factor, calcula un "rendimiento esperado" mensual: rolling mean de
sus últimos 3/6/12 meses (rezagado 1 mes para no ver el futuro), estandarizado
con z-score. Luego, para cada acción:

```
E[R_i,t] = Σ_k  beta_i,k · factor_esperado_k,t
```

usando el beta fijo de la Parte 2 y el factor esperado (que sí varía mes a
mes) de este paso. Guarda un CSV por ventana (3, 6 y 12 meses) más un Excel
resumen con la cobertura de cada uno.

---

## 4. Parámetros que puedes ajustar

Todos están al inicio de cada script (no hay que buscar dentro del código):

- `generar_matriz.py`: `FACTORES`, `FACTOR_BASE`, `RESTRINGIR_SUMA`,
  `MIN_OBS`, `VENTANA_MESES` (recorte de historial), `R2_AVISO`.
- `generar_rendimientos_esperados.py`: `VENTANAS` (lista de rolling means a
  correr), `REZAGO_MESES` (1 = sin look-ahead bias, 0 = literal al
  enunciado), `METODO_Z` (`"expandido"` o `"completo"`), `INCLUIR_ALPHA`.

---

## 5. Notas / limitaciones conocidas

- Empresas que estuvieron muy pocos meses en el top 100 (o cuyo único año de
  membresía coincide con meses donde los factores aún no tenían suficiente
  cobertura) quedan con `n_obs` bajo en la Parte 2 y, por lo tanto, con
  `NaN` en la Parte 3. 
- SEC EDGAR tiene límite de tasa (rate limit); `sec_edgar_fundamentals.py`
  ya mete pausas (`time.sleep`) entre requests — no lo corras en paralelo ni
  bajes la pausa para "ir más rápido", te va a bloquear el `User-Agent`.
