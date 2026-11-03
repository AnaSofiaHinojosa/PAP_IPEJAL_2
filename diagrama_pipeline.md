# PAP_IPEJAL_2: diagrama de conexiones entre archivos

Las flechas continuas (`-->`) son `import` entre scripts. Las flechas punteadas (`-.->`) son archivos de datos que se leen o se escriben. Las variantes `sin_restriccion` y `sin_estandarizar` van agrupadas en un solo nodo por etapa.

```mermaid
flowchart TD

    %% ===== ENTRADAS =====
    subgraph ENTRADAS["Entradas"]
        D1[("Data/*.csv<br/>holdings 2007-2025")]
        D2[("Data/holdings-daily-us-en-spy.xlsx<br/>holdings 2026")]
        D3[("Data/factset/ (opcional)")]
        EXT1(["Yahoo Finance"])
        EXT2(["SEC EDGAR"])
    end

    %% ===== PARTE 1 =====
    subgraph P1["Parte 1: Datos y factores"]
        MAIN["main.py<br/>(orquestador)"]
        DATOS["Datos.py<br/>universo S&P 500 por año"]
        TOP["top100.py<br/>top 100 anual"]
        YAH["yahoo_factor_data.py<br/>precios, volumen, splits"]
        SEC["sec_edgar_fundamentals.py<br/>EPS y acciones"]
        FAC["construir_factores.py<br/>6 factores long-short"]
    end

    D1 -.-> DATOS
    D2 -.-> DATOS
    D3 -.-> DATOS
    EXT1 -.-> YAH
    EXT2 -.-> SEC

    MAIN --> DATOS
    MAIN --> TOP
    MAIN --> YAH
    MAIN --> SEC
    MAIN --> FAC
    DATOS --> TOP

    TOP -.-> SEL[("seleccion_anual_top100.csv")]
    MAIN -.-> EMP[("salidas/empresas/TICKER.xlsx<br/>salidas/mercado.xlsx")]
    SEL -.-> FAC
    EMP -.-> FAC
    FAC -.-> REG[("salidas/regresion/<br/>rendimientos.csv<br/>rendimientos_miembros.csv<br/>factores.csv<br/>membresia.csv")]

    %% ===== PARTE 2 =====
    subgraph P2["Parte 2: Matriz de exposiciones"]
        EXPO["estimar_exposiciones.py<br/>regresion con suma beta = 1"]
        GMAT["generar_matriz.py"]
        GMATV["generar_matriz_sin_restriccion.py<br/>generar_matriz_sin_estandarizar.py<br/>generar_matriz_sin_restriccion_sin_estandarizar.py"]
    end

    REG -.-> GMAT
    GMAT --> EXPO
    GMATV --> GMAT
    GMATV --> EXPO
    GMAT -.-> MEXP[("salidas/exposiciones/<br/>matriz_exposiciones.csv<br/>factores_estandarizacion.csv")]

    %% ===== PARTE 3 =====
    subgraph P3["Parte 3: Rendimiento esperado"]
        ERES["estimar_rendimiento.py<br/>factor esperado + E[R]"]
        GREN["generar_rendimientos.py"]
        GRENV["generar_rendimientos_sin_restriccion.py<br/>generar_rendimientos_sin_estandarizar.py<br/>generar_rendimientos_sin_restriccion_sin_estandarizar.py"]
    end

    MEXP -.-> GREN
    REG -.-> GREN
    GREN --> EXPO
    GREN --> ERES
    GRENV --> GREN
    GREN -.-> RESP[("salidas/rendimientos_esperados/")]

    %% ===== PARTE 4 =====
    subgraph P4["Parte 4: Backtesting y seleccion"]
        BACK["backtesting.py<br/>betas walk-forward + IC"]
        OPT["optimizar_modelos.py<br/>rejilla de 36 modelos"]
        OPTV["optimizar_modelos_sin_restriccion.py<br/>optimizar_modelos_sin_estandarizar.py<br/>optimizar_modelos_sin_restriccion_sin_estandarizar.py"]
        COMP["comparar_modelos.py<br/>comparacion 2x2 de los 4 modelos"]
    end

    REG -.-> OPT
    OPT --> BACK
    OPT --> EXPO
    BACK --> EXPO
    BACK --> ERES
    OPTV --> OPT
    OPT -.-> BT[("salidas/backtesting*/<br/>comparacion_modelos.csv<br/>mejor_modelo_ic_serie.csv")]
    BT -.-> COMP
    COMP --> OPT
    COMP -.-> CMP[("salidas/comparacion_modelos/")]

    %% ===== UTILIDADES =====
    subgraph UTIL["Utilidades"]
        VAL["Validar Salidas.py"]
        DIAG["Diagnostico.py"]
    end

    EMP -.-> VAL
    VAL -.-> VCSV[("salidas/validacion.csv")]
    DIAG --> SEC
    DIAG -.-> DCSV[("salidas/diagnostico_xbrl.csv")]

    %% ===== ESTILOS =====
    classDef script fill:#dbeafe,stroke:#1d4ed8,color:#111;
    classDef datos fill:#fef9c3,stroke:#a16207,color:#111;
    classDef externo fill:#dcfce7,stroke:#15803d,color:#111;

    class MAIN,DATOS,TOP,YAH,SEC,FAC,EXPO,GMAT,GMATV,ERES,GREN,GRENV,BACK,OPT,OPTV,COMP,VAL,DIAG script;
    class D1,D2,D3,SEL,EMP,REG,MEXP,RESP,BT,CMP,VCSV,DCSV datos;
    class EXT1,EXT2 externo;
```

## Leyenda

| Color | Significado |
|---|---|
| Azul | Scripts `.py` |
| Amarillo | Archivos de datos (entradas y salidas) |
| Verde | Fuentes externas (Yahoo Finance, SEC EDGAR) |

## Orden de ejecución

1. `python main.py` (Parte 1)
2. `python generar_matriz.py` (Parte 2)
3. `python generar_rendimientos.py` (Parte 3)
4. `python optimizar_modelos.py` (Parte 4), más las variantes para los modelos 2, 3 y 4
5. `python comparar_modelos.py` (compara los 4 modelos)
