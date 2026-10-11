---
title: Dashboard Café Brasil
emoji: ☕
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# ☕ Dashboard Café Brasil (COE 2023)

Dashboard interactivo en **Streamlit** para el análisis exploratorio (EDA) de las
muestras del *Cup of Excellence* de Brasil.

## Contenido

| Pestaña | Qué muestra |
|---|---|
| Resumen | Métricas clave y registros por año / categoría |
| Estadísticas descriptivas | `describe()` de las columnas clave y nulos |
| Distribuciones | Histogramas interactivos por columna numérica |
| Matriz de correlación | Heatmap con umbral mínimo de \|r\| (Pearson / Spearman) |
| Datos crudos | Tabla filtrada + descarga CSV |
| Ganadores por año | Top N por año, evolución de PONT por categoría |
| Atributos sensoriales | Top atributos correlacionados con PONT |
| PCA | Scree plot, mapa de muestras y loadings |
| Clustering | K-means sobre componentes PCA, perfiles y relación con PONT |
| MFA | Análisis Factorial Múltiple (localización/cultivo vs. sensorial) por región |

## Uso

1. Sube tu Excel desde la barra lateral (`.xlsx` / `.xls`), **o**
2. coloca un archivo llamado `data.xlsx` en la raíz del repositorio (se carga por defecto).

La hoja principal esperada es `BRAZIL COE_Sin Atr`. El encabezado se detecta
automáticamente (busca la fila con `Amostras`, `POS`, `PONT`, ...).

## Ejecutar en local 

```bash
pip install -r requirements.txt
streamlit run app.py
```
## Ejecutada en la cloud de streamlit.app

    https://coffeanajuan.streamlit.app/
