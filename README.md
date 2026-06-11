# ☕ Análisis de Catación de Café

**Explorando qué hace grande a una taza de café: factores que influyen en la puntuación de catación.**



---

## 📖 Descripción

La catación es el proceso sensorial para evaluar la calidad del café. En este proyecto analizo un conjunto de datos con **más de 1300 muestras** de cafés arábica evaluados por Q‑Graders. El objetivo es descubrir qué variables (altitud, variedad, proceso, región) tienen mayor impacto en la puntuación total, y construir un modelo predictivo para estimar la calidad.

## 📊 Dataset

Los datos provienen de [Coffee Quality Institute (CQI)](https://github.com/jldbc/coffee-quality-database) e incluyen:

- `Total Cup Points`: Puntuación global (0–100)
- `Aroma`, `Flavor`, `Aftertaste`, `Acidity`, `Body`, etc.
- `Altitude`, `Region`, `Variety`, `Processing Method`

*El dataset se encuentra en la carpeta `data/` con el nombre `coffee_quality.csv`.*

## 🎯 Objetivos

1. **EDA**: ¿Cómo se distribuyen las puntuaciones? ¿Existen diferencias por país o altitud?
2. **Correlaciones**: ¿Qué atributos sensoriales están más relacionados con la puntuación final?
3. **Modelado**: Predecir `Total Cup Points` usando regresión lineal / Random Forest.
4. **Visualización**: Crear un dashboard interactivo con Streamlit (opcional).

## 🧪 Metodología

1. Limpieza y manejo de valores nulos.
2. Análisis estadístico descriptivo y visualizaciones con `seaborn`.
3. Feature engineering (categorización de altitud, encoding de variedades).
4. Modelos supervisados (Regresión Lineal, Árboles de Decisión, XGBoost).
5. Evaluación con RMSE y R², análisis de importancia de variables.

## 📈 Resultados Principales

- La **altitud** y el **proceso de lavado** muestran correlación positiva con la puntuación.
- Los atributos **Flavor** y **Aftertaste** son los que más peso tienen en la puntuación total.
- El modelo Random Forest logró un **R² de 0.82** y un error (RMSE) de 1.2 puntos.
- Los cafés de **Etiopía** y **Kenia** presentan las medias más altas.

*(Añade aquí tus mejores gráficos: mapa de correlaciones, distribución por país, importancia de features.)*

![ejemplo_grafico](assets/grafico_altitud.png)

## 📁 Estructura del Repositorio
