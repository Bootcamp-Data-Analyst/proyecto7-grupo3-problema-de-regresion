# 📊 Predicción de Precios de Coches — Proyecto de Machine Learning

## Proyecto de Regresión | Bootcamp Data Analytics – Grupo 3

---

## 📌 Descripción general

Este repositorio contiene un **proyecto completo de Machine Learning** para la **predicción de precios de coches usados**, cubriendo todo el flujo de trabajo:

- Limpieza y transformación de datos  
- Análisis exploratorio de datos (EDA)  
- Ingeniería de características y preprocesamiento avanzado  
- Entrenamiento y optimización de modelos  
- Evaluación de resultados  
- **Despliegue de una aplicación interactiva con Streamlit**

El objetivo es construir un modelo robusto y reproducible que permita estimar el precio de un vehículo a partir de sus características principales.

---

## 🗂️ Estructura del proyecto

proyecto7-grupo3-problema-de-regresion
├── notebooks/
│ ├── Data_Cleaning_and_Transformation.ipynb
│ ├── EDA_Andrés.ipynb
│ ├── Modelado_Completo_Precios_Coches.ipynb
├── app/
│ ├── streamlit_app.py
│ ├── rebuild_models.py
├── data/
│ ├── raw/
│ │ ├── train.csv
│ │ ├── test.csv
  ├── processed/
│ │ ├── cleaned_train.csv
│ │ ├── cleaned_test.csv
├── models/
│ ├── optimized_xgb_model.pkl
│ ├── encoder_brand.pkl
│ ├── column_transformer_preprocessor.pkl
├── requirements.txt
└── README.md


---

## 🧹 1. Limpieza y Transformación de Datos

**Notebook:** `notebooks/Data_Cleaning_and_Transformation.ipynb`

### ✔️ Principales tareas realizadas
- Carga y exploración inicial de `train.csv` y `test.csv`
- Análisis de tipos de datos y valores faltantes
- Visualización de missing values con `missingno`
- Conversión correcta de tipos (`category`, `int`, `float`)
- Tratamiento de outliers mediante winsorización
- Guardado de datasets limpios (`cleaned_train.csv`, `cleaned_test.csv`)

## 📊 2. Análisis Exploratorio de Datos (EDA)

**Notebook:** `notebooks/EDA_Andrés.ipynb`

### ✔️ Fortalezas
- Análisis univariado y bivariado completo
- Histogramas, boxplots, scatter plots y heatmaps
- Identificación de relaciones no lineales
- Detección de multicolinealidad
- Documentación clara y visualizaciones interpretables

## 🤖 3. Modelado y Preprocesamiento

**Notebook:** `notebooks/Modelado_Completo_Precios_Coches.ipynb`

### 🛠️ Preprocesamiento aplicado
- Creación de `car_age` a partir de `model_year`
- Transformación logarítmica (`np.log1p`)
- Winsorización (límites 0.05 / 0.05)
- **Nota importante:** en la versión final el pipeline **aplica winsorización también sobre `price` (target)**
- Target Encoding para `brand`
- One-Hot Encoding para variables categóricas
- Uso de `ColumnTransformer` para garantizar consistencia entre entrenamiento e inferencia

---
### 📈 Modelos evaluados
- Regresión Lineal (baseline)
- Random Forest
- Gradient Boosting
- KNN
- SVR
- XGBoost (base y optimizado)

### 🔍 Optimización
- `RandomizedSearchCV`
- Validación cruzada K-Fold
- Selección del mejor modelo según RMSE y R²

---

## 🏆 Resultados del modelo final

**Modelo seleccionado:** XGBoost Optimizado  

Métricas (sobre target transformado):

| Métrica | Valor |
|-------|------|
| MAE | ~0.31 |
| MSE | ~0.17 |
| RMSE | ~0.41 |
| R² | ~0.70 |

📌 El modelo explica aproximadamente el **70% de la varianza** del precio, un resultado sólido para un problema real de precios de vehículos.

> Para volver a la escala original del precio se utiliza `np.expm1()`.

---

## 🖥️ 4. Aplicación Streamlit

**Archivo:** `app/streamlit_app.py`

### ✔️ Funcionalidades
- Interfaz limpia y profesional
- Navegación por pestañas
- Predicción interactiva del precio
- Visualizaciones dinámicas con Plotly
- Carga correcta de modelos y transformaciones
- Preprocesamiento replicado exactamente como en entrenamiento
- Manejo de errores y validaciones básicas


