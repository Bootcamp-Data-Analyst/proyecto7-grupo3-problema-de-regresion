# 📊 Predicción de Precios de Coches — Proyecto de Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Proyecto de Regresión | Bootcamp Data Analytics – Grupo 3

---

## 📌 Descripción general

Este repositorio contiene un **proyecto completo de Machine Learning** para la **predicción de precios de coches usados**, abarcando todo el flujo de trabajo:

- Limpieza y transformación de datos  
- Análisis exploratorio de datos (EDA)  
- Ingeniería de características y preprocesamiento avanzado  
- Entrenamiento y optimización de modelos  
- Evaluación de resultados  
- **Despliegue de una aplicación interactiva con Streamlit**

El objetivo es construir un modelo robusto y reproducible que permita estimar el precio de un vehículo a partir de sus características principales.

---

## 🧹 1. Limpieza y Transformación de Datos

**Notebook:** [`Data_Cleaning_and_Transformation.ipynb`](notebooks/Data_Cleaning_and_Transformation.ipynb)

### ✔️ Tareas realizadas
- Carga y exploración inicial de `train.csv` y `test.csv`.  
- Análisis de tipos de datos y valores faltantes.  
- Visualización de missing values con `missingno`.  
- Conversión correcta de tipos (`category`, `int`, `float`).  
- Tratamiento de outliers mediante winsorización.  
- Extracción de información del motor:  
  - Potencia (`horsepower`)  
  - Tamaño del motor (`engine_size_L`)  
  - Número de cilindros (`cylinders`)  
- Simplificación de la transmisión:  
  - Automática (AT)  
  - Manual (MT)  
- Guardado de datasets limpios (`cleaned_train.csv`, `cleaned_test.csv`).  

---

## 📊 2. Análisis Exploratorio de Datos (EDA)

**Notebook:** [`EDA_Andrés.ipynb`](notebooks/EDA_Andrés.ipynb)

### 🔹 Análisis Univariado
- Histogramas y estadísticas descriptivas de variables numéricas:  
  `model_year`, `milage`, `car_age`, `engine_size_L`, `cylinders`, `horsepower`, `price`.  
- Distribuciones sesgadas a la derecha detectadas en `price`, `milage` y `horsepower`.  
- Outliers identificados mediante boxplots e IQR.  
- Gráficos de barras para variables categóricas: `brand`, `fuel_type`, `accident`, `clean_title`.  

### 🔹 Análisis Bivariado
- Correlaciones entre variables numéricas:  
  - Correlación perfecta: `model_year` y `car_age`.  
  - Correlación negativa moderada: `milage` vs `price`.  
  - Fuerte relación entre variables del motor.  
- Variables categóricas vs precio:  
  - Vehículos sin accidentes → mayor precio promedio.  
  - `brand` como predictor clave.  

### 🔹 Análisis Multivariado
- Scatter plots de `price` vs `milage` por `fuel_type`.  
- Confirmación de relación inversa y segmentación por tipo de combustible.  

### 🔍 Hallazgos clave
- Necesidad de log-transformación del precio.  
- Presencia de multicolinealidad.  
- Target Encoding adecuado para `brand`.  
- Variables más predictivas: `brand`, `milage`, `accident`, `horsepower`.  

---

## 🤖 3. Modelado y Preprocesamiento

**Notebook:** [`Modelado_Completo_Precios_Coches.ipynb`](notebooks/Modelado_Completo_Precios_Coches.ipynb)

### 🛠️ Preprocesamiento aplicado
- Creación de `car_age` a partir de `model_year`.  
- Transformación logarítmica del target (`np.log1p`).  
- Winsorización de outliers (límites 0.05 / 0.95).  
- **Nota:** el pipeline final aplica winsorización también sobre `price` (target).  
- Target Encoding para `brand`.  
- One-Hot Encoding para variables categóricas.  
- Uso de `ColumnTransformer` para garantizar consistencia entre entrenamiento e inferencia.  

### 📈 Modelos evaluados
- Regresión Lineal (baseline)  
- Random Forest  
- Gradient Boosting  
- KNN  
- SVR  
- XGBoost (base y optimizado)  

### 🔍 Optimización
- `RandomizedSearchCV` para búsqueda de hiperparámetros.  
- Validación cruzada K-Fold.  
- Selección del mejor modelo según RMSE y R².  

---

## 🏆 Resultados del modelo final

**Modelo seleccionado:** XGBoost Optimizado  

**Métricas (sobre target transformado):**

| Métrica | Valor |
|---------|-------|
| MAE     | ~0.31 |
| MSE     | ~0.17 |
| RMSE    | ~0.41 |
| R²      | ~0.70 |

📌 El modelo explica aproximadamente el **70% de la varianza** del precio, un resultado sólido para un problema real de predicción de vehículos.  

> Para volver a la escala original del precio se utiliza `np.expm1()`.

---

## 🖥️ 4. Aplicación Streamlit

**Archivo:** [`streamlit_app.py`](app/streamlit_app.py)

### ✔️ Funcionalidades
- Interfaz limpia y profesional.  
- Navegación por pestañas.  
- Predicción interactiva del precio.  
- Visualizaciones dinámicas con Plotly.  
- Carga correcta de modelos y transformaciones.  
- Preprocesamiento replicado exactamente como en entrenamiento.  
- Manejo de errores y validaciones básicas.  

---

## ⚡ Instalación rápida

```bash

# Crear un entorno virtual
python -m venv env
source env/bin/activate  # Linux / Mac
env\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run app/streamlit_app.py
