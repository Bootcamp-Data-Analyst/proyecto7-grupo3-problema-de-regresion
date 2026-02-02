# Proyecto 7: Predicción de Precios de Vehículos (MLOps)

Este proyecto implementa un ecosistema de Machine Learning preparado para producción, enfocado en la predicción de precios de vehículos usados. La arquitectura sigue principios de MLOps para garantizar la escalabilidad, monitorización y reproducibilidad del modelo.

## Niveles de Entrega Alcanzados

- [x] **Nivel Esencial**: Modelo funcional (XGBoost), EDA profundo, control de Overfitting (< 5%), Aplicación Streamlit e informe de rendimiento.
- [x] **Nivel Medio**: Técnicas de Ensemble, optimización Bayesiana (Optuna), sistema de Feedback y Pipeline de ingestión automatizado.
- [x] **Nivel Avanzado**: Contenerización con Docker, persistencia en base de datos SQL (SQLite) y suite de Tests unitarios.
- [x] **Nivel Experto**: Estrategias de MLOps (A/B Testing, Drift Monitoring y Auto-reemplazo de modelos).

## Tecnologías Utilizadas

- **Lenguaje**: Python 3.9+
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Optuna
- **Análisis y Visualización**: Pandas, Numpy, Matplotlib, Seaborn, Plotly
- **Despliegue e Infraestructura**: Streamlit, Docker, Docker Compose, SQLite
- **Calidad de Código**: Pytest

## Estructura del Repositorio

- `analisis_y_modelo.ipynb`: Cuaderno con el análisis exploratorio y prototipado del modelo.
- `app.py`: Interfaz de usuario interactiva desarrollada en Streamlit.
- `train.py`: Sistema de entrenamiento con optimización de hiperparámetros.
- `models/`: Repositorio de artefactos de modelos entrenados (.joblib, .pkl).
- `data/`: Base de datos SQLite y datasets originales (`train.csv`, `test.csv`).
- `tests/`: Pruebas automatizadas para validar la lógica de preprocesamiento.
- `Dockerfile` & `docker-compose.yml`: Configuración para entorno de producción aislado.

## Guía de Inicio Rápido

### Opción 1: Instalación Local

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar la aplicación:

   ```bash
   streamlit run app.py
   ```

### Opción 2: Docker (Recomendado)

1. Construir e iniciar el contenedor:

   ```bash
   docker-compose up --build
   ```

2. Acceder a `http://localhost:8501`
