# Guía de Construcción: Proyecto 7 - Predicción de Precios (MLOps)

Esta guía detalla el proceso paso a paso para replicar el **Proyecto 7**, un ecosistema completo de Machine Learning (MLOps) diseñado para predecir precios de vehículos usados con un enfoque profesional y escalable.

---

## 1. Concepción y Objetivos

El objetivo principal no es solo crear un modelo que prediga bien, sino un sistema que pueda **evolucionar, monitorearse y desplegarse** de forma segura. Se ha elegido este proyecto porque el mercado de vehículos tiene variables muy ruidosas (marcas, estados, kilometraje) que requieren un preprocesamiento avanzado.

## 2. Configuración del Entorno

Primero, preparamos el espacio de trabajo.

1. **Estructura de Carpetas**:
   - `data/`: Para archivos CSV y la base de datos SQLite.
   - `models/`: Para guardar los artefactos del modelo (.joblib, .pkl).
   - `tests/`: Para validación de código.

2. **Dependencias**: Se utiliza un archivo `requirements.txt` con librerías clave:
   - `pandas/numpy`: Manejo de datos.
   - `scikit-learn/xgboost`: Algoritmos de ML.
   - `optuna`: Optimización de hiperparámetros.
   - `streamlit`: Interfaz de usuario.
   - `sqlalchemy`: Persistencia en DB.

## 3. Ingestión de Datos (`pipeline_ingesta.py`)

**¿Por qué este paso?** En entornos reales, los datos no siempre están listos. Necesitamos un script que tome los CSVs, los limpie inicialmente y los cargue en una base de datos para garantizar la integridad.

- **Tarea**: Leer `train.csv`, conectarse a `data/cars.db` mediante SQLite y guardar los datos.
- **Razón**: Usar una base de datos permite que múltiples aplicaciones accedan a la misma verdad de datos sin depender de archivos planos que pueden corromperse.

## 4. Preprocesamiento Avanzado (`preprocessing.py`)

Este es el corazón de la lógica del negocio. Los datos crudos son difíciles de entender para un modelo.

- **Limpieza del Motor**: Usamos expresiones regulares (`Regex`) para extraer caballos de fuerza (HP) y cilindrada (L) de la columna `engine`.
  - _Razón_: El modelo no entiende "300HP 3.0L V6" como un texto, pero sí entiende que 300 y 3.0 son números que influyen en el precio.
- **Target Encoding para Marcas**: Convertimos la marca del coche en el precio promedio histórico de esa marca.
  - _Razón_: Hay cientos de marcas. El One-Hot Encoding crearía demasiadas columnas (maldición de la dimensionalidad). El Target Encoding captura el "prestigio" de la marca en un solo valor numérico.
- **Variables Binarias**: Convertimos accidentes y títulos limpios en 0 y 1.
- **Manejo de Outliers**: Aplicamos una **transformación logarítmica** al precio (`np.log1p`).
  - _Razón_: Los precios suelen tener una distribución sesgada (pocos coches muy caros). El logaritmo "normaliza" esta distribución, facilitando el aprendizaje del modelo.

## 5. Entrenamiento y Optimización (`train.py`)

No usamos un solo modelo; comparamos varios.

1. **Modelos Utilizados**:
   - **Ridge**: Un modelo lineal simple con regularización para evitar que los coeficientes crezcan demasiado.
   - **Random Forest**: Un conjunto de árboles de decisión (Ensemble) que captura relaciones no lineales.
   - **XGBoost**: Un algoritmo de Gradient Boosting extremadamente potente y eficiente para datos tabulares.
2. **Optuna**: Usamos optimización Bayesiana en lugar de búsqueda aleatoria.
   - _Razón_: Optuna prueba combinaciones de parámetros de forma inteligente, encontrando el mejor modelo mucho más rápido que el ojo humano o una búsqueda exhaustiva.
3. **Validación Cruzada**: Dividimos el set de entrenamiento en varias partes para asegurar que el modelo no esté simplemente memorizando (overfitting).

## 6. Interfaz de Usuario (`app.py`)

Creamos una aplicación con **Streamlit** dividida en pestañas para separar las preocupaciones:

- **Pestaña 1: Predicción**: El usuario ingresa los datos de su coche y obtiene el precio.
- **Pestaña 2: Métricas**: Muestra gráficos de rendimiento de los modelos (R², MAE).
  - _Razón_: Transparencia completa sobre qué modelo es más fiable.
- **Pestaña 3: Monitoreo (MLOps)**: Un sistema de feedback donde los usuarios pueden ingresar el precio real de venta.
  - _Razón_: Esto permite detectar el **Data Drift** (cuando el modelo empieza a fallar porque el mercado cambió).

## 7. Automatización y Gobernanza (`mlops_automation.py`)
Para que el sistema sea de nivel experto, implementamos automatizaciones que cuidan la calidad del modelo a lo largo del tiempo.

- **A/B Testing de Modelos**: Antes de reemplazar un modelo en producción, el script compara el nuevo modelo contra el "campeón" actual. Solo si el nuevo es superior, se realiza el despliegue automático.
- **Detección de Data Drift**: Monitoreamos si los datos que entran (por ejemplo, el kilometraje promedio) cambian drásticamente respecto a los datos históricos. 
  - *Razón*: Si la gente empieza a vender coches mucho más viejos, el modelo entrenado con coches nuevos dejará de ser preciso. Detectar esto a tiempo es vital para re-entrenar el sistema.

## 8. Contenerización (`Dockerfile` & `docker-compose.yml`)
Finalmente, para asegurar que el proyecto funcione en cualquier máquina (Windows, Mac, Linux), lo "empaquetamos" en un contenedor Docker.

- **¿Por qué?**: "En mi máquina funciona" es el mayor problema en software. Docker garantiza que todas las librerías y versiones sean exactamente las mismas en desarrollo y producción.

---

## Conclusión: Por qué elegimos estas herramientas
- **XGBoost**: Elegido como modelo principal por su capacidad de manejar valores faltantes internos y su alta precisión en precios.
- **Optuna**: Elegido para automatizar la sintonía fina, ahorrando horas de trabajo manual.
- **SQLite**: Elegido por su simplicidad (no requiere servidor) pero ofreciendo potencia de SQL para persistir feedback.

---
_Documento creado para la documentación oficial del Proyecto 7._
