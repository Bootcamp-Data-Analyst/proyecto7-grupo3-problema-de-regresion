# Informe Técnico de Desarrollo: Car Price MLOps

## 1. Introducción
Este proyecto ha sido desarrollado siguiendo una arquitectura de niveles progresivos, desde un modelo funcional básico hasta una plataforma de MLOps automatizada. El objetivo es predecir el precio de venta de vehículos usados basándose en datos técnicos y de historial.

## 2. Análisis Exploratorio (EDA)
Se identificaron las siguientes tendencias clave:
- **Kilometraje**: Correlación negativa fuerte con el precio.
- **Marca**: Las marcas de lujo (Tesla, BMW, Mercedes) mantienen rangos de precio significativamente superiores.
- **Estado (Accidentes)**: Los vehículos con "None reported" tienen una prima de valor del ~15-20%.
- **Limpieza de datos**: Se realizó extracción de potencia (HP) y cilindrada (L) de la columna de texto `engine` para linealizar la entrada numérica.

## 3. Comparativa de Modelos
Se evaluaron tres algoritmos distintos:

| Modelo | Complejidad | Ventajas | Resultado Esperado |
|--------|------------|----------|-------------------|
| **Ridge Regression** | Baja | Rápido, interpretable | Baseline estable |
| **Random Forest** | Media | Robusto a outliers | Alta precisión inicial |
| **XGBoost** | Alta | Optimización fina, manejo de nulos | **Campeón Final** |

### ¿Por qué XGBoost?
Se seleccionó **XGBoost** como el modelo final debido a su capacidad para capturar interacciones complejas entre variables categóricas (como Brand y Fuel Type) y su robustez contra el overfitting gracias a la regularización interna. Además, mediante la optimización con **Optuna (Bayesian Search)**, se logró ajustar los hiperparámetros para reducir el error de validación por debajo del umbral objetivo.

## 4. Arquitectura MLOps
- **Monitorización de Drift**: El sistema compara la distribución del kilometraje entrante con el histórico.
- **A/B Testing**: Implementado un pipeline que solo actualiza el modelo en producción si el nuevo candidato supera al actual.
- **Feedback Loop**: La base de datos SQLite recoge precios reales introducidos por usuarios para el reentrenamiento futuro.

## 5. Pruebas y Calidad
- **Tests unitarios**: Ubicados en `tests/`, validan la integridad del preprocesamiento.
- **Docker**: La aplicación está completamente aislada en un contenedor para garantizar la reproducibilidad.
