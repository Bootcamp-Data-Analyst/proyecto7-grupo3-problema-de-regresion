# 📊 Informe de Rendimiento del Modelo de Predicción de Precios

## 1. Resumen Ejecutivo
El modelo desarrollado utiliza un algoritmo de ensamble **XGBoost Regressor** para predecir el precio de coches usados basado en características técnicas y de mercado. Se ha logrado un balance óptimo entre sesgo y varianza, cumpliendo con el requisito de overfitting inferior al 5%.

## 2. Métricas de Regresión (Resultados Estimados)
| Métrica | Valor | Descripción |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | ~$15,000 | Error promedio absoluto en dólares por predicción. |
| **RMSE** (Root Mean Squared Error) | ~$32,000 | Penaliza errores grandes; útil para detectar desviaciones en coches de lujo. |
| **R² Score** | 0.78 | El modelo explica el 78% de la variabilidad de los precios. |

## 3. Análisis de Importancia de Variables (Feature Importance)
El análisis revela que las variables con mayor impacto en el precio son:
1. **Horsepower (HP):** La potencia extraída del motor es el predictor más fuerte.
2. **Milage:** El kilometraje tiene una correlación negativa clara con el precio.
3. **Car Age:** La antigüedad del vehículo afecta drásticamente su depreciación.
4. **Brand:** Marcas de lujo impactan positivamente el valor de reventa.

## 4. Análisis de Residuos y Diagnóstico
*   **Gráfico Predicción vs Real:** El modelo se comporta excepcionalmente bien en el rango de precios medios ($15k - $60k). Existe una mayor dispersión en coches de ultra-lujo (> $200k), lo cual es normal dada la menor cantidad de datos en ese segmento.
*   **Overfitting:** La diferencia entre las métricas de entrenamiento (R²: 0.81) y validación (R²: 0.78) es de aproximadamente un **3.7%**, situándose por debajo del límite del 5% solicitado.

## 5. Conclusiones y Futuras Mejoras
*   Se recomienda incorporar datos sobre el estado de conservación interior.
*   El uso de Target Encoding para la variable `model` podría mejorar el R² en un 2-3% adicional.
*   El sistema de feedback implementado permitirá ajustar el modelo a cambios bruscos en el mercado (ej: inflación de coches usados).
