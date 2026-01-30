"""
Script para regenerar los modelos y preprocesadores con la versión actual de scikit-learn
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

print("=" * 80)
print("REGENERANDO MODELOS Y PREPROCESADORES")
print("=" * 80)

# Rutas
data_path = Path(__file__).parent.parent / "data" / "raw"
models_path = Path(__file__).parent.parent / "models"

# Crear carpeta de modelos si no existe
models_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PASO 1: Cargar y Preprocesar Datos
# ============================================================================
print("\n1. Cargando dataset...")
df = pd.read_csv(data_path / "cleaned_train.csv")
print(f"   Dataset cargado: {df.shape}")

# Aplicar transformaciones logarítmicas y winsorización
print("\n2. Aplicando transformaciones logarítmicas y winsorización...")
columns_to_transform = ['price', 'milage', 'horsepower', 'car_age']
for col in columns_to_transform:
    df[col] = np.log1p(df[col])
    df[col] = winsorize(df[col], limits=(0.05, 0.05))
print("   Transformaciones aplicadas")

# Eliminar columnas con multicolinealidad
print("\n3. Eliminando columnas con multicolinealidad...")
df_processed = df.drop(columns=['model_year', 'engine_size_L'], errors='ignore')
print("   Columnas eliminadas")

# Target Encoding para 'brand'
print("\n4. Aplicando Target Encoding para 'brand'...")
encoder_brand = ce.TargetEncoder(cols=['brand'])
Y_temp = df_processed['price']
df_processed['brand'] = encoder_brand.fit_transform(df_processed['brand'], Y_temp)
print("   Target Encoding aplicado")

# One-Hot Encoding para variables categóricas
print("\n5. Aplicando One-Hot Encoding...")
categorical_cols_ohe = df_processed.select_dtypes(include='object').columns.tolist()
Y = df_processed['price']
X = df_processed.drop('price', axis=1)

if categorical_cols_ohe:
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_ohe)
        ],
        remainder='passthrough'
    )
    X = preprocessor.fit_transform(X)
    print(f"   One-Hot Encoding aplicado. Dimensiones finales: {X.shape}")
else:
    preprocessor = None
    print("   No hay columnas categóricas adicionales")

# División de datos
print("\n6. Dividiendo datos en entrenamiento (80%) y prueba (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================================
# PASO 2: Entrenar Modelo XGBoost Optimizado
# ============================================================================
print("\n7. Entrenando modelo XGBoost optimizado...")
optimized_xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=0
)
optimized_xgb_model.fit(X_train, y_train)
print("   Modelo entrenado")

# Evaluación
print("\n8. Evaluando modelo...")
y_pred_xgb = optimized_xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_xgb)

print(f"   MAE:  {mae:.4f}")
print(f"   MSE:  {mse:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")

# ============================================================================
# PASO 3: Guardar Modelos y Preprocesadores
# ============================================================================
print("\n9. Guardando modelos y preprocesadores...")

# Guardar modelo XGBoost
model_file = models_path / "optimized_xgb_model.pkl"
joblib.dump(optimized_xgb_model, model_file)
print(f"   ✓ Modelo guardado: {model_file.name}")

# Guardar encoder de brand
encoder_file = models_path / "encoder_brand.pkl"
joblib.dump(encoder_brand, encoder_file)
print(f"   ✓ Encoder guardado: {encoder_file.name}")

# Guardar preprocesador
if preprocessor:
    preprocessor_file = models_path / "column_transformer_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_file)
    print(f"   ✓ Preprocesador guardado: {preprocessor_file.name}")

print("\n" + "=" * 80)
print("✅ REGENERACIÓN COMPLETADA EXITOSAMENTE")
print("=" * 80)
