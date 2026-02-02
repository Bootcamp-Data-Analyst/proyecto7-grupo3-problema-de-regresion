import pandas as pd
import joblib
import re

def clean_engine(engine_str):
    try:
        if pd.isna(engine_str): return 0.0
        hp = re.findall(r'(\d+\.?\d*)HP', engine_str)
        hp = float(hp[0]) if hp else 0.0
        return hp
    except:
        return 0.0

def preprocess_test(df, model_features):
    df = df.copy()
    df['hp'] = df['engine'].apply(clean_engine)
    df['car_age'] = 2024 - df['model_year']
    df['has_accident'] = df['accident'].apply(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)
    df['fuel_type'] = df['fuel_type'].fillna('Unknown')
    
    # One-hot encoding
    df_processed = pd.get_dummies(df, columns=['brand', 'fuel_type', 'transmission'], drop_first=True)
    
    # Asegurar que coincidan las columnas con el modelo
    # Rellenar con 0 las columnas que faltan
    missing_cols = set(model_features) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0
    
    # Mantener solo las columnas del modelo en el orden correcto
    df_processed = df_processed[model_features]
    return df_processed

def main():
    try:
        model = joblib.load('car_price_model.pkl')
        features = joblib.load('model_features.pkl')
        test_df = pd.read_csv('test.csv')
        
        print("Preprocesando test.csv...")
        X_test = preprocess_test(test_df, features)
        
        print("Generando predicciones...")
        predictions = model.predict(X_test)
        
        output = pd.DataFrame({
            'id': test_df['id'],
            'price': predictions
        })
        
        output.to_csv('submission_test_predictions.csv', index=False)
        print("Archivo 'submission_test_predictions.csv' generado exitosamente.")
    except FileNotFoundError:
        print("Error: No se encontró el modelo o test.csv. Ejecuta primero el notebook.")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()
