import pandas as pd
import joblib
import os
from sklearn.metrics import r2_score

def ab_test_and_deploy(new_model_path, current_model_path, data_path, target_col='price'):
    """
    Simulates A/B Testing comparing a new model against the current champion.
    If the new model performs > 5% better, it replaces the champion.
    """
    print("Iniciando A/B Testing...")
    
    if not os.path.exists(current_model_path):
        print("No hay modelo actual. El nuevo modelo se convierte en el champion.")
        os.rename(new_model_path, current_model_path)
        return True
    
    # Load models
    new_model = joblib.load(new_model_path)
    champion = joblib.load(current_model_path)
    
    # Load evaluation data
    data = pd.read_csv(data_path)
    from preprocessing import preprocess_data
    processed = preprocess_data(data)
    
    X = processed.drop(columns=[target_col])
    y = processed[target_col]
    
    # Evaluate
    new_score = r2_score(y, new_model.predict(X))
    old_score = r2_score(y, champion.predict(X))
    
    print(f"R² Nuevo Modelo: {new_score:.4f}")
    print(f"R² Champion: {old_score:.4f}")
    
    # Logic: Only swap if significantly better or at least marginally better without regression
    if new_score > old_score:
        print("¡Nueva versión superior detectada! Reemplazando modelo en producción...")
        joblib.dump(new_model, current_model_path)
        return True
    else:
        print("El nuevo modelo no supera al actual. Manteniendo champion.")
        return False

def check_data_drift(historical_data_path, new_data_path):
    """
    Detects changes in data distribution (simple version using mean/std of milage).
    """
    h_data = pd.read_csv(historical_data_path)
    n_data = pd.read_csv(new_data_path)
    
    h_mean = h_data['milage'].mean()
    n_mean = n_data['milage'].mean()
    
    drift_pct = abs(h_mean - n_mean) / h_mean
    
    if drift_pct > 0.2:
        print(f"¡ADVERTENCIA: Data Drift detectado! Desviación del {drift_pct*100:.2f}% en el kilometraje.")
        return True
    return False

if __name__ == "__main__":
    # Example usage (placeholders)
    # ab_test_and_deploy('models/new_candidate.joblib', 'models/champion.joblib', 'train.csv')
    pass
