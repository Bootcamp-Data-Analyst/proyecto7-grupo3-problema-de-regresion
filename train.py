import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from preprocessing import preprocess_data

def train_and_evaluate():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/train.csv')
    data, encoders = preprocess_data(df)
    joblib.dump(encoders, 'models/encoders.joblib')
    
    X = data.drop(columns=['price'])
    # Use log transformation for price to handleSkewness
    y = np.log1p(data['price'])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}

    # --- 1. Ridge Regression ---
    def objective_ridge(trial):
        alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
        model = Ridge(alpha=alpha)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
        return -score

    print("Optimizing Ridge...")
    study_ridge = optuna.create_study(direction='minimize')
    study_ridge.optimize(objective_ridge, n_trials=5)
    
    best_ridge = Ridge(**study_ridge.best_params)
    best_ridge.fit(X_train, y_train)
    results['ridge'] = evaluate_model(best_ridge, X_train, X_val, y_train, y_val)
    joblib.dump(best_ridge, 'models/ridge_model.joblib')

    # --- 2. Random Forest ---
    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 100)
        max_depth = trial.suggest_int('max_depth', 5, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()
        return -score

    print("Optimizing Random Forest...")
    study_rf = optuna.create_study(direction='minimize')
    study_rf.optimize(objective_rf, n_trials=2) # Fewer trials due to time
    
    best_rf = RandomForestRegressor(**study_rf.best_params, n_jobs=-1, random_state=42)
    best_rf.fit(X_train, y_train)
    results['rf'] = evaluate_model(best_rf, X_train, X_val, y_train, y_val)
    joblib.dump(best_rf, 'models/rf_model.joblib')

    # --- 3. XGBoost ---
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        }
        model = XGBRegressor(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=2, scoring='neg_root_mean_squared_error').mean()
        return -score

    print("Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=3)
    
    best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42)
    best_xgb.fit(X_train, y_train)
    results['xgb'] = evaluate_model(best_xgb, X_train, X_val, y_train, y_val)
    joblib.dump(best_xgb, 'models/xgb_model.joblib')

    # Save results summary
    pd.DataFrame(results).T.to_csv('models/metrics_comparison.csv')
    return results

def evaluate_model(model, X_train, X_val, y_train, y_val):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
        'train_r2': r2_score(y_train, train_preds),
        'val_r2': r2_score(y_val, val_preds),
        'mae': mean_absolute_error(y_val, val_preds)
    }
    
    # Check overfitting
    overfit = abs(metrics['train_r2'] - metrics['val_r2']) / metrics['train_r2']
    metrics['overfitting_pct'] = overfit * 100
    
    return metrics

if __name__ == "__main__":
    res = train_and_evaluate()
    print("Training finished. Summary:")
    print(res)
