"""
Sistema de Logging y Monitoreo de Predicciones
Guarda predicciones y feedback en SQLite para análisis posterior
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

class ModelMonitor:
    def __init__(self, db_path="model_monitoring.db"):
        self.db_path = Path(__file__).parent / db_path
        self.init_database()
    
    def init_database(self):
        """Crear tablas si no existen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de predicciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                brand TEXT NOT NULL,
                model_year INTEGER NOT NULL,
                fuel_type TEXT NOT NULL,
                milage REAL NOT NULL,
                transmission TEXT NOT NULL,
                accident TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                feedback_status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Tabla de métricas diarias
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                predictions_count INTEGER,
                feedback_count INTEGER,
                mae REAL,
                rmse REAL,
                r2 REAL,
                avg_error REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, brand, model_year, fuel_type, milage, 
                       transmission, accident, predicted_price):
        """Guardar una predicción"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (brand, model_year, fuel_type, milage, transmission, accident, predicted_price)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (brand, model_year, fuel_type, milage, transmission, accident, predicted_price))
        
        conn.commit()
        pred_id = cursor.lastrowid
        conn.close()
        
        return pred_id
    
    def add_feedback(self, pred_id, actual_price):
        """Agregar precio real para comparar"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_price = ?, feedback_status = 'completed'
            WHERE id = ?
        ''', (actual_price, pred_id))
        
        conn.commit()
        conn.close()
    
    def get_daily_metrics(self):
        """Obtener métricas del día"""
        conn = sqlite3.connect(self.db_path)
        
        # Predicciones de hoy
        today = datetime.now().date()
        query = '''
            SELECT 
                COUNT(*) as predictions_count,
                SUM(CASE WHEN actual_price IS NOT NULL THEN 1 ELSE 0 END) as feedback_count,
                ROUND(AVG(ABS(predicted_price - actual_price)), 2) as mae,
                ROUND(SQRT(AVG(POW(predicted_price - actual_price, 2))), 2) as rmse
            FROM predictions
            WHERE DATE(timestamp) = ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(today,))
        conn.close()
        
        return df.iloc[0].to_dict() if len(df) > 0 else {}
    
    def get_predictions_history(self, limit=100):
        """Obtener historial de predicciones"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def export_for_retraining(self, min_feedback=50):
        """Exportar datos con feedback para reentrenamiento"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                brand, model_year, fuel_type, milage, 
                transmission, accident, actual_price as price
            FROM predictions
            WHERE feedback_status = 'completed'
            AND actual_price IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) >= min_feedback:
            export_path = Path(__file__).parent.parent / "data" / "new_predictions.csv"
            df.to_csv(export_path, index=False)
            return export_path, len(df)
        else:
            return None, len(df)
    
    def get_performance_degradation(self):
        """Detectar si hay degradación de performance"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                DATE(timestamp) as date,
                ROUND(AVG(ABS(predicted_price - actual_price)), 2) as daily_mae
            FROM predictions
            WHERE actual_price IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) >= 2:
            latest_mae = df.iloc[0]['daily_mae']
            previous_mae = df.iloc[1:]['daily_mae'].mean()
            
            degradation = ((latest_mae - previous_mae) / previous_mae * 100) if previous_mae > 0 else 0
            
            return {
                'latest_mae': latest_mae,
                'avg_previous_mae': previous_mae,
                'degradation_percent': degradation,
                'alert': degradation > 10  # Alerta si > 10% degradación
            }
        
        return None

if __name__ == "__main__":
    # Test del sistema
    monitor = ModelMonitor()
    print("✅ Base de datos de monitoreo inicializada")
    print(f"📁 Ubicación: {monitor.db_path}")
