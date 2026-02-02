import pandas as pd
import joblib
import os
import datetime

class DataIngestionPipeline:
    def __init__(self, raw_data_path, feedback_data_path):
        self.raw_data_path = raw_data_path
        self.feedback_data_path = feedback_data_path
        
    def collect_feedback(self, input_features, real_price):
        """Recoge datos de feedback en producción para futuros reentrenamientos."""
        feedback_entry = input_features.copy()
        feedback_entry['price'] = real_price
        feedback_entry['timestamp'] = datetime.datetime.now()
        
        file_exists = os.path.isfile(self.feedback_data_path)
        feedback_entry.to_csv(self.feedback_data_path, mode='a', header=not file_exists, index=False)
        print("Feedback guardado exitosamente.")

    def run_ingestion(self):
        """Lógica para cargar y limpiar nuevos datos que llegan a la carpeta data/new."""
        # TODO: Implementar lógica de escaneo automático de nuevos archivos CSV
        print("Escaneando nuevos datos para ingesta...")
        pass

if __name__ == "__main__":
    pipeline = DataIngestionPipeline('train.csv', 'feedback_production.csv')
    pipeline.run_ingestion()
