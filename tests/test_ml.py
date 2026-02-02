import pytest
import pandas as pd
import numpy as np
import os
from preprocessing import preprocess_data, clean_engine

def test_clean_engine():
    hp, l = clean_engine("300.0HP 3.0L V6")
    assert hp == 300.0
    assert l == 3.0
    
    hp, l = clean_engine("Electric Motor")
    assert hp == 0
    assert l == 0

def test_preprocessing_pipeline():
    sample_df = pd.DataFrame([{
        'id': 0, 'brand': 'Audi', 'model': 'A4', 'model_year': 2020,
        'milage': 10000, 'fuel_type': 'Gasoline', 'engine': '250HP 2.0L',
        'transmission': 'A/T', 'ext_col': 'Black', 'int_col': 'Black',
        'accident': 'None reported', 'clean_title': 'Yes', 'price': 30000
    }])
    processed = preprocess_data(sample_df)
    
    assert 'hp' in processed.columns
    assert 'displacement' in processed.columns
    assert 'has_accident' in processed.columns
    assert 'id' not in processed.columns
    assert 'engine' not in processed.columns

def test_metrics_file_exists():
    # This assumes we ran training
    if os.path.exists('models/metrics_comparison.csv'):
        df = pd.read_csv('models/metrics_comparison.csv')
        assert not df.empty
        assert 'val_r2' in df.columns
