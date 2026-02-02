import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def clean_engine(engine_str):
    if pd.isna(engine_str):
        return 0, 0
    hp = re.search(r'(\d+\.?\d*)HP', engine_str)
    l = re.search(r'(\d+\.?\d*)L', engine_str)
    
    hp_val = float(hp.group(1)) if hp else 0
    l_val = float(l.group(1)) if l else 0
    return hp_val, l_val

def preprocess_data(df, encoders=None):
    # Copy to avoid modifying original
    data = df.copy()
    
    # Extract HP and Displacement from engine
    engine_features = data['engine'].apply(clean_engine)
    data['hp'] = engine_features.apply(lambda x: x[0])
    data['displacement'] = engine_features.apply(lambda x: x[1])
    
    # Fill categorical missing values
    categorical_cols = ['fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'brand']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
    
    # Handle accident as binary or ordinal-ish
    if 'accident' in data.columns:
        data['has_accident'] = data['accident'].apply(lambda x: 1 if 'accident' in str(x).lower() else 0)
    
    # Handle clean_title
    if 'clean_title' in data.columns:
        data['has_clean_title'] = data['clean_title'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    # Drop columns that are replaced or have too much cardinality/noise
    cols_to_drop = ['id', 'engine', 'accident', 'clean_title', 'model']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # Encode categorical variables consistently
    cat_to_encode = ['fuel_type', 'transmission', 'ext_col', 'int_col']
    
    if encoders is None:
        # We are in training, fit and return encoders
        new_encoders = {}
        
        # 1. Target Encoding for Brand (Crucial for prestige factor)
        if 'brand' in data.columns and 'price' in df.columns:
            brand_means = df.groupby('brand')['price'].mean().to_dict()
            data['brand'] = data['brand'].map(brand_means).fillna(df['price'].mean())
            new_encoders['brand_mapping'] = brand_means
            new_encoders['global_mean_price'] = df['price'].mean()
        
        for col in cat_to_encode:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                new_encoders[col] = le
        return data, new_encoders
    else:
        # We are in inference, use provided encoders
        
        # 1. Apply Brand Mapping
        if 'brand' in data.columns:
            brand_map = encoders.get('brand_mapping', {})
            global_mean = encoders.get('global_mean_price', 35000)
            data['brand'] = data['brand'].map(brand_map).fillna(global_mean)

        for col in cat_to_encode:
            if col in data.columns:
                le = encoders[col]
                # Handle unknown categories safely
                data[col] = data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                data[col] = le.transform(data[col].astype(str))
        return data

if __name__ == "__main__":
    # Test script
    train = pd.read_csv('train.csv')
    processed = preprocess_data(train)
    print("Processed columns:", processed.columns)
    print(processed.head())
