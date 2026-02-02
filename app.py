import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from preprocessing import preprocess_data, clean_engine
from database import init_db, PredictionFeedback
from sqlalchemy.orm import sessionmaker
import plotly.express as px

st.set_page_config(page_title="Car Price Predictor MLOps", layout="wide")

# Theme / CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load database
engine = init_db()
Session = sessionmaker(bind=engine)

# Load models
@st.cache_resource
def load_models():
    models = {}
    encoders = None
    try:
        models['Ridge'] = joblib.load('models/ridge_model.joblib')
        models['Random Forest'] = joblib.load('models/rf_model.joblib')
        models['XGBoost'] = joblib.load('models/xgb_model.joblib')
        encoders = joblib.load('models/encoders.joblib')
        metrics = pd.read_csv('models/metrics_comparison.csv', index_col=0)
    except:
        metrics = None
    return models, metrics, encoders

# Load unique options for dependent filters
@st.cache_data
def get_brand_options():
    try:
        df = pd.read_csv('data/train.csv')
        # We only need unique combinations to keep it light
        return df[['brand', 'engine', 'fuel_type', 'transmission']].drop_duplicates()
    except:
        return pd.DataFrame(columns=['brand', 'engine', 'fuel_type', 'transmission'])

df_options = get_brand_options()

models, metrics, encoders = load_models()

st.title("Sistema Avanzado de Predicción de Precios de Vehículos")
st.markdown("### Plataforma de Comparación de Modelos y MLOps")

tab1, tab2, tab3 = st.tabs(["Predicción", "Comparativa de Modelos", "Monitoreo & Feedback"])

with tab1:
    st.header("Calcular Precio Estimado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if encoders and 'brand' in encoders:
            brand_list = sorted(encoders['brand'].classes_)
        else:
            brand_list = sorted(df_options['brand'].unique()) if not df_options.empty else ["MINI", "Lincoln", "Chevrolet", "Genesis", "Mercedes-Benz", "Audi", "Ford", "BMW", "Tesla"]
            
        brand = st.selectbox("Marca", brand_list)
        
        # Get options for the selected brand
        brand_data = df_options[df_options['brand'] == brand] if not df_options.empty else pd.DataFrame()
        
        model_year = st.slider("Año del Modelo", 1990, 2024, 2018)
        milage = st.number_input("Kilometraje (milas)", value=50000)
    
    with col2:
        fuel_list = sorted(list(brand_data['fuel_type'].dropna().unique().astype(str))) if not brand_data.empty else ["Gasoline", "E85 Flex Fuel", "Diesel", "Hybrid", "Electric"]
        fuel_type = st.selectbox("Tipo de Combustible", fuel_list)
        
        engine_list = sorted(list(brand_data['engine'].dropna().unique().astype(str))) if not brand_data.empty else ["300.0HP 3.0L V6"]
        engine_str = st.selectbox("Motor", engine_list)
        
        trans_list = sorted(list(brand_data['transmission'].dropna().unique().astype(str))) if not brand_data.empty else ["A/T", "M/T", "CVT Transmission", "9-Speed A/T"]
        transmission = st.selectbox("Transmisión", trans_list)
        
        accident = st.radio("Accidentes reportados", ["None reported", "At least 1 accident or damage reported"])
    
    selected_model_name = st.selectbox("Selecciona el algoritmo", list(models.keys()))
    
    if st.button("Predecir Precio"):
        # Prepare input
        input_df = pd.DataFrame([{
            'brand': brand,
            'model_year': model_year,
            'milage': milage,
            'fuel_type': fuel_type,
            'engine': engine_str,
            'transmission': transmission,
            'ext_col': 'Black', # Defaults for simple UI
            'int_col': 'Black',
            'accident': accident,
            'clean_title': 'Yes'
        }])
        
        # Preprocess (we need to be careful with LabelEncoder mapping here, 
        # in a real app we'd save encoders, but for this demo we'll use a hack or assume consistency)
        # Note: Preprocessing script needs fixed encoders for consistency.
        
        try:
            processed_input = preprocess_data(input_df, encoders=encoders)
            
            # Match columns with training
            training_cols = models[selected_model_name].feature_names_in_
            # Reorder
            processed_input = processed_input[training_cols]
            
            # Predict and reverse the log transformation
            log_prediction = models[selected_model_name].predict(processed_input)[0]
            prediction = np.expm1(log_prediction)
            
            st.success(f"### Precio Estimado ({selected_model_name}): ${prediction:,.2f}")
            
            # Store for feedback (simulated prediction entry)
            st.session_state['last_prediction'] = {
                'brand': brand, 'year': model_year, 'milage': milage, 
                'pred': prediction, 'model': selected_model_name
            }
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.write("¿Fue útil el precio? Ayúdanos a mejorar indicando el precio real si lo conoces:")
        actual_price = st.number_input("Precio Real ($)", value=0.0)
        if st.button("Enviar Feedback"):
            session = Session()
            new_feedback = PredictionFeedback(
                brand=st.session_state['last_prediction']['brand'],
                model_year=st.session_state['last_prediction']['year'],
                milage=st.session_state['last_prediction']['milage'],
                predicted_price=st.session_state['last_prediction']['pred'],
                actual_price=actual_price,
                model_used=st.session_state['last_prediction']['model']
            )
            session.add(new_feedback)
            session.commit()
            st.toast("Feedback guardado. ¡Gracias!")

with tab2:
    st.header("Comparación de Rendimiento")
    if metrics is not None:
        st.dataframe(metrics.style.highlight_max(axis=0, subset=['val_r2']))
        
        # Plot R2 comparison
        fig = px.bar(metrics.reset_index(), x='index', y='val_r2', title="R² Score en Validación", labels={'index': 'Modelo', 'val_r2': 'R²'})
        st.plotly_chart(fig)
        
        best_model = metrics['val_r2'].idxmax()
        st.info(f"El modelo recomendado es **{best_model}** por tener el mayor R² y estabilidad.")
    else:
        st.warning("Modelos no entrenados. Ejecuta train.py primero.")

with tab3:
    st.header("Monitoreo de MLOps")
    session = Session()
    feedback_data = pd.read_sql(session.query(PredictionFeedback).statement, session.bind)
    
    if not feedback_data.empty:
        st.subheader("Análisis de Residuos (Feedback Real)")
        feedback_data['error'] = feedback_data['actual_price'] - feedback_data['predicted_price']
        fig_drift = px.scatter(feedback_data, x='predicted_price', y='actual_price', color='model_used', 
                             hover_data=['brand'], title="Predicción vs Real (Feedback)")
        st.plotly_chart(fig_drift)
        
        st.metric("Drift promedio (MAE Feedback)", f"${feedback_data['error'].abs().mean():,.2f}")
    else:
        st.info("No hay datos de feedback aún. Realiza una predicción y envía el precio real.")
