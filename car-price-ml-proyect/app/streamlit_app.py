import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats.mstats import winsorize
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar página
st.set_page_config(
    page_title="Predictor de Precios de Coches",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Cargar modelo y preprocesadores
@st.cache_resource
def load_models():
    """Cargar el modelo y los preprocesadores"""
    model_path = Path(__file__).parent.parent / "models"
    
    try:
        model = joblib.load(model_path / "optimized_xgb_model.pkl")
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None, None
    
    try:
        preprocessor = joblib.load(model_path / "column_transformer_preprocessor.pkl")
    except Exception as e:
        st.error(f"Error cargando preprocesador: {e}")
        return None, None, None
    
    try:
        encoder_brand = joblib.load(model_path / "encoder_brand.pkl")
    except Exception as e:
        st.error(f"Error cargando encoder de marca: {e}")
        return None, None, None
    
    return model, preprocessor, encoder_brand

# Cargar datos de entrenamiento para visualizaciones
@st.cache_data
def load_training_data():
    """Cargar datos de entrenamiento para estadísticas"""
    data_path = Path(__file__).parent.parent / "data" / "raw"
    df = pd.read_csv(data_path / "cleaned_train.csv")
    return df

# Función para preparar datos
def prepare_input(data):
    """Preparar input para el modelo"""
    df = pd.DataFrame([data])
    return df

# Aplicación principal
def main():
    st.title("🚗 Predictor de Precios de Coches")
    st.markdown("---")
    
    # Cargar modelo
    model, preprocessor, encoder_brand = load_models()
    df_train = load_training_data()
    
    # Crear tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis", "🔮 Predicción", "📈 Estadísticas", "🎯 Performance"])
    
    # TAB 1: Análisis
    with tab1:
        st.subheader("Análisis del Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Coches", len(df_train))
        with col2:
            st.metric("Precio Promedio", f"${df_train['price'].mean():,.0f}")
        with col3:
            st.metric("Precio Máximo", f"${df_train['price'].max():,.0f}")
        
        st.markdown("---")
        
        # Gráficos de análisis
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de precios
            fig = px.histogram(df_train, x='price', nbins=50, 
                              title='Distribución de Precios',
                              labels={'price': 'Precio ($)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top marcas
            top_brands = df_train['brand'].value_counts().head(10)
            fig = px.bar(top_brands, 
                        title='Top 10 Marcas más Comunes',
                        labels={'index': 'Marca', 'value': 'Cantidad'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Precio por tipo de combustible
            fig = px.box(df_train, x='fuel_type', y='price',
                        title='Distribución de Precios por Tipo de Combustible',
                        labels={'fuel_type': 'Tipo de Combustible', 'price': 'Precio ($)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Precio vs Año del Modelo
            fig = px.scatter(df_train, x='model_year', y='price',
                           title='Precio vs Año del Modelo',
                           labels={'model_year': 'Año del Modelo', 'price': 'Precio ($)'},
                           opacity=0.5)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Predicción
    with tab2:
        st.subheader("Predice el Precio de tu Coche")
        st.info("ℹ️ Complete los datos principales del coche para obtener una predicción de precio")
        
        # Crear formulario en 2 columnas
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox(
                "🏎️ Marca del Coche",
                sorted(df_train['brand'].unique()),
                key="brand"
            )
            
            # Filtrar datos por marca
            df_brand_filtered = df_train[df_train['brand'] == brand]
            
            model_year = st.slider(
                "📅 Año del Modelo",
                min_value=int(df_train['model_year'].min()),
                max_value=int(df_train['model_year'].max()),
                value=2015,
                step=1
            )
            
            # Solo mostrar tipos de combustible que tiene esta marca
            fuel_options = sorted(df_brand_filtered['fuel_type'].unique())
            fuel_type = st.selectbox(
                "⛽ Tipo de Combustible",
                fuel_options,
                key="fuel"
            )
        
        with col2:
            milage = st.number_input(
                "📊 Millaje (en millas)",
                min_value=0.0,
                max_value=float(df_train['milage'].max()),
                value=50000.0,
                step=1000.0
            )
            
            # Solo mostrar transmisiones que tiene esta marca
            transmission_options = sorted(df_brand_filtered['transmission_simple'].unique())
            transmission = st.selectbox(
                "⚙️ Transmisión",
                transmission_options,
                key="transmission"
            )
            
            # Solo mostrar tipos de accidentes que tiene esta marca
            accident_options = sorted(df_brand_filtered['accident'].unique())
            accident = st.selectbox(
                "⚠️ Accidentes Reportados",
                accident_options,
                key="accident"
            )
        
        st.markdown("---")
        
        # Botón de predicción
        if st.button("🔮 Predecir Precio", use_container_width=True):
            try:
                from scipy.stats.mstats import winsorize
                
                # Calcular car_age basado en model_year
                current_year = 2026
                car_age = current_year - model_year
                
                # Valores por defecto para variables no mostradas
                # Usar promedios/moda del dataset específico de la marca
                horsepower = float(df_brand_filtered['horsepower'].mean())  # Promedio de marca
                engine_size = float(df_brand_filtered['engine_size_L'].mean())  # Promedio de marca
                cylinders = float(df_brand_filtered['cylinders'].mode()[0])  # Moda de marca
                clean_title = df_brand_filtered['clean_title'].mode()[0]  # Moda de marca
                
                # Crear input con todas las características necesarias
                input_data = {
                    'brand': brand,
                    'model_year': model_year,
                    'milage': milage,
                    'fuel_type': fuel_type,
                    'accident': accident,
                    'clean_title': clean_title,
                    'horsepower': horsepower,
                    'engine_size_L': engine_size,
                    'cylinders': cylinders,
                    'transmission_simple': transmission,
                    'car_age': car_age,
                    'price': 0  # Dummy value
                }
                
                # Preparar dataframe
                df_input = prepare_input(input_data)
                
                # PASO 1: Aplicar Target Encoder a 'brand'
                df_input['brand'] = encoder_brand.transform(df_input[['brand']])
                
                # PASO 2: Aplicar transformaciones logarítmicas y winsorización
                cols_to_log = ['price', 'milage', 'horsepower', 'car_age']
                for col in cols_to_log:
                    df_input[col] = np.log1p(df_input[col])
                    df_input[col] = winsorize(df_input[col], limits=(0.05, 0.05))
                
                # PASO 3: Eliminar columnas que no están en el modelo
                df_input = df_input.drop(columns=['price', 'model_year', 'engine_size_L'], errors='ignore')
                
                # PASO 4: Aplicar One-Hot Encoding
                X_processed = preprocessor.transform(df_input)
                
                # Hacer predicción
                prediction = model.predict(X_processed)[0]
                
                # Revertir la transformación logarítmica
                prediction_original = np.expm1(prediction)
                
                # Mostrar resultado
                st.success("✅ Predicción Realizada")
                st.markdown("---")
                
                # Mostrar precio predicho en grande
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("💰 Precio Predicho", f"${prediction_original:,.2f}", 
                             delta=None)
                
                with col2:
                    # Comparar con promedio
                    avg_price_log = df_train['price'].mean()
                    avg_price = np.expm1(avg_price_log)
                    diff_percent = ((prediction_original - avg_price) / avg_price) * 100
                    st.metric("📊 vs Promedio", f"{diff_percent:+.1f}%")
                
                # Información adicional
                st.markdown("---")
                st.subheader("📋 Resumen del Coche")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Marca:** {brand}")
                    st.write(f"**Año:** {model_year}")
                    st.write(f"**Millaje:** {milage:,.0f} millas")
                
                with col2:
                    st.write(f"**Combustible:** {fuel_type}")
                    st.write(f"**Transmisión:** {transmission}")
                    st.write(f"**Accidentes:** {accident}")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
    
    # TAB 3: Estadísticas
    with tab3:
        st.subheader("Estadísticas Detalladas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Estadísticas de Precio")
            price_stats = df_train['price'].describe()
            st.dataframe(price_stats, use_container_width=True)
        
        with col2:
            st.markdown("### Correlación con Precio")
            numeric_cols = df_train.select_dtypes(include=[np.number]).columns
            correlations = df_train[numeric_cols].corr()['price'].sort_values(ascending=False)
            
            fig = px.bar(correlations.drop('price'),
                        title='Correlación de Variables con el Precio',
                        labels={'value': 'Correlación', 'index': 'Variable'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Gráficos adicionales
        col1, col2 = st.columns(2)
        
        with col1:
            # Precio promedio por marca
            brand_avg = df_train.groupby('brand')['price'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(brand_avg,
                        title='Precio Promedio por Marca (Top 15)',
                        labels={'value': 'Precio Promedio ($)', 'index': 'Marca'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Edad del coche vs Precio
            fig = px.scatter(df_train, x='car_age', y='price',
                           title='Edad del Coche vs Precio',
                           labels={'car_age': 'Edad (años)', 'price': 'Precio ($)'},
                           opacity=0.5)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Performance del Modelo
    with tab4:
        st.subheader("🎯 Performance y Validación del Modelo")
        
        # Cargar datos de test para análisis
        data_path = Path(__file__).parent.parent / "data" / "raw"
        df_full = pd.read_csv(data_path / "cleaned_train.csv")
        
        # Preparar datos igual que en rebuild_models.py
        from scipy.stats.mstats import winsorize
        import category_encoders as ce
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        # Aplicar transformaciones
        columns_to_transform = ['price', 'milage', 'horsepower', 'car_age']
        for col in columns_to_transform:
            df_full[col] = np.log1p(df_full[col])
            df_full[col] = winsorize(df_full[col], limits=(0.05, 0.05))
        
        df_full = df_full.drop(columns=['model_year', 'engine_size_L'], errors='ignore')
        
        encoder_brand_eval = joblib.load(Path(__file__).parent.parent / "models" / "encoder_brand.pkl")
        Y_temp = df_full['price']
        df_full['brand'] = encoder_brand_eval.transform(df_full[['brand']])
        
        categorical_cols = df_full.select_dtypes(include='object').columns.tolist()
        Y = df_full['price']
        X = df_full.drop('price', axis=1)
        
        # Guardar nombres de features ANTES de aplicar preprocessor
        X_feature_names = X.columns.tolist()
        
        if categorical_cols:
            preprocessor_eval = joblib.load(Path(__file__).parent.parent / "models" / "column_transformer_preprocessor.pkl")
            X = preprocessor_eval.transform(X)
            # Obtener nombres de features después del encoding
            try:
                X_feature_names = preprocessor_eval.get_feature_names_out(X_feature_names).tolist()
            except:
                pass
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Mostrar métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("R² Train", f"{r2_train:.4f}")
        
        with col2:
            st.metric("R² Test", f"{r2_test:.4f}")
        
        with col3:
            overfitting = (r2_train - r2_test) / r2_train * 100 if r2_train > 0 else 0
            st.metric("Overfitting %", f"{overfitting:.2f}%")
        
        with col4:
            st.metric("MAE Test", f"{mae_test:.4f}")
        
        with col5:
            st.metric("RMSE Test", f"{rmse_test:.4f}")
        
        # Alerta de overfitting
        if overfitting > 5:
            st.warning(f"⚠️ Overfitting detectado: {overfitting:.2f}% > 5%")
        else:
            st.success(f"✅ Overfitting bajo: {overfitting:.2f}% < 5%")
        
        st.markdown("---")
        
        # Gráfico Predicted vs Actual
        st.subheader("📊 Predicciones vs Valores Reales (Test Set)")
        
        y_test_original = np.expm1(y_test)
        y_pred_test_original = np.expm1(y_pred_test)
        
        fig = go.Figure()
        
        # Scatter de predicciones
        fig.add_trace(go.Scatter(
            x=y_test_original,
            y=y_pred_test_original,
            mode='markers',
            name='Predicciones',
            marker=dict(size=5, opacity=0.5, color='blue')
        ))
        
        # Línea ideal y=x
        min_val = min(y_test_original.min(), y_pred_test_original.min())
        max_val = max(y_test_original.max(), y_pred_test_original.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Ideal (y=x)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Predicciones vs Valores Reales',
            xaxis_title='Precio Real ($)',
            yaxis_title='Precio Predicho ($)',
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Análisis de Residuos
        st.subheader("📉 Análisis de Residuos")
        
        residuals = y_test_original - y_pred_test_original
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de residuos
            fig = px.histogram(residuals, nbins=50,
                            title='Distribución de Residuos',
                            labels={'value': 'Residuo ($)', 'count': 'Frecuencia'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuos vs Predichos
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred_test_original,
                y=residuals,
                mode='markers',
                marker=dict(size=5, opacity=0.5, color='green'),
                name='Residuos'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title='Residuos vs Predicciones',
                xaxis_title='Precio Predicho ($)',
                yaxis_title='Residuo ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas de residuos
        st.markdown("---")
        st.subheader("📋 Estadísticas de Residuos")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Media", f"${residuals.mean():,.2f}")
        
        with col2:
            st.metric("Std Dev", f"${residuals.std():,.2f}")
        
        with col3:
            st.metric("Min", f"${residuals.min():,.2f}")
        
        with col4:
            st.metric("Max", f"${residuals.max():,.2f}")
        
        st.markdown("---")
        
        # Feature Importance
        st.subheader("⭐ Importancia de Variables")
        
        # Usar los nombres de features guardados
        feature_names = X_feature_names
        feature_importance = model.feature_importances_
        
        # Crear DataFrame para visualizar
        importance_df = pd.DataFrame({
            'Variable': feature_names,
            'Importancia': feature_importance
        }).sort_values('Importancia', ascending=True).tail(15)  # Top 15
        
        # Gráfico de importancia
        fig = px.bar(importance_df, 
                    x='Importancia', 
                    y='Variable',
                    orientation='h',
                    title='Top 15 Variables Más Importantes',
                    labels={'Importancia': 'Importancia (%)', 'Variable': 'Variable'},
                    color='Importancia',
                    color_continuous_scale='Viridis')
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.markdown("---")
        st.subheader("📊 Detalles de Importancia")
        
        detailed_importance = pd.DataFrame({
            'Variable': feature_names,
            'Importancia': feature_importance,
            'Importancia %': (feature_importance / feature_importance.sum() * 100).round(2)
        }).sort_values('Importancia', ascending=False)
        
        # Mostrar top 10
        st.dataframe(detailed_importance.head(10), use_container_width=True)
        
        # Insight
        top_feature = detailed_importance.iloc[0]
        st.info(f"💡 **Variable más importante:** {top_feature['Variable']} ({top_feature['Importancia %']:.1f}% de la importancia total)")

if __name__ == "__main__":
    main()
