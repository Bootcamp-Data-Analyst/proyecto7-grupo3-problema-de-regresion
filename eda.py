import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(file_path):
    df = pd.read_csv(file_path)
    os.makedirs('static', exist_ok=True)
    
    # 1. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Distribución de Precios de Vehículos')
    plt.xlabel('Precio')
    plt.ylabel('Frecuencia')
    plt.savefig('static/price_dist.png')
    plt.close()
    
    # 2. Correlation Matrix
    # Pre-clean for corr
    from preprocessing import preprocess_data
    df_clean, _ = preprocess_data(df)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_clean.corr(), annot=False, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.savefig('static/correlation.png')
    plt.close()
    
    # 3. Price vs Milage
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='milage', y='price', alpha=0.5)
    plt.title('Relación Precio vs Kilometraje')
    plt.savefig('static/price_vs_milage.png')
    plt.close()
    
    # 4. Brand vs Price (Top 10)
    top_brands = df['brand'].value_counts().nlargest(10).index
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[df['brand'].isin(top_brands)], x='brand', y='price')
    plt.xticks(rotation=45)
    plt.title('Precio por Marca (Top 10)')
    plt.savefig('static/brand_vs_price.png')
    plt.close()

if __name__ == "__main__":
    run_eda('train.csv')
    print("EDA plots saved in static/")
