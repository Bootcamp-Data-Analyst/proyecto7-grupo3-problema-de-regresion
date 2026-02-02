# Car Price Prediction - Machine Learning Project

**Español:** Resumen y documentación del proyecto para predecir precios de coches. (See English below)

---

## Project Overview
This repository contains a full machine learning workflow to predict car prices from used car listings. It includes:
- Data cleaning and feature engineering
- Exploratory data analysis (EDA)
- Preprocessing pipelines (log transforms, winsorization, encoders)
- Model training and hyperparameter tuning (XGBoost optimized)
- Model evaluation and visualizations
- A Streamlit app for interactive predictions

---

## Key Results
- Final selected model: **XGBoost (optimized)**
- Reported metrics (on log-transformed target):
  - MAE: 0.3056
  - MSE: 0.1648
  - RMSE: 0.4060
  - R²: 0.7050

> Note: Use `np.expm1()` to revert predictions to the original price scale for interpretation.

---

## Dataset
- Raw data files: `data/raw/train.csv`, `data/raw/test.csv`
- Cleaned datasets saved: `data/processed/cleaned_train.csv`, `data/processed/cleaned_test.csv`
- The cleaning notebook extracts features like `horsepower`, `engine_size_L`, `cylinders`, and computes `car_age`.

---

## Preprocessing & Feature Engineering
- Extracted engine features (horsepower, engine size, cylinders)
- Created `car_age` from `model_year`
- Applied `np.log1p` to the target and skewed predictors, and applied winsorization (limits 0.05/0.05) consistently to the same set of variables — **note:** in this version the training pipeline **does winsorize the `price`** (target) along with predictors.
- Target encoding for `brand` is performed using `category_encoders` after the target transformations used in training.
- The current pipeline does **not** create or persist `brand_stats` nor an `is_luxury` flag; only `encoder_brand.pkl` and `column_transformer_preprocessor.pkl` are saved as artifacts.
- Column transformer ensures consistent preprocessing for training and inference


**Quick example (training) — current pipeline:**
```python
# Apply log + winsorize to price and predictors, then target-encode brand
from scipy.stats.mstats import winsorize
import category_encoders as ce
import joblib

# Example (training)
df['price'] = np.log1p(df['price'])
df['price'] = winsorize(df['price'], limits=(0.05, 0.05))
# apply same transforms to predictors (e.g., milage, horsepower, car_age)

encoder_brand = ce.TargetEncoder(cols=['brand'])
df['brand'] = encoder_brand.fit_transform(df[['brand']], df['price'])
joblib.dump(encoder_brand, 'models/encoder_brand.pkl')
```

**Quick inference snippet (test):**
```python
# During inference, transform brand with the saved encoder and apply the ColumnTransformer
# (log/winsorize for inputs should follow the same rules used in training when appropriate)
df_test['brand'] = encoder_brand.transform(df_test[['brand']])
X_test = preprocessor.transform(df_test.drop(columns=['price'], errors='ignore'))
pred_log = model.predict(X_test)
pred_price = np.expm1(pred_log)
```

---

## How to Reproduce (Windows instructions)
1. Clone the repo
2. Create and activate a virtual environment:
   - PowerShell:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
3. Install requirements:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. (If needed) register the kernel:
   ```powershell
   python -m ipykernel install --user --name=carprice-env --display-name "carprice-env"
   ```
5. Run notebooks in VS Code or JupyterLab, selecting the `carprice-env` kernel.

**Windows long-path note:** If you encounter pip install errors due to long paths (OSError while unpacking packages), create a short junction (example):
```powershell
cmd /c mklink /J C:\proj "C:\Users\USUARIO\Desktop\Bootcamp DataAnalytics Andrés Pérez\Modulo3\proyecto7-..."
```
Then operate from `C:\proj` while installing.

---

## Run Streamlit App (local)
```powershell
cd app
streamlit run streamlit_app.py
```
Open the URL printed by Streamlit in your browser.

---

## Notebooks
- `notebooks/Data_Cleaning_and_Transformation.ipynb` — cleaning, imputation, saving `cleaned_*` CSVs
- `notebooks/Modelado_Completo_Precios_Coches.ipynb` — preprocessing, models comparison, hyperparameter optimization, final model training, evaluation, visualizations
- `notebooks/EDA_Andrés.ipynb` — exploratory analyses and visual investigations

---

## Saved Artifacts
- `models/optimized_xgb_model.pkl` — final XGBoost model
- `models/encoder_brand.pkl` — TargetEncoder for `brand`
- `models/column_transformer_preprocessor.pkl` — ColumnTransformer used for OHE and passthrough

---

## Recommendations & Next Steps
- Add SHAP or feature importance visualizations for model explainability
- Create CSV of test predictions and evaluation in original price scale
- Add automated tests for preprocessing functions
- Consider ensembling and model serving (API/CI for re-training)

---

## Contributing
1. Fork the repo
2. Create a feature branch
3. Add tests for new preprocessing or modeling code
4. Open a pull request describing changes

---

## License & Authors
- Authors: Grupo 3 - Bootcamp DataAnalytics (Andrés Pérez and collaborators)
- License: Add your preferred license (e.g., MIT)

---

Thanks for checking this project — final version notes:

- **Final:** This release **does NOT** use an `is_luxury` flag nor persist `brand_stats`. The training pipeline applies log + winsorize to `price` and predictors and persists `encoder_brand.pkl` and `column_transformer_preprocessor.pkl` in `models/`.
- **Regenerate models:** From the repo root run `python app/rebuild_models.py` to retrain and save artifacts.
- **Run the app:** `cd app` then `streamlit run streamlit_app.py`.

If you'd like, I can add a SHAP notebook and a `predictions.csv` export step to the pipeline.