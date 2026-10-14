# HeartPredict: Heart Disease Prediction Using Machine Learning

Mini Capstone Project , Sol Plaatje University.
Student: Sithelile Mlaba 

HeartPredict estimates a person's risk of heart disease from 13 routine clinical measurements and explains each result in plain English. It was built as a 10-notebook machine learning pipeline plus a Streamlit web application.

> **Disclaimer:** This is an educational proof-of-concept. It is not a medical device and does not replace a doctor's diagnosis.

## What is in this repository

| File | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Merge, clean and remove duplicates from the raw dataset |
| `02_eda.ipynb` | Exploratory data analysis |
| `03_key_findings_visuals.ipynb` | Summary charts of the main findings |
| `04_preprocessing.ipynb` | Scaling and the stratified 80/20 train/test split |
| `05_unsupervised_clustering.ipynb` | K-Means clustering with PCA |
| `06_model_training.ipynb` | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| `07_hyperparameter_tuning.ipynb` | Bayesian optimisation (Optuna) |
| `08_final_evaluation.ipynb` | Accuracy, precision, recall, F1, AUC-ROC, confusion matrices |
| `09_shap_explainability.ipynb` | SHAP explanations |
| `10_save_final_models.ipynb` | Saves the four models and `model_metadata.json` |
| `app.py` | The Streamlit web application |
| `*.joblib`, `model_metadata.json` | Saved models, scaler and metadata used by the app |
| `heart_disease_raw.xlsx`, `heart_disease_cleaned.csv`, `X_*.csv`, `y_*.csv` | Data files |

## Dataset

A heart disease dataset from Kaggle, merged from five public datasets. After cleaning, 894 unique patient records and 13 clinical features remain.

## How to run

1. Install Python 3.10 or newer.
2. Install the required packages:
   ```
   pip install streamlit plotly shap xgboost reportlab scikit-learn pandas numpy matplotlib joblib optuna jupyter
   ```
3. To rebuild the models, run notebooks `01` to `10` in order (Kernel > Restart & Run All).
4. To start the web application, run this from the project folder:
   ```
   streamlit run app.py
   ```
5. Create an account on the sign-up tab (patient or healthcare worker) and log in. A local database file, `heart_app.db`, is created automatically on first run.

## Main results

On the held-out test set (179 patients), XGBoost was the best overall model (accuracy 76.5%, AUC-ROC 0.846), with small gaps to the other models. See notebook `08` for the full comparison.

## Methodology

CRISP-DM for the machine learning pipeline, and an SDLC approach for the web application.
