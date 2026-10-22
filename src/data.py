"""
Data acquisition, verification, and cleaning utilities.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
CSV_FILENAME = "stroke_risk_prediction_dataset.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

def ensure_dataset_present() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}. Please place '{CSV_FILENAME}' into '{DATA_DIR}/'."
        )
    return CSV_PATH

def load_and_clean_data(filepath: str = CSV_PATH) -> pd.DataFrame:
    """
    Cleans raw dataset according to rubric specifications:
    - Drops ID column.
    - Defines binary stroke target from Stroke_Risk.
    - Drops leakage columns (Stroke_Risk_Score, AI_Health_Recommendation, Doctor_Consultation_Needed).
    """
    if not os.path.exists(filepath):
        filepath = ensure_dataset_present()

    df = pd.read_csv(filepath)

    if 'Patient_ID' in df.columns:
        df = df.drop(columns=['Patient_ID'])
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Define binary target for stroke risk screening
    if 'stroke' not in df.columns and 'Stroke_Risk' in df.columns:
        df['stroke'] = (df['Stroke_Risk'] == 'High').astype(int)

    # Remove leakage columns
    leakage = ['Stroke_Risk', 'Stroke_Risk_Score', 'AI_Health_Recommendation', 'Doctor_Consultation_Needed']
    df = df.drop(columns=[c for c in leakage if c in df.columns])

    # Standardize data types
    numeric_candidates = [
        'Age', 'Height_cm', 'Weight_kg', 'BMI', 'Blood_Pressure_Systolic',
        'Blood_Pressure_Diastolic', 'Heart_Rate', 'Blood_Glucose', 'HbA1c',
        'Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'Sleep_Hours',
        'Exercise_Hours_Per_Week', 'Daily_Walking_Minutes'
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df

def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Computes exact empirical statistics for EDA reporting."""
    total_rows = len(df)
    positives = int(df['stroke'].sum())
    prevalence = (positives / total_rows) * 100

    outlier_counts = {}
    key_numerics = ['Age', 'Blood_Pressure_Systolic', 'Blood_Glucose', 'BMI', 'Total_Cholesterol']
    for col in key_numerics:
        if col in df.columns:
            series = df[col].dropna()
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lb = q25 - 1.5 * iqr
            ub = q75 + 1.5 * iqr
            outlier_counts[col] = int(((series < lb) | (series > ub)).sum())

    return {
        "total_rows": total_rows,
        "positives": positives,
        "prevalence_pct": prevalence,
        "missing_records": int(df.isna().any(axis=1).sum()),
        "outlier_counts": outlier_counts
    }