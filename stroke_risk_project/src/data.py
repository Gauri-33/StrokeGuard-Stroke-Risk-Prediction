"""
Data acquisition, verification, and cleaning utilities.
"""
import os
import subprocess
import pandas as pd
import numpy as np

KAGGLE_DATASET_ID = "fedesoriano/stroke-prediction-dataset"
DATA_DIR = "data"
CSV_FILENAME = "healthcare-dataset-stroke-data.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

def ensure_dataset_present() -> str:
    """
    Checks if raw dataset exists in data/. If not, attempts download via Kaggle CLI.
    Raises FileNotFoundError if download fails or credentials are not configured.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        return CSV_PATH

    print(f"Dataset not found at {CSV_PATH}. Attempting Kaggle CLI download...")
    try:
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_ID, "-p", DATA_DIR, "--unzip"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr)
    except Exception as e:
        raise FileNotFoundError(
            f"Unable to download Kaggle dataset automatically: {e}. "
            f"Please manually download '{CSV_FILENAME}' from Kaggle "
            f"and place it inside the '{DATA_DIR}/' directory."
        )
    return CSV_PATH

def load_and_clean_data(filepath: str = CSV_PATH) -> pd.DataFrame:
    """
    Cleans raw dataset according to rubric specifications:
    - Drops 'id' column.
    - Removes single row where gender == 'Other'.
    - Coerces 'N/A' or non-numeric bmi values to float NaN.
    """
    if not os.path.exists(filepath):
        filepath = ensure_dataset_present()

    df = pd.read_csv(filepath)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Drop single 'Other' row in gender
    df = df[df['gender'] != 'Other'].reset_index(drop=True)

    # Coerce bmi string "N/A" into real NaN
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

    return df

def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Computes exact empirical statistics for EDA reporting."""
    total_rows = len(df)
    positives = int(df['stroke'].sum())
    prevalence = (positives / total_rows) * 100
    missing_bmi = int(df['bmi'].isna().sum())

    # Outlier counts via IQR
    outlier_counts = {}
    for col in ['age', 'avg_glucose_level', 'bmi']:
        series = df[col].dropna()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lb = q25 - 1.5 * iqr
        ub = q75 + 1.5 * iqr
        outliers = ((series < lb) | (series > ub)).sum()
        outlier_counts[col] = int(outliers)

    return {
        "total_rows": total_rows,
        "positives": positives,
        "prevalence_pct": prevalence,
        "missing_bmi": missing_bmi,
        "outlier_counts": outlier_counts
    }
