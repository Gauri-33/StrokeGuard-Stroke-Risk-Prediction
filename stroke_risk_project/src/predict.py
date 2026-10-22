"""
Inference runner for new samples using saved pipeline artifacts.
"""
import joblib
import pandas as pd

def predict_single_instance(sample_dict: dict, artifact_path: str = "models/winning_pipeline.joblib") -> dict:
    artifacts = joblib.load(artifact_path)
    pipeline = artifacts['pipeline']
    threshold = artifacts['threshold']

    df_sample = pd.DataFrame([sample_dict])
    prob = pipeline.predict_proba(df_sample)[0][1]
    is_high_risk = bool(prob >= threshold)

    return {
        "predicted_stroke_probability": float(prob),
        "threshold": float(threshold),
        "is_high_risk": is_high_risk
    }
