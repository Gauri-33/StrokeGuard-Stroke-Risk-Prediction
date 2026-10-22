"""
Transformers and pipeline builders ensuring strict zero data leakage.
All statistics (medians, IQR fences, scalers, encoders) are fit on training folds only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    'Age', 'Height_cm', 'Weight_kg', 'BMI', 'Blood_Pressure_Systolic',
    'Blood_Pressure_Diastolic', 'Heart_Rate', 'Blood_Glucose', 'HbA1c',
    'Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'Sleep_Hours',
    'Exercise_Hours_Per_Week', 'Daily_Walking_Minutes'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Country', 'Smoking_Status', 'Alcohol_Consumption',
    'Physical_Activity_Level', 'Stress_Level', 'Diet_Quality',
    'Family_History_Stroke', 'Family_History_Heart_Disease', 'Diabetes',
    'Hypertension', 'Heart_Disease', 'Previous_TIA', 'Atrial_Fibrillation',
    'Chronic_Kidney_Disease', 'Medication_Adherence', 'Work_Type',
    'Residence_Type', 'Air_Pollution_Exposure'
]

class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Winsorization transformer capping outliers at 1.5 * IQR bounds.
    Fences are learned exclusively during fit() on the training folds.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in X_df.columns:
            valid_vals = X_df[col].dropna()
            if len(valid_vals) > 0:
                q25 = np.percentile(valid_vals, 25)
                q75 = np.percentile(valid_vals, 75)
                iqr = q75 - q25
                self.lower_bounds_[col] = q25 - self.factor * iqr
                self.upper_bounds_[col] = q75 + self.factor * iqr
            else:
                self.lower_bounds_[col] = -np.inf
                self.upper_bounds_[col] = np.inf
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.lower_bounds_:
                lb = self.lower_bounds_[col]
                ub = self.upper_bounds_[col]
                X_df[col] = np.clip(X_df[col], lb, ub)
        return X_df.values

def get_preprocessor(numeric_cols=NUMERIC_FEATURES, categorical_cols=CATEGORICAL_FEATURES):
    """
    Builds the ColumnTransformer containing imputation, IQR capping, scaling, and OHE.
    """
    num_pipeline = Pipeline([
        ('median_imputer', SimpleImputer(strategy='median')),
        ('iqr_capper', IQRCapper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('mode_imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )
    return preprocessor