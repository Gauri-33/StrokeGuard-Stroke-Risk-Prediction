"""
Find-S concept learning implementation and discretization helpers.
"""
import numpy as np
import pandas as pd

class FindSClassifier:
    """
    Find-S specific-to-general concept learner.
    Learns a conjunctive hypothesis from positive training instances.
    """
    def __init__(self):
        self.hypothesis = None
        self.feature_names = None

    @staticmethod
    def discretize(df: pd.DataFrame) -> pd.DataFrame:
        disc = pd.DataFrame(index=df.index)
        disc['age'] = np.where(df['age'] <= 60, '<=60', '>60')
        disc['hypertension'] = np.where(df['hypertension'] == 1, 'Yes', 'No')
        disc['heart_disease'] = np.where(df['heart_disease'] == 1, 'Yes', 'No')
        disc['glucose'] = np.where(df['avg_glucose_level'] <= 200, '<=200', '>200')

        bmi_val = df['bmi'].fillna(df['bmi'].median())
        conditions = [
            (bmi_val < 18.5),
            (bmi_val >= 18.5) & (bmi_val < 25.0),
            (bmi_val >= 25.0) & (bmi_val < 30.0),
            (bmi_val >= 30.0)
        ]
        labels = ['underweight', 'normal', 'overweight', 'obese']
        disc['bmi_band'] = np.select(conditions, labels, default='normal')
        disc['smoking_status'] = df['smoking_status'].fillna('Unknown')
        return disc

    def fit(self, X_disc: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X_disc.columns)
        positives = X_disc[y == 1]

        if len(positives) == 0:
            self.hypothesis = ['?'] * len(self.feature_names)
            return self

        h = list(positives.iloc[0].values)
        for _, row in positives.iloc[1:].iterrows():
            for i, val in enumerate(row.values):
                if h[i] != '?' and h[i] != val:
                    h[i] = '?'
        self.hypothesis = h
        return self

    def predict(self, X_disc: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X_disc.iterrows():
            match = True
            for i, val in enumerate(row.values):
                if self.hypothesis[i] != '?' and self.hypothesis[i] != val:
                    match = False
                    break
            preds.append(1 if match else 0)
        return np.array(preds)
