# StrokeGuard — Machine Learning-Based Stroke Prediction (Educational Prototype)

An end-to-end mini project that cleans an imbalanced public dataset, compares a concept-learning baseline and supervised classifiers, selects a decision threshold from out-of-fold predictions, and provides an interactive web interface.

> **Safety disclaimer:** This is an academic prototype, not a medical device. Predictions are not diagnoses or clinically validated individual-risk estimates. Do not use them for treatment or health decisions.

## Project structure

```text
stroke_risk_project/
├── app/
│   ├── app.py                 # Streamlit responsive dashboard
│   ├── main.py                # Optional FastAPI service
│   └── templates/index.html   # FastAPI web interface
├── data/                      # Place the Kaggle CSV here (not committed)
├── models/                    # Trained pipeline artifact
├── reports/                   # Model metrics, fairness audit and figures
├── src/                       # Data, preprocessing, training, evaluation
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

- Source: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Expected file: `data/healthcare-dataset-stroke-data.csv`
- The raw dataset is intentionally excluded from Git history. Download it from the source and place it in `data/`.
- The training cleaner removes the identifier, drops the single `gender == 'Other'` row, and converts BMI to numeric. Missing BMI values are handled inside the model pipeline.

## Run locally

Use Python 3.11 or 3.12 for a straightforward setup. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
streamlit run app/app.py
```

The training script writes `reports/model_comparison.csv`, plots, a fairness audit, and `models/winning_pipeline.joblib`. The selected model is chosen from the evaluated model families using cross-validation PR-AUC; it is not hard-coded to Random Forest.

## Optional FastAPI interface

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The interactive Streamlit dashboard is the primary student-facing UI and includes risk-assessment, model-analytics, and project-overview sections.

## Methodology

- Stratified 80/20 train/test split with a fixed random seed.
- Five-fold stratified cross-validation for model-family and imbalance-strategy comparison.
- Numeric median imputation, IQR-based outlier capping and scaling inside the pipeline.
- Categorical mode imputation and one-hot encoding inside the pipeline.
- Comparison of Dummy Classifier, Logistic Regression, KNN, Decision Tree and Random Forest.
- Baseline, class-weighted and SMOTE strategies where supported. SMOTE is kept inside the training pipeline.
- Decision threshold selected from out-of-fold training predictions using F2-score.
- PR-AUC, ROC-AUC, recall, precision, F1 and accuracy reported; PR-AUC is emphasized due to class imbalance.
- Group-wise recall and precision checks included as an initial fairness diagnostic, not proof of fairness.

## Deployment notes

The Streamlit interface is responsive for desktop and mobile browser widths. For Streamlit Community Cloud, push the repository, set the app entry point to `app/app.py`, and ensure the trained model artifact is included in the repository. For container deployment, use the included Dockerfile. Never publish real patient data or credentials.

## Important limitations

The dataset has a low positive-class prevalence and only a limited number of stroke-positive examples. A model probability is not necessarily calibrated. The test metrics are split-specific, may vary with random seed, and do not establish clinical validity. Independent external validation, calibration analysis, and qualified clinical review would be required before any real-world use.
