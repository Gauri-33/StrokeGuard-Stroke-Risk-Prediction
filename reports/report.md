# Machine Learning-Based Stroke Risk Prediction and Comparative Model Analysis

## Abstract
Stroke represents a leading cause of long-term disability and mortality worldwide. In this educational project, we design an end-to-end machine learning pipeline to predict individual stroke occurrence from demographic, physiological, and lifestyle attributes using the Kaggle Stroke Prediction Dataset (~5,100 records, ~5% positive class prevalence). We implement and contrast Concept Learning (Find-S) with four standard machine learning architectures (Logistic Regression, K-Nearest Neighbors, Decision Trees, and Random Forests) alongside a zero-information baseline (Dummy Classifier). Data preprocessing strictly eliminates leakage by encapsulating median imputation, custom 1.5 x IQR Winsorization, standard scaling, and one-hot encoding within cross-validation folds. Class imbalance is systematically tackled across unweighted, class-weighted, and SMOTE regimes. Because clinical utility demands minimizing false negatives in low-prevalence screening, decision thresholds are optimized via out-of-fold F2-score rather than arbitrary 0.5 cutoffs. The final model is selected by cross-validation PR-AUC across the evaluated model families; the selected family can vary by dataset split and training run. Finally, the model is packaged into a responsive Streamlit dashboard with model analytics and an optional FastAPI interface.

## Introduction
Early identification of stroke risk factors enables timely preventive lifestyle and clinical interventions. However, automated risk prediction poses severe statistical hurdles:
1. Severe Imbalance: Only ~5% of patients suffer a stroke in the dataset, rendering raw classification accuracy misleading.
2. Data Leakage Risks: Preprocessing transformations (imputation medians, outlier bounds, scaling statistics) that span the entire dataset corrupt cross-validation boundaries.
3. Clinical Asymmetry: A false negative (missing an individual at risk of stroke) carries much higher clinical consequences than a false positive (recommending additional diagnostic monitoring).

This study systematically addresses each challenge, following a strict four-part rubric spanning data cleaning, model comparison, deployment, and code quality.

## Dataset Description & Licence
- Source: Kaggle Stroke Prediction Dataset (fedesoriano/stroke-prediction-dataset).
- Licence: Open Database, Public Domain (CC0: Public Domain / Open Data).
- Exact Statistics:
  - Total records: 5,109 (after dropping 1 row with gender == 'Other').
  - Positives (Stroke = 1): 249 (~4.87% prevalence).
  - Missing values: bmi has 201 missing values (~3.93%); all other columns are complete.
  - Outliers: Identified in avg_glucose_level and bmi via the 1.5 x IQR standard.

## EDA and Zero-Leakage Preprocessing
To adhere strictly to statistical discipline:
- An 80/20 stratified train/test split was established prior to any transformation.
- A 5-fold Stratified Cross-Validation was conducted strictly on the training partition.
- Custom IQR Winsorization Transformer: Extreme outliers in glucose and BMI are not deleted; instead, 1.5 x IQR bounds are computed on training folds and values are clipped.
- Categorical features are imputed with the most frequent value and one-hot encoded, with singular reference categories dropped to eliminate collinearity.

## Find-S Concept Learning Method & Findings
Find-S was applied by discretizing features:
- Age: <= 60 / > 60
- Hypertension: Yes / No
- Heart Disease: Yes / No
- Glucose: <= 200 / > 200
- BMI: WHO bands (Underweight, Normal, Overweight, Obese)
- Smoking Status: Categorical

### Primary Run
Trained on all positive stroke examples. Because stroke occurs across patients of different ages, BMI classifications, and smoking statuses, the hypothesis rapidly generalizes every attribute to '?' (the maximally general rule ['?', '?', '?', '?', '?', '?']).
- Limitation: Find-S considers only positive examples, assumes zero label noise, cannot handle disjunctive concepts, and outputs a binary match without calibrated probability.

### Secondary Run (Age > 60)
When restricted to patients older than 60, Find-S retains the constraint Age > 60, highlighting how cohort stratification can isolate local conjunctive patterns.

## Models, Tuning, and Imbalance Handling
We evaluated five algorithmic families:
1. Dummy Classifier: Predicts the most frequent class (0), demonstrating ~95% accuracy with 0.0% recall.
2. Logistic Regression: Regularization parameter C in {0.01, 0.1, 1.0, 10.0}.
3. K-Nearest Neighbors (KNN): k in {3, 5, 9, 15}, utilizing standard-scaled distances.
4. Decision Tree: Tree depth constrained to prevent overfitting (depth in {3, 5, 7}).
5. Random Forest: Ensemble of decorrelated trees (n in {100, 150}).

### Imbalance Regimes
Each algorithm was tested across three setups:
- Baseline (No adjustment).
- Cost-sensitive learning (class_weight='balanced').
- Synthetic Minority Over-sampling Technique (SMOTE), encapsulated entirely within imblearn.pipeline.Pipeline to guarantee oversampled instances never leak into validation folds.

### Decision Threshold Optimization
Because standard 0.5 probability thresholds fail under 5% prevalence, optimal thresholds were chosen by maximizing the F2-score across out-of-fold training predictions.

## Deployment & Limitations
The selected model pipeline and threshold are packaged with joblib and served via the Streamlit application (app/app.py), with an optional FastAPI service (app/main.py). The dashboard includes validated inputs, model-output probability, threshold comparison, model-comparison analytics, global feature-importance display when supported, and a prominent educational-use disclaimer. The probability must not be interpreted as a clinically validated personal risk estimate.


## Limitations and Responsible Use

The model is trained on a single public dataset and is not externally validated. The low prevalence of positive cases makes precision and recall sensitive to threshold choice. A high recall can come with many false positives; both recall and precision should be discussed. Group-level metrics are exploratory and can be unstable when subgroup positive counts are small. Feature importance describes model reliance, not causation. The interface is an educational demonstration only and must not be used to diagnose, triage, or guide treatment.
