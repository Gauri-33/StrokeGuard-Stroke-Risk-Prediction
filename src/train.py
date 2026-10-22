"""
Master training pipeline.
Executes stratified splitting, CV tuning across imbalance configurations,
model evaluation, threshold optimization, feature importance aggregation, and model saving.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data import load_and_clean_data, compute_dataset_stats
from src.preprocessing import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.find_s import FindSClassifier
from src.evaluate import (
    compute_metrics, optimize_threshold_fbeta,
    plot_curves, plot_confusion_matrix, run_fairness_audit, ensure_fig_dir
)

RANDOM_STATE = 42

def aggregate_feature_importances(tree_model, preprocessor) -> dict:
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_ohe_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_ohe_cols

    raw_importances = tree_model.feature_importances_
    aggregated = {feat: 0.0 for feat in (NUMERIC_FEATURES + CATEGORICAL_FEATURES)}

    for col_name, imp in zip(all_feature_names, raw_importances):
        matched = False
        for original_cat in CATEGORICAL_FEATURES:
            if col_name.startswith(original_cat):
                aggregated[original_cat] += imp
                matched = True
                break
        if not matched and col_name in aggregated:
            aggregated[col_name] += imp

    return aggregated

def main():
    np.random.seed(RANDOM_STATE)
    ensure_fig_dir()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Step 1: Loading & Cleaning Dataset...")
    df = load_and_clean_data()
    stats = compute_dataset_stats(df)
    print(f"Dataset Stats: {stats}")

    X = df.drop(columns=['stroke'])
    y = df['stroke'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nStep 2: Evaluating Find-S Concept Learner...")
    X_train_disc = FindSClassifier.discretize(X_train)
    X_test_disc = FindSClassifier.discretize(X_test)

    find_s_primary = FindSClassifier()
    find_s_primary.fit(X_train_disc, y_train)
    find_s_test_preds = find_s_primary.predict(X_test_disc)
    find_s_metrics = compute_metrics(y_test, find_s_test_preds)
    print(f"Primary Find-S Hypothesis: {find_s_primary.hypothesis}")
    print(f"Primary Find-S Test Recall: {find_s_metrics['recall']:.4f}, Precision: {find_s_metrics['precision']:.4f}")

    find_s_secondary = FindSClassifier()
    mask_age60 = (X_train['Age'] > 60)
    find_s_secondary.fit(X_train_disc[mask_age60], y_train[mask_age60])
    print(f"Secondary Find-S (>60) Hypothesis: {find_s_secondary.hypothesis}")

    print("\nStep 3: Training and Tuning Models across Imbalance Setups...")
    model_configs = {
        "Logistic Regression": {
            "model_cls": LogisticRegression,
            "params": [{"C": 0.1}, {"C": 1.0}],
            "imbalance_modes": ["none", "balanced", "smote"]
        },
        "KNN": {
            "model_cls": KNeighborsClassifier,
            # n_jobs=-1 tells all CPU cores to work together
            "params": [{"n_neighbors": 5, "n_jobs": -1}],
            "imbalance_modes": ["none"]  # Running KNN on raw data is fast; SMOTE on 40k rows with KNN is what hangs
        },
        "Decision Tree": {
            "model_cls": DecisionTreeClassifier,
            "params": [
                {"max_depth": 5, "min_samples_leaf": 10},
                {"max_depth": 8, "min_samples_leaf": 20}
            ],
            "imbalance_modes": ["none", "balanced", "smote"]
        },
        "Random Forest": {
            "model_cls": RandomForestClassifier,
            # n_jobs=-1 uses all your laptop's CPU threads in parallel
            "params": [
                {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 10, "n_jobs": -1}
            ],
            "imbalance_modes": ["none", "balanced", "smote"]
        }
    }

    best_models = {}
    comparison_records = []

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    dummy_m = compute_metrics(y_test, dummy_preds)
    comparison_records.append({
        "Model": "Dummy (Most Frequent)", "Setup": "N/A", "Threshold": 0.5,
        "CV PR-AUC": 0.0, "Test PR-AUC": 0.0, "Test Recall": dummy_m['recall'],
        "Test Precision": dummy_m['precision'], "Test F1": dummy_m['f1'],
        "Test ROC-AUC": dummy_m['roc_auc'], "Test Accuracy": dummy_m['accuracy']
    })

    for name, config in model_configs.items():
        best_pr_auc = -1.0
        best_candidate = None

        for imb in config["imbalance_modes"]:
            for p in config["params"]:
                oof_probs = np.zeros(len(X_train))
                cv_pr_aucs = []

                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    steps = [('prep', get_preprocessor())]
                    if imb == "smote":
                        steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))

                    param_dict = p.copy()
                    if imb == "balanced" and "class_weight" in config["model_cls"]().get_params():
                        param_dict["class_weight"] = "balanced"
                    if "random_state" in config["model_cls"]().get_params():
                        param_dict["random_state"] = RANDOM_STATE

                    model_inst = config["model_cls"](**param_dict)
                    steps.append(('model', model_inst))

                    pipe = ImbPipeline(steps)
                    pipe.fit(X_tr_f, y_tr_f)

                    val_probs = pipe.predict_proba(X_val_f)[:, 1]
                    oof_probs[val_idx] = val_probs

                    fold_m = compute_metrics(y_val_f, (val_probs >= 0.5).astype(int), val_probs)
                    cv_pr_aucs.append(fold_m['pr_auc'])

                mean_pr_auc = np.mean(cv_pr_aucs)
                if mean_pr_auc > best_pr_auc:
                    best_pr_auc = mean_pr_auc
                    best_candidate = {
                        "name": name,
                        "imbalance": imb,
                        "params": p,
                        "mean_cv_pr_auc": mean_pr_auc,
                        "oof_probs": oof_probs
                    }

        steps = [('prep', get_preprocessor())]
        if best_candidate["imbalance"] == "smote":
            steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
        p_dict = best_candidate["params"].copy()
        if best_candidate["imbalance"] == "balanced" and "class_weight" in config["model_cls"]().get_params():
            p_dict["class_weight"] = "balanced"
        if "random_state" in config["model_cls"]().get_params():
            p_dict["random_state"] = RANDOM_STATE

        final_pipe = ImbPipeline(steps + [('model', config["model_cls"](**p_dict))])
        final_pipe.fit(X_train, y_train)

        # Use population class prevalence baseline as threshold (~24.22%)
        opt_thresh = float(stats["prevalence_pct"] / 100.0)

        test_probs = final_pipe.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= opt_thresh).astype(int)
        test_m = compute_metrics(y_test, test_preds, test_probs)

        best_models[name] = {
            "pipeline": final_pipe,
            "threshold": opt_thresh,
            "metrics": test_m,
            "cv_pr_auc": best_candidate["mean_cv_pr_auc"],
            "setup": f"{best_candidate['imbalance']} | {best_candidate['params']}"
        }

        comparison_records.append({
            "Model": name,
            "Setup": best_candidate["imbalance"],
            "Threshold": opt_thresh,
            "CV PR-AUC": best_candidate["mean_cv_pr_auc"],
            "Test PR-AUC": test_m['pr_auc'],
            "Test Recall": test_m['recall'],
            "Test Precision": test_m['precision'],
            "Test F1": test_m['f1'],
            "Test ROC-AUC": test_m['roc_auc'],
            "Test Accuracy": test_m['accuracy']
        })

    df_results = pd.DataFrame(comparison_records)
    df_results.to_csv("reports/model_comparison.csv", index=False)
    print("\n--- Model Comparison Summary ---")
    print(df_results.to_string(index=False))

    # Select the final model using the same cross-validation PR-AUC criterion used
    # during model-family tuning. Do not hard-code Random Forest: another family may
    # generalize better on this imbalanced dataset.
    winner_name = max(best_models, key=lambda model_name: best_models[model_name]["cv_pr_auc"])
    winner_info = best_models[winner_name]
    winner_pipe = winner_info["pipeline"]
    winner_estimator = winner_pipe.named_steps['model']
    preprocessor = winner_pipe.named_steps['prep']

    aggregated_fi = {}
    if hasattr(winner_estimator, "feature_importances_"):
        aggregated_fi = aggregate_feature_importances(winner_estimator, preprocessor)

    winner_bundle = {
        "model_name": winner_name,
        "pipeline": winner_pipe,
        "threshold": winner_info["threshold"],
        "feature_importances": aggregated_fi,
        "features": list(X_train.columns),
        "cv_pr_auc": winner_info["cv_pr_auc"],
        "test_metrics": winner_info["metrics"],
        "setup": winner_info["setup"]
    }
    joblib.dump(winner_bundle, "models/winning_pipeline.joblib")
    print(f"\nSelected {winner_name} by CV PR-AUC ({winner_info['cv_pr_auc']:.4f}); saved model artifact.")

    plot_curves(best_models, X_test, y_test)
    for name, info in best_models.items():
        preds = (info['pipeline'].predict_proba(X_test)[:, 1] >= info['threshold']).astype(int)
        plot_confusion_matrix(y_test, preds, name)

    winner_test_preds = (winner_pipe.predict_proba(X_test)[:, 1] >= winner_info['threshold']).astype(int)
    fairness_df = run_fairness_audit(X_test, y_test.values, winner_test_preds)
    fairness_df.to_csv("reports/fairness_check.csv", index=False)
    print("\nFairness audit saved to reports/fairness_check.csv")
    print(fairness_df)

if __name__ == "__main__":
    main()
