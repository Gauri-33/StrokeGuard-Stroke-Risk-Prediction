"""
Evaluation utilities: PR-AUC, ROC-AUC, threshold optimization (F2), curves, and fairness audits.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix
)

FIG_DIR = "reports/figures"

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred, y_probs=None) -> dict:
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    pr_auc = np.nan
    roc_auc = np.nan
    if y_probs is not None:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(r_curve, p_curve)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

    return {
        "recall": rec,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": prec,
        "roc_auc": roc_auc,
        "accuracy": acc
    }

def optimize_threshold_fbeta(y_true, y_probs, beta: float = 2.0):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        denom = (beta**2 * p) + r
        f = ((1 + beta**2) * p * r) / denom if denom > 0 else 0.0
        f_scores.append(f)
    best_idx = np.argmax(f_scores)
    return thresholds[best_idx], f_scores[best_idx]

def plot_curves(models_dict, X_test, y_test):
    ensure_fig_dir()
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))

    for name, model_info in models_dict.items():
        pipeline = model_info['pipeline']
        if hasattr(pipeline, "predict_proba"):
            y_probs = pipeline.predict_proba(X_test)[:, 1]
            p, r, _ = precision_recall_curve(y_test, y_probs)
            ax_pr.plot(r, p, label=f"{name} (AUC={auc(r, p):.3f})")

            fpr, tpr, _ = roc_curve(y_test, y_probs)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax_pr.set_title("Precision-Recall Curves (Test Set)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="best")
    ax_pr.grid(True)

    ax_roc.set_title("ROC Curves (Test Set)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="best")
    ax_roc.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_pr_curves.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    ensure_fig_dir()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

def run_fairness_audit(df_test: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    records = []
    gender_col = 'Gender' if 'Gender' in df_test.columns else 'gender'
    age_col = 'Age' if 'Age' in df_test.columns else 'age'

    for g in df_test[gender_col].dropna().unique():
        idx = (df_test[gender_col] == g)
        if idx.sum() > 0:
            rec = recall_score(y_true[idx], y_pred[idx], zero_division=0)
            prec = precision_score(y_true[idx], y_pred[idx], zero_division=0)
            records.append({'attribute': 'gender', 'group': g, 'count': int(idx.sum()), 'recall': rec, 'precision': prec})

    age_bands = pd.cut(df_test[age_col], bins=[0, 45, 65, 120], labels=['<45', '45-65', '>65'])
    for band in age_bands.cat.categories:
        idx = (age_bands == band)
        if idx.sum() > 0:
            rec = recall_score(y_true[idx], y_pred[idx], zero_division=0)
            prec = precision_score(y_true[idx], y_pred[idx], zero_division=0)
            records.append({'attribute': 'age_band', 'group': str(band), 'count': int(idx.sum()), 'recall': rec, 'precision': prec})

    return pd.DataFrame(records)