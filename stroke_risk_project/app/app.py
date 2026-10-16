"""StrokeGuard — responsive educational stroke-risk ML dashboard."""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import IQRCapper  # noqa: F401 — needed to load the saved pipeline

st.set_page_config(
    page_title="StrokeGuard | Risk Insights",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design system: deep ink, clean cards, teal accent, accessible contrast.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#122033; --muted:#65758b; --teal:#087f8c; --teal2:#0ea5a4; --line:#e5edf2; --paper:#f4f8fb; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f4f8fb 0%,#f8fbfd 52%,#eef5f8 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1220px; padding-top:1.3rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] { background:#10243a; border-right:1px solid #20394f; }
section[data-testid="stSidebar"] * { color:#eaf3f7 !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b8cbd7 !important; }
.hero { position:relative; overflow:hidden; border-radius:24px; padding:30px 34px; margin-bottom:22px; color:white; background:radial-gradient(circle at 85% 15%,rgba(42,195,184,.28),transparent 28%),linear-gradient(120deg,#10263d 0%,#153c52 60%,#087f8c 100%); box-shadow:0 18px 40px rgba(19,54,77,.13); }
.hero:after { content:'+'; position:absolute; right:7%; top:-55px; font-size:230px; line-height:1; color:rgba(255,255,255,.045); font-weight:800; }
.eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; color:#a9e9e5; }
.hero h1 { font-family:Manrope,sans-serif; font-size:clamp(1.8rem,3.5vw,2.65rem); line-height:1.15; margin:.55rem 0 .65rem; color:white; letter-spacing:-.04em; }
.hero p { max-width:720px; margin:0; color:#d1e3eb; font-size:1rem; }
.hero-chip { display:inline-flex; margin-top:17px; padding:7px 11px; border-radius:999px; border:1px solid rgba(255,255,255,.22); background:rgba(255,255,255,.08); font-size:.78rem; color:#e6fbfa; }
.metric-card { background:white; border:1px solid var(--line); border-radius:17px; padding:18px 19px; box-shadow:0 7px 20px rgba(22,53,74,.035); min-height:112px; }
.metric-label { color:#718196; font-size:.78rem; font-weight:600; letter-spacing:.035em; }
.metric-value { font-family:Manrope,sans-serif; font-weight:800; font-size:1.75rem; color:#17354b; margin-top:5px; letter-spacing:-.04em; }
.metric-note { color:#8493a4; font-size:.75rem; margin-top:2px; }
.section-title { font-family:Manrope,sans-serif; font-weight:800; font-size:1.25rem; color:#17354b; margin:.35rem 0 .2rem; letter-spacing:-.025em; }
.section-sub { color:#718196; font-size:.88rem; margin-bottom:1rem; }
.panel { background:white; border:1px solid var(--line); border-radius:18px; padding:22px; box-shadow:0 7px 20px rgba(22,53,74,.035); }
.notice { border-radius:13px; border:1px solid #f0d9a8; background:#fff9eb; color:#76541c; padding:13px 16px; font-size:.82rem; line-height:1.55; margin:12px 0 22px; }
.stButton>button, .stFormSubmitButton>button { border:0; border-radius:11px; padding:.72rem 1.2rem; font-weight:700; background:linear-gradient(100deg,#087f8c,#0b9b9c); color:white; box-shadow:0 7px 16px rgba(8,127,140,.18); }
.stButton>button:hover, .stFormSubmitButton>button:hover { border:0; background:#086c78; color:white; }
[data-testid="stForm"] { background:white; border:1px solid var(--line); border-radius:18px; padding:20px 22px 14px; box-shadow:0 7px 20px rgba(22,53,74,.035); }
[data-testid="stTabs"] button { font-weight:700; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius:10px; }
footer { visibility:hidden; }
@media(max-width:700px) { .hero{padding:24px 20px}.panel{padding:16px}.metric-value{font-size:1.45rem} }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    path = os.path.join(ROOT_DIR, "models", "winning_pipeline.joblib")
    return joblib.load(path)

@st.cache_data
def load_comparison():
    path = os.path.join(ROOT_DIR, "reports", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

try:
    artifacts = load_artifacts()
    pipeline = artifacts["pipeline"]
    threshold = float(artifacts.get("threshold", 0.5))
    feature_importances = artifacts.get("feature_importances", {}) or {}
    model_name = artifacts.get("model_name", "Selected classifier")
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("From the project root, run `python -m src.train` and restart the app.")
    st.stop()

# Sidebar: project identity and navigation.
with st.sidebar:
    st.markdown("<div style='font-size:2rem;margin:5px 0 8px'>🫀</div><div style='font-size:1.18rem;font-weight:800;font-family:Manrope,sans-serif'>StrokeGuard</div><div style='font-size:.78rem;color:#b8cbd7!important;margin:3px 0 20px'>ML Risk Insights Lab</div>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Risk assessment", "Model analytics", "About the project"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**PROJECT SNAPSHOT**")
    st.caption("Dataset · Kaggle Stroke Prediction")
    st.caption("Pipeline · Imputation + IQR capping + encoding")
    st.caption("Evaluation · PR-AUC, recall, precision, F1")
    st.markdown("---")
    st.caption("Academic prototype · Not for clinical use")

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Machine learning · Preventive health research</div>
  <h1>{'Risk Assessment Workspace' if page == 'Risk assessment' else ('Model Performance Studio' if page == 'Model analytics' else 'Project Overview')}</h1>
  <p>Explore a transparent machine-learning workflow built to study stroke-related patterns in a public dataset. Designed for learning and evaluation—not for medical decision-making.</p>
  <span class="hero-chip">● Model loaded &nbsp; · &nbsp; {model_name}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='notice'><strong>Important safety note:</strong> This is an educational research prototype. Its output is not a diagnosis, a clinically validated estimate, or a substitute for a qualified healthcare professional. Do not use it to make treatment decisions.</div>", unsafe_allow_html=True)

comparison = load_comparison()

if page == "Risk assessment":
    st.markdown("<div class='section-title'>Patient profile</div><div class='section-sub'>Enter a hypothetical profile to see how the trained classifier responds.</div>", unsafe_allow_html=True)
    with st.form("risk_assessment_form"):
        st.markdown("**01 / Demographic information**")
        a, b, c = st.columns(3)
        with a:
            age = st.slider("Age", 1, 100, 55, help="Age in years")
            gender = st.selectbox("Gender", ["Female", "Male"])
        with b:
            ever_married = st.selectbox("Ever married?", ["Yes", "No"])
            residence = st.selectbox("Residence", ["Urban", "Rural"])
        with c:
            work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], format_func=lambda x: {"Private":"Private sector", "Self-employed":"Self-employed", "Govt_job":"Government job", "children":"Child / student", "Never_worked":"Never worked"}[x])
            smoking = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"], format_func=lambda x: {"never smoked":"Never smoked", "formerly smoked":"Formerly smoked", "smokes":"Currently smokes", "Unknown":"Unknown / not recorded"}[x])
        st.markdown("**02 / Cardiovascular and metabolic indicators**")
        d, e, f = st.columns(3)
        with d:
            hypertension = st.selectbox("Hypertension history", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with e:
            heart_disease = st.selectbox("Heart disease history", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with f:
            glucose = st.number_input("Average glucose (mg/dL)", min_value=40.0, max_value=350.0, value=110.0, step=1.0)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
        submitted = st.form_submit_button("Generate model output  →", use_container_width=True)

    if submitted:
        sample = pd.DataFrame([{
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "ever_married": ever_married,
            "work_type": work_type, "Residence_type": residence,
            "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking,
        }])
        try:
            probability = float(pipeline.predict_proba(sample)[0][1])
            high = probability >= threshold
            st.markdown("<div class='section-title' style='margin-top:26px'>Assessment output</div><div class='section-sub'>A model score based on patterns in the training dataset.</div>", unsafe_allow_html=True)
            x1, x2, x3 = st.columns(3)
            with x1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>MODEL PROBABILITY</div><div class='metric-value'>{probability*100:.2f}%</div><div class='metric-note'>Uncalibrated model output</div></div>", unsafe_allow_html=True)
            with x2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>THRESHOLD USED</div><div class='metric-value'>{threshold:.3f}</div><div class='metric-note'>Selected during model tuning</div></div>", unsafe_allow_html=True)
            with x3:
                color = "#b45309" if high else "#087f8c"
                if probability >= threshold:
                    classifier_output = "Stroke Predicted"
                else:
                    classifier_output = "Stroke Not Predicted"
                st.markdown(f"<div class='metric-card'><div class='metric-label'>CLASSIFIER OUTPUT</div><div class='metric-value' style='font-size:1.35rem;color:{color}'>{status}</div><div class='metric-note'>Not a medical risk category</div></div>", unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0), text="Predicted probability (0–100%)")
            if high:
                st.info("This is a machine-learning prediction for educational purposes, ""not a medical diagnosis or a guarantee of a future stroke.")
            else:
                st.info("The model score is below its selected classification threshold. A below-threshold result does not rule out medical risk.")
            with st.expander("View the input record sent to the model"):
                st.dataframe(sample, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
    else:
        st.markdown("<div class='panel'><strong style='color:#17354b'>How to use this workspace</strong><p style='color:#718196;margin:.5rem 0 0'>Complete the profile fields and select <em>Generate model output</em>. Results appear below the form. Use hypothetical data for academic demonstration only.</p></div>", unsafe_allow_html=True)

elif page == "Model analytics":
    st.markdown("<div class='section-title'>Benchmark overview</div><div class='section-sub'>Compare model families using minority-class-aware metrics. Accuracy alone can be misleading on an imbalanced dataset.</div>", unsafe_allow_html=True)
    if comparison.empty:
        st.warning("Model comparison report was not found. Run the training script to generate it.")
    else:
        display = comparison.copy()
        metric_cols = [c for c in ["CV PR-AUC", "Test PR-AUC", "Test Recall", "Test Precision", "Test F1", "Test ROC-AUC", "Test Accuracy"] if c in display.columns]
        for col in metric_cols:
            display[col] = pd.to_numeric(display[col], errors="coerce")
        eligible = display[display["Model"].str.lower().ne("dummy (most frequent)")] if "Model" in display else display
        if not eligible.empty:
            best_row = eligible.sort_values("CV PR-AUC", ascending=False).iloc[0] if "CV PR-AUC" in eligible else eligible.iloc[0]
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f"<div class='metric-card'><div class='metric-label'>MODELS COMPARED</div><div class='metric-value'>{len(eligible)}</div><div class='metric-note'>Supervised classifiers</div></div>", unsafe_allow_html=True)
            with m2: st.markdown(f"<div class='metric-card'><div class='metric-label'>TOP CV PR-AUC</div><div class='metric-value'>{best_row.get('CV PR-AUC', float('nan')):.3f}</div><div class='metric-note'>{best_row['Model']} · cross-validation</div></div>", unsafe_allow_html=True)
            with m3: st.markdown(f"<div class='metric-card'><div class='metric-label'>SELECTED THRESHOLD</div><div class='metric-value'>{threshold:.3f}</div><div class='metric-note'>{model_name}</div></div>", unsafe_allow_html=True)
        st.markdown("#### Test-set performance")
        st.dataframe(display, use_container_width=True, hide_index=True, column_config={
            "CV PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Test PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Test Recall": st.column_config.NumberColumn(format="%.3f"),
            "Test Precision": st.column_config.NumberColumn(format="%.3f"),
            "Test F1": st.column_config.NumberColumn(format="%.3f"),
            "Test ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Test Accuracy": st.column_config.NumberColumn(format="%.3f"),
        })
        if "Model" in display and "Test PR-AUC" in display:
            chart_data = display.set_index("Model")[[c for c in ["Test PR-AUC", "Test Recall", "Test Precision", "Test F1"] if c in display]].dropna(how="all")
            st.markdown("#### Metric comparison")
            st.bar_chart(chart_data, use_container_width=True)
        st.caption("PR-AUC is especially useful when positive cases are rare. Metrics are specific to this dataset split and are not evidence of clinical validity.")
    if feature_importances:
        st.markdown("#### Global feature importance")
        fi = pd.DataFrame({"Feature": list(feature_importances.keys()), "Relative importance": list(feature_importances.values())}).sort_values("Relative importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("white")
        ax.barh(fi["Feature"], fi["Relative importance"], color="#0b9298")
        ax.set_xlabel("Relative importance (not causal effect)", color="#526477")
        ax.tick_params(axis="both", colors="#526477")
        for spine in ax.spines.values(): spine.set_color("#e5edf2")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

else:
    st.markdown("<div class='section-title'>About this project</div><div class='section-sub'>An end-to-end academic machine-learning project combining data cleaning, model benchmarking, threshold selection and an interactive interface.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'><h4 style='color:#17354b;margin-top:0'>Data workflow</h4><p style='color:#718196'>The dataset is cleaned by removing the identifier and unsupported gender category, coercing BMI to numeric, and handling missing values inside training pipelines.</p><p style='color:#718196'>Numeric preprocessing includes median imputation, IQR-based capping and scaling. Categorical variables are imputed and one-hot encoded.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><h4 style='color:#17354b;margin-top:0'>Evaluation approach</h4><p style='color:#718196'>The project compares a majority-class baseline with Logistic Regression, KNN, Decision Tree and Random Forest. It reports PR-AUC, ROC-AUC, recall, precision, F1 and accuracy.</p><p style='color:#718196'>A decision threshold is selected from out-of-fold predictions to prioritize recall-oriented performance. Results should be interpreted cautiously.</p></div>", unsafe_allow_html=True)
    st.markdown("#### Current project limitations")
    st.markdown("- The dataset is imbalanced and contains relatively few positive examples.\n- A model probability is not necessarily calibrated as an individual clinical risk.\n- Group-wise metrics can reveal performance differences but do not establish fairness or clinical suitability.\n- Independent external validation and clinical review would be required before any real-world use.")
    st.markdown("#### Technologies")
    st.write("Python · Pandas · NumPy · scikit-learn · imbalanced-learn · Streamlit · Matplotlib · Joblib")

st.markdown("<div style='text-align:center;color:#8a9aaa;font-size:.75rem;padding:24px 0 0'>STROKEGUARD · ACADEMIC ML PROJECT &nbsp; · &nbsp; FOR EDUCATION ONLY</div>", unsafe_allow_html=True)
