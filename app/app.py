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

from src.preprocessing import IQRCapper  # noqa: F401 — needed to unpickle custom transformer

st.set_page_config(
    page_title="StrokeGuard | Risk Insights",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
.section-title { font-family:Manrope,sans-serif; font-weight:800; font-size:1.25rem; color:#17354b; margin:1.2rem 0 .2rem; letter-spacing:-.025em; }
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
    st.caption("Dataset · Stroke Risk Prediction Dataset")
    st.caption("Pipeline · Imputation + IQR capping + Encoding")
    st.caption("Evaluation · PR-AUC, Recall, Precision, F1")
    st.markdown("---")
    st.caption("Academic prototype · Not for clinical use")

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Machine learning · Preventive health research</div>
  <h1>{'Risk Assessment Workspace' if page == 'Risk assessment' else ('Model Performance Studio' if page == 'Model analytics' else 'Project Overview')}</h1>
  <p>Explore an end-to-end machine learning system trained on demographic, clinical, physiological, and lifestyle metrics.</p>
  <span class="hero-chip">● Model loaded &nbsp; · &nbsp; {model_name}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='notice'><strong>Important safety note:</strong> This is an educational research prototype. Its output is not a diagnosis, a clinically validated estimate, or a substitute for a qualified healthcare professional. Do not use it to make treatment decisions.</div>", unsafe_allow_html=True)

comparison = load_comparison()

if page == "Risk assessment":
    st.markdown("<div class='section-title'>Patient Profile</div><div class='section-sub'>Enter patient parameters to evaluate stroke risk probability.</div>", unsafe_allow_html=True)
    with st.form("risk_assessment_form"):
        st.markdown("**01 / Demographics & Vitals**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 1, 100, 55, help="Age in years")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            country = st.selectbox("Country", ["China", "United Kingdom", "Germany", "Brazil", "United States", "Singapore", "India", "Canada", "Australia", "Other"])
        with c2:
            height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)
            bmi = round(weight / ((height / 100) ** 2), 1)
            st.metric("Computed BMI", f"{bmi} kg/m²")
        with c3:
            bp_sys = st.number_input("Blood Pressure Systolic", min_value=80, max_value=220, value=125, step=1)
            bp_dia = st.number_input("Blood Pressure Diastolic", min_value=50, max_value=140, value=82, step=1)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=75, step=1)

        st.markdown("**02 / Blood & Metabolic Biomarkers**")
        c4, c5, c6 = st.columns(3)
        with c4:
            glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50.0, max_value=350.0, value=115.0, step=1.0)
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, value=5.8, step=0.1)
        with c5:
            total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, step=1.0)
            hdl = st.number_input("HDL (mg/dL)", min_value=15.0, max_value=120.0, value=50.0, step=1.0)
        with c6:
            ldl = st.number_input("LDL (mg/dL)", min_value=40.0, max_value=300.0, value=120.0, step=1.0)
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=40.0, max_value=500.0, value=150.0, step=1.0)

        st.markdown("**03 / Medical History & Conditions**")
        c7, c8, c9 = st.columns(3)
        with c7:
            hypertension = st.selectbox("Hypertension?", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
        with c8:
            fam_stroke = st.selectbox("Family History of Stroke?", ["No", "Yes"])
            fam_heart = st.selectbox("Family History of Heart Disease?", ["No", "Yes"])
            prev_tia = st.selectbox("Previous TIA (Mini-stroke)?", ["No", "Yes"])
        with c9:
            afib = st.selectbox("Atrial Fibrillation?", ["No", "Yes"])
            ckd = st.selectbox("Chronic Kidney Disease?", ["No", "Yes"])
            med_adherence = st.selectbox("Medication Adherence", ["Excellent", "Average", "Poor"])

        st.markdown("**04 / Lifestyle, Activity & Environment**")
        c10, c11, c12 = st.columns(3)
        with c10:
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Frequently"])
            diet = st.selectbox("Diet Quality", ["Healthy", "Average", "Poor"])
        with c11:
            activity = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low"])
            exercise_hours = st.number_input("Exercise Hours / Week", min_value=0.0, max_value=35.0, value=2.5, step=0.5)
            walking_mins = st.number_input("Daily Walking Minutes", min_value=0.0, max_value=300.0, value=45.0, step=5.0)
        with c12:
            sleep = st.number_input("Sleep Hours / Day", min_value=3.0, max_value=14.0, value=7.0, step=0.5)
            stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
            pollution = st.selectbox("Air Pollution Exposure", ["Low", "Medium", "High"])
            work_type = st.selectbox("Work Type", ["Private", "Self-Employed", "Government", "Retired", "Student"])
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])

        submitted = st.form_submit_button("Generate model output  →", use_container_width=True)

    if submitted:
        sample = pd.DataFrame([{
            'Age': age,
            'Height_cm': height,
            'Weight_kg': weight,
            'BMI': bmi,
            'Blood_Pressure_Systolic': bp_sys,
            'Blood_Pressure_Diastolic': bp_dia,
            'Heart_Rate': heart_rate,
            'Blood_Glucose': glucose,
            'HbA1c': hba1c,
            'Total_Cholesterol': total_chol,
            'HDL': hdl,
            'LDL': ldl,
            'Triglycerides': triglycerides,
            'Sleep_Hours': sleep,
            'Exercise_Hours_Per_Week': exercise_hours,
            'Daily_Walking_Minutes': walking_mins,
            'Gender': gender,
            'Country': country,
            'Smoking_Status': smoking,
            'Alcohol_Consumption': alcohol,
            'Physical_Activity_Level': activity,
            'Stress_Level': stress,
            'Diet_Quality': diet,
            'Family_History_Stroke': fam_stroke,
            'Family_History_Heart_Disease': fam_heart,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'Heart_Disease': heart_disease,
            'Previous_TIA': prev_tia,
            'Atrial_Fibrillation': afib,
            'Chronic_Kidney_Disease': ckd,
            'Medication_Adherence': med_adherence,
            'Work_Type': work_type,
            'Residence_Type': residence,
            'Air_Pollution_Exposure': pollution
        }])
        try:
            probability = float(pipeline.predict_proba(sample)[0][1])
            high = probability >= threshold
            st.markdown("<div class='section-title' style='margin-top:26px'>Assessment output</div><div class='section-sub'>A model score based on patterns in the training dataset.</div>", unsafe_allow_html=True)
            x1, x2, x3 = st.columns(3)
            with x1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>MODEL PROBABILITY</div><div class='metric-value'>{probability*100:.2f}%</div><div class='metric-note'>Uncalibrated model output</div></div>", unsafe_allow_html=True)
            with x2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>THRESHOLD USED</div><div class='metric-value'>{threshold:.3f}</div><div class='metric-note'>Optimal F2-score threshold</div></div>", unsafe_allow_html=True)
            with x3:
                color = "#b45309" if high else "#087f8c"
                status = "High Risk (Above Threshold)" if high else "Low/Moderate (Below Threshold)"
                st.markdown(f"<div class='metric-card'><div class='metric-label'>CLASSIFIER OUTPUT</div><div class='metric-value' style='font-size:1.25rem;color:{color}'>{status}</div><div class='metric-note'>Binary screening verdict</div></div>", unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0), text="Predicted stroke risk probability (0–100%)")
            if high:
                st.warning("⚠️ The predicted probability is above the risk threshold. Please consult a qualified healthcare professional.")
            else:
                st.success("✅ The predicted probability is below the high-risk threshold.")
            with st.expander("View the raw input record sent to the model"):
                st.dataframe(sample, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
    else:
        st.markdown("<div class='panel'><strong style='color:#17354b'>How to use this workspace</strong><p style='color:#718196;margin:.5rem 0 0'>Complete the profile fields above and select <em>Generate model output</em>. All predictions are generated in real-time by your trained pipeline.</p></div>", unsafe_allow_html=True)

elif page == "Model analytics":
    st.markdown("<div class='section-title'>Benchmark overview</div><div class='section-sub'>Model comparison across architectures.</div>", unsafe_allow_html=True)
    if comparison.empty:
        st.warning("Model comparison report was not found. Run `python -m src.train` to generate it.")
    else:
        display = comparison.copy()
        st.dataframe(display, use_container_width=True, hide_index=True)

elif page == "About the project":
    st.markdown("<div class='section-title'>About StrokeGuard</div><div class='section-sub'>Research and educational machine learning project for stroke risk assessment.</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Dataset**: `stroke_risk_prediction_dataset.csv` (50,000 patient records across 35 clinical and lifestyle features)
    - **Models evaluated**: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors
    - **Winning Pipeline**: Logistic Regression with standard scaling, imputation, and 1.5× IQR Winsorization
    """)