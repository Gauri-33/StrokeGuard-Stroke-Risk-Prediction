import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Add project root to sys.path so joblib can unpickle custom transformers
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import IQRCapper  # Required for unpickling

ml_bundle = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.path.join(ROOT_DIR, "models", "winning_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run 'python -m src.train' first.")
    
    artifacts = joblib.load(model_path)
    ml_bundle["pipeline"] = artifacts["pipeline"]
    ml_bundle["threshold"] = float(artifacts["threshold"])
    ml_bundle["feature_importances"] = artifacts.get("feature_importances", {})
    # Retrieve feature column names from training artifact
    ml_bundle["features"] = artifacts.get("features", [
        "gender", "age", "hypertension", "heart_disease", "ever_married",
        "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
    ])
    yield
    ml_bundle.clear()

app = FastAPI(title="Stroke Risk Prediction", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

class PatientInput(BaseModel):
    Age: float = Field(..., ge=1, le=120)
    Gender: str
    Country: str = "United States"
    Height_cm: float = Field(170.0, ge=100, le=250)
    Weight_kg: float = Field(70.0, ge=30, le=250)
    BMI: float = Field(24.2, ge=10, le=70)
    Blood_Pressure_Systolic: int = Field(120, ge=70, le=240)
    Blood_Pressure_Diastolic: int = Field(80, ge=40, le=140)
    Heart_Rate: int = Field(72, ge=40, le=200)
    Blood_Glucose: float = Field(100.0, ge=40, le=400)
    HbA1c: float = Field(5.5, ge=3.0, le=15.0)
    Total_Cholesterol: float = Field(190.0, ge=80, le=450)
    HDL: float = Field(50.0, ge=15, le=120)
    LDL: float = Field(110.0, ge=30, le=300)
    Triglycerides: float = Field(140.0, ge=30, le=600)
    Smoking_Status: str = "Never"
    Alcohol_Consumption: str = "Never"
    Physical_Activity_Level: str = "Moderate"
    Sleep_Hours: float = Field(7.0, ge=2, le=16)
    Stress_Level: str = "Moderate"
    Diet_Quality: str = "Average"
    Family_History_Stroke: str = "No"
    Family_History_Heart_Disease: str = "No"
    Diabetes: str = "No"
    Hypertension: str = "No"
    Heart_Disease: str = "No"
    Previous_TIA: str = "No"
    Atrial_Fibrillation: str = "No"
    Chronic_Kidney_Disease: str = "No"
    Medication_Adherence: str = "Average"
    Exercise_Hours_Per_Week: float = Field(2.5, ge=0, le=40)
    Daily_Walking_Minutes: float = Field(30.0, ge=0, le=300)
    Work_Type: str = "Private"
    Residence_Type: str = "Urban"
    Air_Pollution_Exposure: str = "Low"

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "threshold": round(float(ml_bundle.get("threshold", 0.5)), 3),
            "feature_importances": ml_bundle.get("feature_importances", {})
        }
    )

@app.post("/predict")
def predict(payload: PatientInput):
    try:
        pipeline = ml_bundle["pipeline"]
        threshold = float(ml_bundle["threshold"])
        feature_cols = ml_bundle["features"]

        input_dict = payload.model_dump()
        df_sample = pd.DataFrame([input_dict])[feature_cols]

        prob = float(pipeline.predict_proba(df_sample)[0][1])
        is_high = bool(prob >= threshold)

        # Display the model's raw probability as a percentage. This is not a
        # calibrated clinical risk estimate; the classification threshold is separate.
        return {
            "raw_probability": round(prob * 100, 2),
            "risk_score": round(prob * 100, 1),
            "threshold": round(threshold, 3),
            "is_high_risk": is_high
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))