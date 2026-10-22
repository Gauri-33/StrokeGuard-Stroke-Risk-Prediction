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
    gender: str
    age: float = Field(..., ge=1, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float = Field(..., ge=40, le=350)
    bmi: float = Field(..., ge=10, le=70)
    smoking_status: str

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