from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"
FEATURES = [
    "tenure_months",
    "support_tickets",
    "monthly_spend_inr",
    "last_login_days",
    "plan_type",
]

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description=(
        "Educational inference API. The supplied 12-row sample is not "
        "sufficient for production use; predictions are experimental only."
    ),
)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=600, description="Customer tenure in months")
    support_tickets: int = Field(ge=0, le=10000, description="Support tickets in the agreed observation period")
    monthly_spend_inr: float = Field(ge=0, le=10_000_000, description="Monthly spend in INR")
    last_login_days: int = Field(ge=0, le=36500, description="Days since last login")
    plan_type: str = Field(description="One of Basic, Standard, Pro")

    @field_validator("plan_type")
    @classmethod
    def validate_plan_type(cls, value: str) -> str:
        cleaned = value.strip().title()
        if cleaned not in {"Basic", "Standard", "Pro"}:
            raise ValueError("plan_type must be Basic, Standard, or Pro")
        return cleaned

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: float
    model_status: str
    warning: str

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model artifact is missing. Run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        probability = float(model.predict_proba(row)[0][1])
        prediction = int(probability >= 0.5)
        return PredictionResponse(
            prediction=prediction,
            label="churned" if prediction == 1 else "not_churned",
            churn_probability=round(probability, 6),
            model_status="experimental_not_validated",
            warning=(
                "Educational demonstration only: trained on 12 records. "
                "Do not use for real customer decisions or automated actions."
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
