from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.ml.data import generate_customer_data
from app.ml.train import train_xgboost_model
from pydantic import BaseModel, Field
from app.ml.predict import predict_churn
from app.ml.explain import explain_prediction
from app.ml.fairness import analyze_fairness
from app.ml.model_status import get_model_status
from typing import Literal

import joblib

class CustomerInput(BaseModel):
    age: float = Field(..., ge=18, le=90)
    tenure_months: float = Field(..., ge=1, le=120)
    monthly_charge: float = Field(..., ge=20, le=150)
    num_products: int = Field(..., ge=1, le=5)
    support_calls: int = Field(..., ge=0, le=20)

    gender: Literal["Male", "Female"]
    contract_type: Literal["Month-to-Month", "One Year", "Two Year"]
    payment_method: Literal[
        "Credit Card",
        "Bank Transfer",
        "Electronic Check",
        "Mailed Check"
    ]

app = FastAPI(
    title="Responsible ML Churn API",
    description="Production-grade churn prediction with explainability and fairness analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Responsible ML Churn API Running"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "responsible-ml-churn"}

@app.post("/generate-data")
def generate_data():

    df = generate_customer_data()

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "sample_data": df.head(5).to_dict(orient="records")
    }

@app.post("/train")
def train_model():
    metrics = train_xgboost_model()

    return {
        "message": "Model trained successfully",
        "metrics": metrics
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    result = predict_churn(customer.model_dump())

    return {
        "message": "Prediction generated successfully",
        "input": customer.model_dump(),
        "result": result
    }

@app.get("/metrics")
def get_metrics():
    metrics = joblib.load("artifacts/models/metrics.joblib")

    return {
        "message": "Model metrics retrieved successfully",
        "metrics": metrics
    }

@app.post("/explain")
def explain(customer: CustomerInput):
    result = explain_prediction(customer.model_dump())

    return {
        "message": "SHAP explanation generated successfully",
        "input": customer.model_dump(),
        "explanation": result
    }

@app.get("/fairness")
def fairness_report():
    result = analyze_fairness()

    return {
        "message": "Fairness analysis completed successfully",
        "fairness_report": result
    }

@app.get("/model-status")
def model_status():
    return get_model_status()