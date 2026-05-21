import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from app.ml.data import load_customer_churn_data
from app.ml.train import train_xgboost_model
from app.ml.predict import predict_churn
from app.ml.explain import explain_prediction
from app.ml.fairness import analyze_fairness
from app.ml.model_status import get_model_status


class CustomerInput(BaseModel):
    senior_citizen: int = Field(..., ge=0, le=1)
    tenure_months: float = Field(..., ge=0, le=100)
    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)

    gender: Literal["Male", "Female"]
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    online_security: Literal["Yes", "No", "No internet service"]


app = FastAPI(
    title="Responsible ML Churn API",
    description="Explainable customer churn prediction using IBM Telco Customer Churn data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://responsible-ml-churn-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Responsible ML Churn API Running",
        "dataset": "IBM Telco Customer Churn",
        "version": "2.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "responsible-ml-churn",
        "dataset": "IBM Telco Customer Churn"
    }


@app.get("/dataset-preview")
def dataset_preview():
    df = load_customer_churn_data()

    return {
        "dataset": "IBM Telco Customer Churn",
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
