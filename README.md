# Responsible AI Customer Retention Platform

Production-grade Explainable AI platform for customer churn prediction using FastAPI, Next.js, XGBoost, SHAP explainability, and fairness analysis on the IBM Telco Customer Churn dataset.

---

## Overview

here is the demo link: https://responsible-ml-churn-frontend.onrender.com/

This project is a full-stack Responsible AI application designed to predict customer churn for SaaS and telecom businesses while ensuring transparency, explainability, and fairness in machine learning predictions.

The platform combines:

- Machine Learning with XGBoost
- Explainable AI using SHAP
- Fairness Analysis across demographic groups
- FastAPI backend APIs
- Next.js production-grade frontend
- Real-world IBM Telco Customer Churn dataset

The application allows users to:
- Predict customer churn risk
- Understand why predictions were made
- Analyze fairness across protected groups
- Visualize SHAP feature importance
- Monitor model health and metrics

---

# Live Features

## AI-Powered Churn Prediction
Predicts whether a customer is likely to churn based on behavioral and subscription features.

## Explainable AI (XAI)
Uses SHAP (SHapley Additive Explanations) to explain:
- Which factors increased churn risk
- Which factors reduced churn risk
- Relative feature importance

## Fairness Analysis
Evaluates model fairness across demographic groups such as gender.

Metrics include:
- AUC comparison
- Accuracy comparison
- Churn rate comparison
- Fairness status evaluation

## Production-Grade Full Stack Architecture
Frontend and backend are independently deployable and communicate through REST APIs.

---

# Tech Stack

## Frontend
- Next.js 15
- React
- TypeScript
- Tailwind CSS
- Recharts
- Axios

## Backend
- FastAPI
- Python
- Pydantic
- Uvicorn

## Machine Learning
- XGBoost
- Scikit-learn
- SHAP
- Pandas
- NumPy

## Deployment
- Render (Frontend + Backend)

---

# System Architecture

```text
Frontend (Next.js)
        ↓
FastAPI Backend
        ↓
Prediction Service
        ↓
XGBoost ML Model
        ↓
SHAP Explainability Engine
        ↓
Fairness Analysis Module
```

# Dataset
This project uses the real-world IBM Telco Customer Churn dataset from Kaggle.

Dataset Link:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset contains:

Customer demographics
Subscription details
Billing information
Service usage
Churn labels

The dataset is used to train and evaluate the XGBoost churn prediction model.

# Machine Learning Workflow
IBM Telco Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
One-Hot Encoding
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
XGBoost Training
        ↓
Prediction API
        ↓
SHAP Explainability
        ↓
Fairness Evaluation

# Features Used

Numerical Features
    senior_citizen
    tenure_months
    monthly_charge
    total_charges

Categorical Features
    gender
    contract_type
    payment_method
    internet_service
    tech_support
    online_security

# Explainable AI with SHAP
The project integrates SHAP to provide transparent explanations for every prediction.

Example insights:

Month-to-month contracts increase churn risk
Fiber optic users may show higher churn probability
Long customer tenure reduces churn risk
Lack of tech support increases churn risk

SHAP values are visualized directly in the frontend dashboard.

# Fairness Analysis
The platform evaluates fairness across demographic groups.

Current fairness checks:

Gender-based performance comparison
AUC difference analysis
Accuracy difference analysis

This ensures the model behaves consistently across groups.

# API Endpoints
Health Check: GET /health
Dataset Preview: GET /dataset-preview
Train Model: POST /train
Predict Churn: POST /predict

Example Request:

{
  "senior_citizen": 0,
  "tenure_months": 12,
  "monthly_charge": 85,
  "total_charges": 1200,
  "gender": "Female",
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "online_security": "No"
}
Explain Prediction: POST /explain
Fairness Analysis: GET /fairness
Model Metrics: GET /metrics

# Project Structure
# Project Structure

```text
responsible-ml-churn/
│
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── data.py
│   │   │   ├── preprocessing.py
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   ├── explain.py
│   │   │   ├── fairness.py
│   │   │   └── model_status.py
│   │   │
│   │   └── main.py
│   │
│   ├── artifacts/
│   │   └── models/
│   │
│   ├── data/
│   │   └── telco_churn.csv
│   │
│   ├── requirements.txt
│   └── venv/
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── layout.tsx
│   │
│   ├── lib/
│   │   └── api.ts
│   │
│   ├── package.json
│   └── next.config.ts
│
└── README.md
```

# Local Development Setup
Clone Repository
git clone https://github.com/JyothiLakshmiN/responsible-ml-churn.git

Backend Setup:
cd backend

python3 -m venv venv

source venv/bin/activate
pip install -r requirements.txt

run backend: python3 -m uvicorn app.main:app --reload

Backend runs on: http://127.0.0.1:8000
Swagger Docs: http://127.0.0.1:8000/docs

# Frontend Setup
cd frontend

npm install

npm run dev

Frontend runs on: http://localhost:3000

# Model Training
Train the model using: POST /train (go to Swagger Docs and see this call)

Artifacts generated:
xgboost_churn_model.joblib
scaler.joblib
feature_names.joblib
metrics.joblib

# Deployment
Frontend Deployment
    Render Static Site / Web Service
Backend Deployment
    Render Web Service

# Environment Variable:
NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com

# Production Features
REST API architecture
Typed request validation with Pydantic
Scalable frontend/backend separation
Real-world dataset integration
Explainable AI
Fairness evaluation
Responsive UI dashboard
Deployment-ready architecture

# Future Improvements
Authentication & RBAC
MLflow experiment tracking
CI/CD pipelines
Docker containerization
Drift monitoring
Real-time inference logging
Advanced fairness metrics
LLM-generated customer retention recommendations

Author

Jyothi Lakshmi N

GitHub:
https://github.com/JyothiLakshmiN

LinkedIn:
https://www.linkedin.com/in/jyothilakshmin/
