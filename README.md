# Responsible ML Churn

Production-grade Explainable AI platform for trustworthy customer churn prediction using FastAPI, XGBoost, SHAP, and Next.js.

![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-blue)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![Responsible AI](https://img.shields.io/badge/AI-Responsible%20AI-purple)

---

# Overview

Responsible ML Churn is an end-to-end AI platform that predicts customer churn while providing transparent and explainable model decisions.

Unlike traditional black-box ML systems, this platform emphasizes:

- Explainable AI (XAI)
- Fairness analysis
- Trustworthy predictions
- Production-ready APIs
- Interactive visualization dashboards

The application combines modern AI engineering practices with full-stack product development.

---

# Key Features

## AI-Powered Churn Prediction

Predicts whether a customer is likely to churn using an XGBoost machine learning model.

## Explainable AI with SHAP

Provides feature-level explanations showing exactly why the model made a prediction.

## Fairness Analysis

Evaluates model behavior across demographic groups to detect potential bias.

## Real-Time Inference APIs

FastAPI backend serving live predictions and explainability results.

## Interactive Dashboard

Next.js frontend with:
- Customer risk input forms
- Prediction visualizations
- SHAP impact analysis
- Fairness metrics
- Risk indicators

## Production-Oriented Architecture

Designed using scalable AI application patterns:
- Modular ML pipeline
- Artifact persistence
- API-driven architecture
- Frontend/backend separation

---

# Demo

## Customer Risk Prediction

Users can input:
- Age
- Tenure
- Monthly charges
- Product usage
- Support interactions
- Contract type
- Payment method

The system returns:
- Churn probability
- Risk level
- Prediction confidence
- Feature importance analysis

---

# Explainable AI

The platform uses SHAP (SHapley Additive exPlanations) to explain predictions.

Example:

| Feature | Impact |
|---|---|
| Support Calls | Increased churn risk |
| Short Tenure | Increased churn risk |
| Month-to-Month Contract | Increased churn risk |
| Age | Reduced churn risk |

This enables transparent and interpretable AI decisions.

---

# Fairness & Responsible AI

The platform includes fairness analysis across customer groups.

Metrics include:
- AUC comparison
- Accuracy comparison
- Churn rate distribution
- Fairness threshold checks

This aligns with modern Responsible AI principles used in enterprise AI systems.

---

# Tech Stack

## Frontend
- Next.js 15
- TypeScript
- Tailwind CSS
- Recharts
- Axios

## Backend
- FastAPI
- Pydantic
- Uvicorn

## Machine Learning
- XGBoost
- Scikit-learn
- SHAP
- Pandas
- NumPy

## AI Engineering
- Explainable AI (XAI)
- Responsible AI
- Fairness Evaluation
- ML Artifact Management

---

# Architecture

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

# Project Structure
responsible-ml-churn/

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
│   │   └── main.py
│   │
│   ├── artifacts/
│   │   └── models/
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   ├── lib/
│   └── components/
│
└── README.md

# API Endpoints

| Endpoint        | Method | Description                |
| --------------- | ------ | -------------------------- |
| `/health`       | GET    | API health check           |
| `/train`        | POST   | Train churn model          |
| `/predict`      | POST   | Predict customer churn     |
| `/explain`      | POST   | Generate SHAP explanations |
| `/fairness`     | GET    | Fairness analysis          |
| `/metrics`      | GET    | Model performance metrics  |
| `/model-status` | GET    | Model artifact status      |

# Model Performance

Example Metrics:

AUC-ROC: 0.66+
Accuracy: 61%+
Explainability: Enabled
Fairness Monitoring: Enabled


# Running Locally

git clone https://github.com/YOUR_USERNAME/responsible-ml-churn.git
cd responsible-ml-churn

# Backend Setup

cd backend

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python3 -m uvicorn app.main:app --reload

# Frontend Setup

cd frontend

npm install

npm run dev