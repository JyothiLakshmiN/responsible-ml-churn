import joblib
import pandas as pd

MODEL_PATH = "artifacts/models/xgboost_churn_model.joblib"
SCALER_PATH = "artifacts/models/scaler.joblib"
FEATURES_PATH = "artifacts/models/feature_names.joblib"


def predict_churn(customer_data: dict):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    df = pd.DataFrame([customer_data])

    df_encoded = pd.get_dummies(
        df,
        columns=["gender", "contract_type", "payment_method"]
    )

    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

    df_scaled = scaler.transform(df_encoded)

    churn_probability = float(model.predict_proba(df_scaled)[0][1])
    churn_prediction = int(churn_probability >= 0.5)

    risk_level = "High" if churn_probability >= 0.7 else "Medium" if churn_probability >= 0.4 else "Low"

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(churn_probability, 4),
        "risk_level": risk_level
    }