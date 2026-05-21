import joblib
import pandas as pd

MODEL_PATH = "artifacts/models/xgboost_churn_model.joblib"
SCALER_PATH = "artifacts/models/scaler.joblib"
FEATURES_PATH = "artifacts/models/feature_names.joblib"

NUMERICAL_COLUMNS = [
    "senior_citizen",
    "tenure_months",
    "monthly_charge",
    "total_charges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "internet_service",
    "online_security",
    "OnlineBackup",
    "DeviceProtection",
    "tech_support",
    "StreamingTV",
    "StreamingMovies",
    "contract_type",
    "PaperlessBilling",
    "payment_method",
]


def predict_churn(customer_data: dict):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    df = pd.DataFrame([customer_data])

    available_categorical_columns = [
        col for col in CATEGORICAL_COLUMNS
        if col in df.columns
    ]

    df_encoded = pd.get_dummies(
        df,
        columns=available_categorical_columns,
        drop_first=False,
        dtype=int
    )

    df_encoded = df_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )
    
    df_encoded = df_encoded.astype(float)

    available_numerical_columns = [
        col for col in NUMERICAL_COLUMNS
        if col in df_encoded.columns
    ]

    df_encoded.loc[:, available_numerical_columns] = scaler.transform(
        df_encoded[available_numerical_columns]
    )


    churn_probability = float(model.predict_proba(df_encoded)[0][1])
    churn_prediction = int(churn_probability >= 0.5)

    risk_level = (
        "High"
        if churn_probability >= 0.7
        else "Medium"
        if churn_probability >= 0.4
        else "Low"
    )

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(churn_probability, 4),
        "risk_level": risk_level
    }