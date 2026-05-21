import joblib
import pandas as pd
import shap

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


def explain_prediction(customer_data: dict):
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

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(df_encoded)

    explanation = []

    for feature, value, shap_value in zip(
        feature_names,
        df_encoded.iloc[0].values,
        shap_values[0]
    ):
        explanation.append({
            "feature": feature,
            "input_value": round(float(value), 4),
            "shap_value": round(float(shap_value), 4),
            "impact": (
                "increases churn risk"
                if shap_value > 0
                else "decreases churn risk"
            )
        })

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return {
        "top_factors": explanation[:8],
        "all_factors": explanation
    }