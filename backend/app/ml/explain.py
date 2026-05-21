import joblib
import pandas as pd
import shap

MODEL_PATH = "artifacts/models/xgboost_churn_model.joblib"
SCALER_PATH = "artifacts/models/scaler.joblib"
FEATURES_PATH = "artifacts/models/feature_names.joblib"


def explain_prediction(customer_data: dict):
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scaled)

    explanation = []

    for feature, value, shap_value in zip(
        feature_names,
        df_encoded.iloc[0].values,
        shap_values[0]
    ):
        explanation.append({
            "feature": feature,
            "input_value": float(value),
            "shap_value": round(float(shap_value), 4),
            "impact": "increases churn risk" if shap_value > 0 else "decreases churn risk"
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