import joblib
from sklearn.metrics import roc_auc_score, accuracy_score

from app.ml.data import generate_customer_data
from app.ml.preprocessing import preprocess_data

MODEL_PATH = "artifacts/models/xgboost_churn_model.joblib"


def analyze_fairness():
    model = joblib.load(MODEL_PATH)

    df = generate_customer_data(n_samples=10000)

    _, X_test, _, y_test, X_test_original, _ = preprocess_data(df)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    gender_cols = [col for col in X_test_original.columns if col.startswith("gender_")]

    results = []

    for col in gender_cols:
        group_name = col.replace("gender_", "")
        mask = X_test_original[col].values == 1

        group_y = y_test[mask]
        group_proba = y_proba[mask]
        group_pred = y_pred[mask]

        results.append({
            "group": group_name,
            "count": int(mask.sum()),
            "churn_rate": round(float(group_y.mean()), 4),
            "auc_roc": round(float(roc_auc_score(group_y, group_proba)), 4),
            "accuracy": round(float(accuracy_score(group_y, group_pred)), 4)
        })

    auc_values = [item["auc_roc"] for item in results]
    accuracy_values = [item["accuracy"] for item in results]

    fairness_summary = {
        "auc_difference": round(float(max(auc_values) - min(auc_values)), 4),
        "accuracy_difference": round(float(max(accuracy_values) - min(accuracy_values)), 4),
        "fairness_status": "Fair" if max(auc_values) - min(auc_values) < 0.02 else "Review Needed"
    }

    return {
        "protected_attribute": "gender",
        "group_metrics": results,
        "fairness_summary": fairness_summary
    }