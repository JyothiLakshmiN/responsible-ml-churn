import os
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from app.ml.data import load_customer_churn_data
from app.ml.preprocessing import preprocess_data

MODEL_PATH = "artifacts/models/xgboost_churn_model.joblib"
METRICS_PATH = "artifacts/models/metrics.joblib"


def train_xgboost_model():
    os.makedirs("artifacts/models", exist_ok=True)

    df = load_customer_churn_data()

    X_train, X_test, y_train, y_test, X_test_original, feature_names = preprocess_data(df)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "dataset": "IBM Telco Customer Churn",
        "model_name": "XGBoost Classifier",
        "problem_type": "Customer churn prediction",
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            output_dict=True,
        ),
        "features": feature_names,
        "training_rows": int(len(df)),
        "test_rows": int(len(y_test)),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return metrics