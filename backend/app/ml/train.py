import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from app.ml.data import generate_customer_data
from app.ml.preprocessing import preprocess_data

MODEL_PATH = "artifacts/models/xgboost_churn_model.joblib"
METRICS_PATH = "artifacts/models/metrics.joblib"


def train_xgboost_model():
    df = generate_customer_data(n_samples=10000)

    X_train, X_test, y_train, y_test, X_test_original, feature_names = preprocess_data(df)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            output_dict=True,
        ),
        "features": feature_names,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return metrics