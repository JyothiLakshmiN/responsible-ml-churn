import os

REQUIRED_MODEL_FILES = [
    "artifacts/models/xgboost_churn_model.joblib",
    "artifacts/models/scaler.joblib",
    "artifacts/models/feature_names.joblib",
    "artifacts/models/metrics.joblib",
]


def get_model_status():
    files = []

    for file_path in REQUIRED_MODEL_FILES:
        files.append({
            "file": file_path,
            "exists": os.path.exists(file_path)
        })

    is_ready = all(item["exists"] for item in files)

    return {
        "model_ready": is_ready,
        "required_files": files
    }