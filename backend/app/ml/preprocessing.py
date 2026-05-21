import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "artifacts/models/scaler.joblib"
FEATURES_PATH = "artifacts/models/feature_names.joblib"


def preprocess_data(df: pd.DataFrame):
    os.makedirs("artifacts/models", exist_ok=True)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    categorical_columns = [
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

    categorical_columns = [
        col for col in categorical_columns if col in X.columns
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False,
        dtype=int
    )

    X = X.astype(float)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numerical_columns = [
        "senior_citizen",
        "tenure_months",
        "monthly_charge",
        "total_charges",
    ]

    numerical_columns = [
        col for col in numerical_columns if col in X_train.columns
    ]

    scaler = StandardScaler()

    X_train.loc[:, numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test.loc[:, numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_test.copy(),
        feature_names
    )