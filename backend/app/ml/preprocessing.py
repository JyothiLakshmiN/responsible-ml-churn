import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "artifacts/models/scaler.joblib"
FEATURES_PATH = "artifacts/models/feature_names.joblib"


def preprocess_data(df: pd.DataFrame):
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X = pd.get_dummies(
        X,
        columns=["gender", "contract_type", "payment_method"]
    )

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test, feature_names