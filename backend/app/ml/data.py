import pandas as pd


def load_customer_churn_data():

    df = pd.read_csv("data/telco_churn.csv")

    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna()

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    df = df.rename(columns={
        "SeniorCitizen": "senior_citizen",
        "tenure": "tenure_months",
        "MonthlyCharges": "monthly_charge",
        "TotalCharges": "total_charges",
        "Contract": "contract_type",
        "PaymentMethod": "payment_method",
        "InternetService": "internet_service",
        "TechSupport": "tech_support",
        "OnlineSecurity": "online_security",
        "gender": "gender",
        "Churn": "churn"
    })

    return df