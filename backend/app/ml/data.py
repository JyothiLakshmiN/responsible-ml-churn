import numpy as np
import pandas as pd


def generate_customer_data(n_samples=10000, random_state=42):

    np.random.seed(random_state)

    age = np.random.normal(45, 15, n_samples).clip(18, 90)

    tenure_months = np.random.exponential(24, n_samples).clip(1, 120)

    monthly_charge = np.random.normal(65, 20, n_samples).clip(20, 150)

    num_products = np.random.randint(1, 6, n_samples)

    support_calls = np.random.poisson(2, n_samples)

    gender = np.random.choice(['Male', 'Female'], n_samples)

    contract_type = np.random.choice(
        ['Month-to-Month', 'One Year', 'Two Year'],
        n_samples,
        p=[0.5, 0.3, 0.2]
    )

    payment_method = np.random.choice(
        ['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'],
        n_samples
    )

    churn_prob = (
        0.3
        - 0.003 * tenure_months
        + 0.05 * support_calls
        + 0.003 * monthly_charge
        - 0.05 * num_products
        + 0.2 * (contract_type == 'Month-to-Month')
        + np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)

    churn = (
        np.random.uniform(0, 1, n_samples) < churn_prob
    ).astype(int)

    df = pd.DataFrame({
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charge': monthly_charge,
        'num_products': num_products,
        'support_calls': support_calls,
        'gender': gender,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'churn': churn
    })

    return df