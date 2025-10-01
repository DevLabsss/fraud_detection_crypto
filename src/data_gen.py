
import numpy as np
import pandas as pd

def generate_synthetic_strong(n_rows=8000, random_state=42):
    rng = np.random.default_rng(random_state)
    amount = np.round(np.exp(rng.normal(3.6, 1.0, n_rows)) * 10, 2)
    tx_freq = rng.poisson(3, size=n_rows)
    account_age = rng.integers(1, 365*3, size=n_rows)
    is_weekend = rng.integers(0, 2, size=n_rows)

    risk_score = (
        0.9*(amount > np.percentile(amount, 85)).astype(float) +
        0.7*(account_age < 60).astype(float) +
        0.6*(tx_freq >= 7).astype(float) +
        0.3*(is_weekend == 1).astype(float)
    ).astype(float)

    y = (risk_score > 0.6).astype(int)
    flip = rng.binomial(1, 0.02, n_rows).astype(bool)  # 2% noise
    y[flip] = 1 - y[flip]

    return pd.DataFrame({
        "amount": amount,
        "transaction_freq_24h": tx_freq,
        "account_age_days": account_age,
        "is_weekend": is_weekend,
        "risk_score": risk_score,
        "is_fraud": y
    })
