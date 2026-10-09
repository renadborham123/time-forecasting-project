"""Supervised churn classification results and dynamic risk distribution."""
import numpy as np
from .customer_service import customers, load_state
from .config import risk_label

RISK_LEVELS=("STABLE","WATCHLIST","AT RISK","CRITICAL")

def classify_customers():
    bundle, _, _, features, metrics, _ = load_state()
    X = features[bundle["features"]].replace([np.inf, -np.inf], 0).fillna(0)
    probabilities = dict(zip(features.index, bundle["classifier"].predict_proba(X)[:, 1]))
    rows=customers()
    for row in rows:
        probability = float(probabilities[row["customer_id"]])
        row.update(churn_probability=probability, risk=risk_label(probability))
    counts={risk:sum(row["risk"]==risk for row in rows) for risk in RISK_LEVELS}
    return {"stage":"classified","model":metrics["classification"]["selected_model"],"counts":counts,"customers":rows}
