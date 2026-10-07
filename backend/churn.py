"""Supervised churn classification results and dynamic risk distribution."""
from .customer_service import customers

RISK_LEVELS=("STABLE","WATCHLIST","AT RISK","CRITICAL")

def classify_customers():
    rows=customers()
    counts={risk:sum(row["risk"]==risk for row in rows) for risk in RISK_LEVELS}
    return {"stage":"classified","counts":counts,"customers":rows}
