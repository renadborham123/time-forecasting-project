"""Behavioral segmentation results for the customer intelligence map."""
from .customer_service import customers

def segment_customers():
    rows=customers()
    return {"stage":"segmented","count":len(rows),"clusters":len({row["segment"] for row in rows}),"customers":rows}
