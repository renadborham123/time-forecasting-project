"""Behavioral segmentation results for the customer intelligence map."""
import numpy as np
from .customer_service import customers, load_state

def segment_customers():
    bundle, _, _, features, _, _ = load_state()
    X = features[bundle["features"]].replace([np.inf, -np.inf], 0).fillna(0)
    labels = bundle["kmeans"].predict(bundle["scaler"].transform(X))
    assignments = dict(zip(features.index, labels))
    rows = customers()
    for row in rows:
        cluster = int(assignments[row["customer_id"]])
        row.update(cluster=cluster, segment=bundle["cluster_names"][cluster])
    return {"stage":"segmented","model":"K-Means","count":len(rows),"clusters":len(set(labels)),"customers":rows}
