"""Shared loaded data and customer profile access for application services."""
from __future__ import annotations
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_DIR, DATA_DIR, risk_label
from scripts.export_models import engineer_features

@lru_cache(maxsize=1)
def load_state():
    bundle=joblib.load(MODEL_DIR/"pipeline.joblib")
    history=pd.read_csv(DATA_DIR/"customer_weekly_history.csv")
    profiles=pd.read_csv(DATA_DIR/"customer_profiles.csv").set_index("customer_id")
    feature_frame=engineer_features(history).set_index("customer_id")
    metrics=json.loads((MODEL_DIR/"metrics.json").read_text(encoding="utf-8"))
    X=feature_frame[bundle["features"]].replace([np.inf,-np.inf],0).fillna(0)
    probabilities=bundle["classifier"].predict_proba(X)[:,1]
    scaled=bundle["scaler"].transform(X)
    clusters=bundle["kmeans"].predict(scaled)
    embedding=bundle["embedding"].transform(scaled)
    predictions=pd.DataFrame({"customer_id":feature_frame.index,"churn_probability":probabilities,"cluster":clusters,"embedding_x":embedding[:,0],"embedding_y":embedding[:,1]}).set_index("customer_id")
    predictions["risk"]=[risk_label(float(x)) for x in probabilities]
    predictions["segment"]=[bundle["cluster_names"][int(x)] for x in clusters]
    return bundle,history,profiles,feature_frame,metrics,predictions

def customers():
    _,history,_,features,_,pred=load_state()
    recent=history.sort_values("week").groupby("customer_id").tail(4).groupby("customer_id").agg(recent_minutes=("usage_minutes","mean"))
    out=pred.join(recent)
    prior=(features.recent_usage_minutes-features.usage_trend).replace(0,np.nan)
    out["usage_change_pct"]=(features.usage_trend/prior*100).replace([np.inf,-np.inf],np.nan).fillna(0)
    out["recent_minutes"]=recent.recent_minutes
    out.index.name="customer_id"
    return out.reset_index().replace({np.nan:None}).to_dict("records")

def _get_customer(customer_id):
    bundle,history,profiles,features,metrics,pred=load_state()
    if customer_id not in features.index: raise KeyError(customer_id)
    return bundle,history,profiles,features.loc[customer_id],metrics,pred.loc[customer_id]

def profile(customer_id):
    bundle,history,profiles,features,metrics,pred=_get_customer(customer_id)
    series=history[history.customer_id==customer_id].sort_values("week")
    return {"customer_id":customer_id,"risk":pred.risk,"churn_probability":float(pred.churn_probability),"segment":pred.segment,"cluster":int(pred.cluster),"embedding":{"x":float(pred.embedding_x),"y":float(pred.embedding_y)},"features":{k:float(features[k]) for k in bundle["features"]},"preferences":profiles.loc[customer_id].to_dict(),"history":series.to_dict("records"),"metrics":metrics}
