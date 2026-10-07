"""Customer-level SHAP evidence for the selected churn classifier."""
from functools import lru_cache
import numpy as np
import pandas as pd
import shap
from .customer_service import _get_customer, load_state

FEATURE_LABELS={"days_since_last_activity":"Recent inactivity","sessions_trend":"Declining sessions","usage_trend":"Usage trend","recent_sessions":"Recent sessions","recent_usage_minutes":"Recent usage","active_days_trend":"Active days trend","recent_completed_lessons":"Lesson completion","recent_support_requests":"Support requests","engagement_volatility":"Engagement volatility","subscription_age_weeks":"Subscription history"}

@lru_cache(maxsize=1)
def _explainer():
    bundle,*_=load_state(); model=bundle["classifier"]
    if hasattr(model,"named_steps"):
        background=model.named_steps["scale"].transform(bundle["shap_background"])
        return shap.LinearExplainer(model.named_steps["model"],background)
    return shap.Explainer(model,bundle["shap_background"])

def explanation(customer_id):
    bundle,_,_,features,_,pred=_get_customer(customer_id)
    X=pd.DataFrame([features[bundle["features"]]],columns=bundle["features"])
    model=bundle["classifier"]
    values=_explainer()(model.named_steps["scale"].transform(X) if hasattr(model,"named_steps") else X)
    vals=np.asarray(values.values)
    vals=vals[0,:,1] if vals.ndim==3 else vals[0]
    ranked=sorted(zip(bundle["features"],vals),key=lambda pair:abs(float(pair[1])),reverse=True)[:8]
    return {"customer_id":customer_id,"method":"SHAP","base_value":float(np.asarray(values.base_values).reshape(-1)[-1]),"values":[{"feature":key,"label":FEATURE_LABELS.get(key,key.replace('_',' ').title()),"value":float(value)} for key,value in ranked],"probability":float(pred.churn_probability)}
