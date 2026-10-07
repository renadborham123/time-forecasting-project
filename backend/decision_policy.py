"""Deterministic eligibility layer and safe action catalog."""
from .customer_service import profile

CATALOG={"NO_ACTION":"No retention contact","PERSONALIZED_CONTENT_RECOMMENDATION":"Personalized content recommendation","PREMIUM_FEATURE_TRIAL":"7-day premium feature trial","ONE_FREE_MONTH":"One free month","STUDY_PLAN_RECOMMENDATION":"Personalized study plan","RE_ENGAGEMENT_MESSAGE":"Re-engagement message","HUMAN_SUPPORT_OUTREACH":"Human support outreach","LEARNING_PATH_RECOMMENDATION":"Learning path recommendation"}

def eligible_actions(customer_id):
    p=profile(customer_id); f=p["features"]; tier=p["risk"]; prefs=p["preferences"]
    if tier=="STABLE": return ["NO_ACTION","PERSONALIZED_CONTENT_RECOMMENDATION"]
    eligible=["RE_ENGAGEMENT_MESSAGE","PERSONALIZED_CONTENT_RECOMMENDATION","STUDY_PLAN_RECOMMENDATION","LEARNING_PATH_RECOMMENDATION"]
    if f["recent_support_requests"]>=2: eligible.insert(0,"HUMAN_SUPPORT_OUTREACH")
    if tier in ("AT RISK","CRITICAL") and prefs["historical_engagement_level"]=="High": eligible.insert(0,"PREMIUM_FEATURE_TRIAL")
    if tier=="CRITICAL" and f["recent_support_requests"]<2: eligible.insert(1,"ONE_FREE_MONTH")
    return list(dict.fromkeys(eligible))

def deterministic_recommendation(customer_id):
    p=profile(customer_id); f=p["features"]; allowed=eligible_actions(customer_id); prefs=p["preferences"]
    if "HUMAN_SUPPORT_OUTREACH" in allowed: action="HUMAN_SUPPORT_OUTREACH"
    elif "PREMIUM_FEATURE_TRIAL" in allowed: action="PREMIUM_FEATURE_TRIAL"
    elif "PERSONALIZED_CONTENT_RECOMMENDATION" in allowed and f["usage_trend"]<0: action="PERSONALIZED_CONTENT_RECOMMENDATION"
    elif p["risk"]=="STABLE": action="NO_ACTION"
    else: action="STUDY_PLAN_RECOMMENDATION"
    topic=prefs["favorite_topic"]; content=prefs["preferred_content_type"]
    msg=f"Hi! We picked out {topic} {content.lower()} that fit your interests. You can explore them whenever it works for you."
    reason=f"The model assigns {p['churn_probability']:.0%} churn probability ({p['risk']}). The leading measured context is {('recent inactivity and declining engagement' if f['usage_trend']<0 else 'current engagement patterns')}; the customer profile lists {topic} as a favorite topic."
    return {"action":action,"action_label":CATALOG[action],"reason":reason,"message":msg,"provider":"deterministic","fallback":True,"eligible_actions":allowed}
