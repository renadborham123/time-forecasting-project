"""Preference-aware eligibility and deterministic retention policy."""
from .customer_service import profile

CATALOG = {
    "NO_ACTION": "No retention contact",
    "PERSONALIZED_CONTENT_RECOMMENDATION": "Personalized content recommendation",
    "PREMIUM_FEATURE_TRIAL": "7-day premium feature trial",
    "ONE_FREE_MONTH": "One free month",
    "STUDY_PLAN_RECOMMENDATION": "Personalized study plan",
    "RE_ENGAGEMENT_MESSAGE": "Re-engagement message",
    "HUMAN_SUPPORT_OUTREACH": "Human support outreach",
    "LEARNING_PATH_RECOMMENDATION": "Learning path recommendation",
}


def eligible_actions(customer_id):
    p = profile(customer_id)
    f, prefs, tier = p["features"], p["preferences"], p["risk"]
    try:
        from .forecasting import forecast as customer_forecast
        forecast_decline = customer_forecast(customer_id)["expected_change_pct"] <= -20
    except Exception:
        forecast_decline = False
    declining = (forecast_decline or f["usage_trend"] <= -15
                 or f.get("sessions_trend", 0) <= -1.5
                 or f.get("usage_slope_recent", 0) <= -3)
    support_needed = f["recent_support_requests"] >= 2

    # A stable learner is left alone unless repeated support requests warrant
    # a human response. Non-promotional content stays optional and relevant.
    if tier == "STABLE":
        actions = ["NO_ACTION", "PERSONALIZED_CONTENT_RECOMMENDATION"]
        if support_needed:
            actions.insert(0, "HUMAN_SUPPORT_OUTREACH")
        return actions

    actions = ["RE_ENGAGEMENT_MESSAGE", "PERSONALIZED_CONTENT_RECOMMENDATION", "STUDY_PLAN_RECOMMENDATION"]
    if declining and prefs.get("favorite_topic"):
        actions.append("LEARNING_PATH_RECOMMENDATION")
    if prefs.get("preferred_content_type") == "Projects" or prefs.get("preferred_feature") == "Projects":
        actions.append("LEARNING_PATH_RECOMMENDATION")

    # Repeated help requests take priority and suppress promotions.
    if support_needed:
        return list(dict.fromkeys(["HUMAN_SUPPORT_OUTREACH", *actions]))

    high_engagement = prefs.get("historical_engagement_level") == "High"
    if tier in ("AT RISK", "CRITICAL") and high_engagement and declining:
        if prefs.get("preferred_feature") == "Interactive Labs":
            actions.insert(0, "PREMIUM_FEATURE_TRIAL")
        elif prefs.get("preferred_content_type") == "Interactive Labs":
            actions.insert(0, "PREMIUM_FEATURE_TRIAL")
    if tier == "CRITICAL" and high_engagement:
        actions.extend(["ONE_FREE_MONTH", "RE_ENGAGEMENT_MESSAGE"])
    return list(dict.fromkeys(actions))


def deterministic_recommendation(customer_id):
    p = profile(customer_id)
    f, prefs = p["features"], p["preferences"]
    allowed = eligible_actions(customer_id)
    if "HUMAN_SUPPORT_OUTREACH" in allowed:
        action = "HUMAN_SUPPORT_OUTREACH"
    elif p["risk"] == "STABLE":
        action = "NO_ACTION"
    elif "PREMIUM_FEATURE_TRIAL" in allowed:
        action = "PREMIUM_FEATURE_TRIAL"
    elif "LEARNING_PATH_RECOMMENDATION" in allowed and (prefs.get("preferred_content_type") == "Projects" or prefs.get("preferred_feature") == "Projects"):
        action = "LEARNING_PATH_RECOMMENDATION"
    elif "PERSONALIZED_CONTENT_RECOMMENDATION" in allowed and (f["usage_trend"] < 0 or f.get("sessions_trend", 0) < 0):
        action = "PERSONALIZED_CONTENT_RECOMMENDATION"
    else:
        action = "STUDY_PLAN_RECOMMENDATION"

    topic = prefs.get("favorite_topic", "your favorite topics")
    content = prefs.get("preferred_content_type", "learning content")
    feature = prefs.get("preferred_feature", "")
    if action == "HUMAN_SUPPORT_OUTREACH":
        message = "Hi! We noticed your recent support requests and would be glad to help. Reply with the issue you would like us to look at."
        reason = "Repeated recent support requests make personal assistance the first eligible response; promotions are withheld."
    elif action == "NO_ACTION":
        message = ""
        reason = "The classifier currently places this learner in the STABLE tier, so the policy recommends no retention contact."
    elif action == "PREMIUM_FEATURE_TRIAL":
        message = f"Hi! Since you prefer {feature or content}, you can try the premium learning features and explore more {topic} practice."
        reason = "The learner has historically high engagement, a declining recent trend, and a recorded preference that matches the trial."
    elif action == "LEARNING_PATH_RECOMMENDATION":
        message = f"Hi! We put together a project-based learning path around {topic} for you to explore at your own pace."
        reason = "The learning path matches the recorded project preference and the measured change in recent engagement."
    else:
        message = f"Hi! We picked out {topic} {content.lower()} that fit your interests. Explore them whenever it works for you."
        reason = f"The classifier assigns {p['churn_probability']:.0%} churn probability ({p['risk']}); this personalized option uses the learner's recorded {topic} preference."
    return {
        "action": action,
        "action_label": CATALOG[action],
        "reason": reason,
        "message": message,
        "provider": "deterministic",
        "fallback": True,
        "eligible_actions": allowed,
    }
