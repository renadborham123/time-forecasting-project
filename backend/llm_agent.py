"""Source-aware local QA and structured recommendation agent."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from functools import lru_cache

from .config import OLLAMA_URL, OLLAMA_MODEL
from .customer_service import profile, load_state
from .forecasting import forecast
from .explainability import explanation
from .decision_policy import eligible_actions, deterministic_recommendation, CATALOG

QA_SYSTEM_PROMPT = (
    "You answer questions about a learner using only the supplied verified context. "
    "Answer naturally in prose; do not require or emit JSON. Never invent numbers, history, "
    "preferences, evidence, or causal explanations. Keep sources distinct: the classifier "
    "estimates churn risk; the forecasting model projects usage; SHAP reports model "
    "contributions, not causes; preferences are recorded customer data. If evidence is "
    "missing, say so. Do not claim an action was executed."
)

RECOMMENDATION_SYSTEM_PROMPT = (
    "You select a retention recommendation using only the supplied verified context. "
    "Return only a JSON object with exactly the keys action, reason, and message. "
    "The action value must exactly match one code in eligible_actions. Do not invent "
    "evidence, numbers, preferences, or causal claims. SHAP values are model contributions, "
    "not causal effects. The recommendation is simulated and must not claim delivery."
)

_NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")


def _context(customer_id):
    p = profile(customer_id)
    fc = forecast(customer_id)
    sh = explanation(customer_id)
    actions = eligible_actions(customer_id)
    shap_rows = [{
        "feature": v["feature"], "label": v["label"], "contribution": round(v["value"], 6),
        "source": "SHAP model contribution; not a causal effect",
    } for v in sh["values"][:5]]
    context = {
        "customer_id": customer_id,
        "ml_model": {
            "classifier": p["metrics"]["classification"]["selected_model"],
            "churn_probability": round(p["churn_probability"], 4),
            "risk_tier": p["risk"],
            "prediction_cutoff_week": 28,
            "target": "will_churn_next_4_weeks (weeks 29–32)",
            "source": "churn classifier evaluated from behavior observed through week 28",
        },
        "forecast": {
            "selected_global_model": fc["selected_global_model"],
            "model_used": fc["model_used"],
            "current_usage_minutes": round(fc["current_weekly_usage_minutes"], 1),
            "week_4_forecast_minutes": round(fc["week_4_minutes"], 1),
            "expected_change_pct": round(fc["expected_change_pct"], 1),
            "source": "usage forecast from the last observed week in the 32-week history",
        },
        "shap": shap_rows,
        "customer_preferences": {
            "age_group": p["preferences"].get("age_group"),
            "favorite_topic": p["preferences"].get("favorite_topic"),
            "preferred_content_type": p["preferences"].get("preferred_content_type"),
            "preferred_feature": p["preferences"].get("preferred_feature"),
            "preferred_study_time": p["preferences"].get("preferred_study_time"),
            "historical_engagement_level": p["preferences"].get("historical_engagement_level"),
            "subscription_age_weeks_at_cutoff": p["features"].get("subscription_age_weeks"),
            "source": "recorded synthetic customer profile and week-28 feature snapshot",
        },
        "behavior_segment": {"label": p["segment"], "source": "exploratory K-Means behavioral profile"},
        "behavioral_drivers": {
            "usage_trend": round(p["features"].get("usage_trend", 0), 2),
            "sessions_trend": round(p["features"].get("sessions_trend", 0), 2),
            "recent_support_requests": int(p["features"].get("recent_support_requests", 0)),
            "source": "observed behavior through week 28",
        },
        "eligible_actions": [{"code": action, "label": CATALOG[action]} for action in actions],
    }
    context["verified_numeric_evidence"] = _numeric_evidence(context)
    return context


def _add_number(allowed, value, percent=False):
    if value is None or isinstance(value, bool):
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    for precision in (0, 1, 2, 3, 4, 5, 6):
        token = f"{number:.{precision}f}".rstrip("0").rstrip(".") if precision else str(round(number))
        allowed.add(token)
        if percent:
            allowed.add(token + "%")


def _numeric_evidence(context):
    allowed = set()
    ml, fc = context["ml_model"], context["forecast"]
    _add_number(allowed, ml["churn_probability"])
    _add_number(allowed, ml["churn_probability"] * 100, percent=True)
    _add_number(allowed, ml["prediction_cutoff_week"])
    _add_number(allowed, 4)
    for week in (29, 32):
        _add_number(allowed, week)
    _add_number(allowed, fc["current_usage_minutes"])
    _add_number(allowed, fc["week_4_forecast_minutes"])
    _add_number(allowed, fc["expected_change_pct"], percent=True)
    for item in context["shap"]:
        _add_number(allowed, item["contribution"])
    prefs = context["customer_preferences"]
    _add_number(allowed, prefs.get("subscription_age_weeks_at_cutoff"))
    age_group = str(prefs.get("age_group", ""))
    age_values = _NUMBER_PATTERN.findall(age_group)
    for match in age_values:
        allowed.add(match.rstrip("%"))
    if re.search(r"\d\s*[-–—]\s*\d", age_group) and len(age_values) >= 2:
        # A hyphen in a demographic range is a separator, not a negative sign.
        allowed.add("-" + age_values[-1].lstrip("-"))
    _add_number(allowed, context["behavioral_drivers"]["usage_trend"])
    _add_number(allowed, context["behavioral_drivers"]["sessions_trend"])
    _add_number(allowed, context["behavioral_drivers"]["recent_support_requests"])
    for match in _NUMBER_PATTERN.findall(context["customer_id"]):
        allowed.add(match.rstrip("%"))
    # Policy numbers are licensed by the fixed action catalog, not by ML evidence.
    for label in CATALOG.values():
        for match in _NUMBER_PATTERN.findall(label):
            allowed.add(match.rstrip("%"))
    return sorted(allowed)


def _numbers_are_grounded(text, context):
    known = set(context["verified_numeric_evidence"])
    for value in _NUMBER_PATTERN.findall(text):
        token = value.replace(",", "").rstrip("%")
        if token not in known and value not in known:
            return False
    return True


def _ollama(messages, structured=False):
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False,
               "options": {"temperature": 0.15}}
    if structured:
        payload["format"] = "json"
    request = urllib.request.Request(
        OLLAMA_URL + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        result = json.loads(response.read().decode())
    return result["message"]["content"]


def _evidence_lines(context):
    ml, fc = context["ml_model"], context["forecast"]
    prefs = context["customer_preferences"]
    top = context["shap"][:1]
    shap_evidence = (f"SHAP: {top[0]['label']} contributed {top[0]['contribution']:+.4f} to model output (not causal)."
                     if top else "SHAP evidence is unavailable.")
    return [
        f"Classifier estimate: {ml['churn_probability']:.0%} churn probability · {ml['risk_tier']} risk.",
        f"Forecast: {fc['expected_change_pct']:+.0f}% usage change by forecast week 4.",
        shap_evidence,
        f"Recorded preference: {prefs.get('favorite_topic')} · {prefs.get('preferred_content_type')}.",
        f"Behavior segment: {context['behavior_segment']['label']} (exploratory grouping).",
    ]


def recommend(customer_id, conversation=None, user_prompt="Recommend an appropriate retention action."):
    context = _context(customer_id)
    fallback = {**deterministic_recommendation(customer_id), "verified_evidence": _evidence_lines(context)}
    messages = [{"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT +
                 "\nVERIFIED CONTEXT:\n" + json.dumps(context, ensure_ascii=False)}]
    for item in (conversation or [])[-6:]:
        if item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": str(item.get("content", ""))[:1200]})
    messages.append({"role": "user", "content": user_prompt[:1200]})
    try:
        result = json.loads(_ollama(messages, structured=True))
        if not isinstance(result, dict) or set(result) != {"action", "reason", "message"}:
            return fallback
        action, reason, message = result["action"], str(result["reason"]), str(result["message"])
        allowed = [item["code"] for item in context["eligible_actions"]]
        if action not in allowed or not _numbers_are_grounded(reason + " " + message, context):
            return fallback
        return {
            "action": action, "action_label": CATALOG[action],
            "reason": reason[:1200] or fallback["reason"],
            "message": message[:1200] or fallback["message"],
            "provider": OLLAMA_MODEL, "fallback": False, "eligible_actions": allowed,
            "verified_evidence": _evidence_lines(context),
        }
    except Exception:
        return fallback


def _qa_fallback(context):
    ml, fc = context["ml_model"], context["forecast"]
    top = context["shap"][:3]
    shap_text = ", ".join(item["label"] for item in top) or "no SHAP values are available"
    return (
        f"The churn classifier estimates {ml['churn_probability']:.0%} churn probability and assigns "
        f"{ml['risk_tier']} risk using behavior observed through week 28. The target is churn during weeks 29–32. "
        f"The forecasting model projects usage from {fc['current_usage_minutes']:.0f} minutes to "
        f"{fc['week_4_forecast_minutes']:.0f} minutes by forecast week 4 ({fc['expected_change_pct']:+.0f}%). "
        f"SHAP's strongest listed model contributions include {shap_text}; they describe the classifier output, "
        "not causes. Recorded learner preferences are separate profile data."
    )


def _render_recommendation(result, context):
    ml, fc = context["ml_model"], context["forecast"]
    top = context["shap"][:1]
    shap_line = f"Strong SHAP contributor: {top[0]['label']} ({top[0]['contribution']:+.4f} model contribution)" if top else "SHAP evidence unavailable"
    evidence = [
        f"Churn probability: {ml['churn_probability']:.0%} ({ml['risk_tier']})",
        f"Usage forecast: {fc['expected_change_pct']:+.0f}% by week 4",
        shap_line,
        f"Favorite topic: {context['customer_preferences'].get('favorite_topic')}",
        f"Preferred content: {context['customer_preferences'].get('preferred_content_type')}",
    ]
    return "VERIFIED EVIDENCE\n- " + "\n- ".join(evidence) + "\n\nAI RECOMMENDATION\n" + \
        f"- Action: {result['action_label']}\n- Reason: {result['reason']}\n- Personalized message: {result['message']}"


def _quick_answer(customer_id, question):
    """Answer example questions directly from verified, customer-specific evidence."""
    context = _context(customer_id)
    question_lower = question.lower()
    action_question = any(word in question_lower for word in (
        "recommend", "should we", "what should", "action", "message", "draft",
        "intervention", "رسالة", "رساله", "نعمل", "توصية", "اقترح",
    ))
    if action_question:
        result = {**deterministic_recommendation(customer_id),
                  "verified_evidence": _evidence_lines(context)}
        if any(word in question_lower for word in ("message", "draft", "رسالة", "رساله")):
            text = (f"Suggested draft for {customer_id}:\n\n{result['message']}\n\n"
                    "This is a draft for review. No message has been sent.") if result["message"] else (
                        f"No retention message is recommended for {customer_id}. {result['reason']}"
                    )
        else:
            text = (f"For {customer_id}, I recommend: {result['action_label']}.\n\n"
                    f"{result['reason']}\n\nReview the recommendation before simulating the action.")
        return {"answer": text, "recommendation": result,
                "status": "Instant policy answer"}

    ml, fc = context["ml_model"], context["forecast"]
    if any(word in question_lower for word in ("forecast", "usage", "توقع", "استخدام")):
        names = {"exp_smoothing": "Exponential Smoothing", "naive": "Last-value baseline",
                 "arima": "ARIMA", "lag_regression": "Lag Regression"}
        method = names.get(fc["model_used"], fc["model_used"])
        text = (f"Here’s the forecast story for {customer_id}.\n\n"
                f"We read the learner’s 32 weeks of recorded activity. Four methods were compared "
                f"on earlier time windows; the saved winner is {names.get(fc['selected_global_model'], fc['selected_global_model'])}. "
                f"The method used for this learner is {method}, fitted to the most recent 16 weeks.\n\n"
                f"Weekly usage is forecast to move from "
                f"{fc['current_usage_minutes']:.0f} to {fc['week_4_forecast_minutes']:.0f} minutes "
                f"by forecast week 4 ({fc['expected_change_pct']:+.0f}%), compared with the average "
                "of the last four observed weeks.\n\n"
                "The solid line is history; the dashed line is the projection. The shaded range "
                "comes from errors in earlier forecast tests. It reminds us that the direction "
                "is an estimate, not a promise. The forecast projects usage; the separate classifier estimates churn risk.")
    else:
        contributions = "\n".join(
            f"• {item['label']}: {item['contribution']:+.3f} model contribution"
            for item in context["shap"][:3]
        )
        text = (f"{customer_id} has an estimated {ml['churn_probability']:.1%} churn probability "
                f"and is classified as {ml['risk_tier']}.\n\n"
                f"The strongest listed prediction drivers are:\n{contributions}\n\n"
                "Positive contributions raise the classifier score; negative contributions lower it. "
                "They describe model output, not causes. The classifier uses behavior through week 28 "
                "to estimate churn during weeks 29–32.")
    return {"answer": text, "recommendation": None,
            "status": "Instant evidence answer"}


@lru_cache(maxsize=64)
def pipeline_guide(stage, completed=False, customer_id=None):
    """Explain a known UI stage; the client owns and validates all action buttons."""
    bundle, history, _, _, metrics, predictions = load_state()
    if customer_id:
        profile(customer_id)  # Reject unknown IDs before attempting a provider call.
    count, weeks = len(predictions), int(history.week.max())
    classifier = metrics["classification"]["selected_model"]
    selected = metrics["forecasting"]["selected_model"]
    names = {"exp_smoothing": "Exponential Smoothing", "naive": "Last-value baseline",
             "arima": "ARIMA", "lag_regression": "Lag Regression"}
    instructions = [
        f"Load {count} synthetic learners, each with {weeks} recorded weeks. Each dot represents one learner. We’ll group their habits, estimate churn risk, then forecast usage.",
        "K-Means assigns behavior groups. Watch the dots take their segment colors, then gather into five separate clouds. Their spacing is for readability; the groups can overlap in real behavior.",
        f"{classifier} estimates churn risk using behavior through week 28. The dots stay in their behavior groups; their colors change to show risk. Next we’ll forecast a learner’s usage.",
        f"Read the {weeks}-week history, then forecast the next four weeks. Four methods were evaluated on earlier windows; {names.get(selected, selected)} was selected and fits the latest 16 weeks. Look at the dashed line and its shaded uncertainty range.",
        "Choose a suggested question to understand the forecast, review risk evidence, or select an eligible next step. You can also type a question. Recommendations and message drafts stay inside the conversation; actions are simulated.",
    ]
    if completed and stage == 1:
        instructions[1] = "The five behavior groups are ready. Same segment color, same outlined cloud. Next, classify churn risk for the same learners; the group positions will stay in place."
    if completed and stage == 2:
        instructions[2] = "The classifier’s risk estimates are ready. Choose a dot to use that learner, or explore C003’s forecast as an example. Forecasting will project usage for the next four weeks."
    if stage == 3 and completed and customer_id:
        result = forecast(customer_id)
        instructions[3] = (
            f"For {customer_id}, {names.get(result['model_used'], result['model_used'])} projects "
            f"{result['expected_change_pct']:+.0f}% usage change by forecast week 4. "
            "The solid line is the observed history; the dashed line is the forecast. "
            "The shaded range shows uncertainty. Open the assistant to discuss this result."
        )
    text = instructions[stage]
    facts = {"stage": stage, "completed": completed, "verified_explanation": text,
             "next_action": ["Load Data", "Classify risk" if completed else "Segment learners",
                             "Explore forecasts" if completed else "Classify risk",
                             "Open assistant" if completed else "Run forecast", "Ask about the forecast"][stage]}
    try:
        explanation_text = _ollama([
            {"role": "system", "content": "You are a friendly UI guide. Rewrite only the supplied verified explanation in 35–55 words of simple English. Describe what the user sees and the next action. Do not add facts, numbers, causal claims, hidden reasoning, markdown, URLs, or claims that an action ran. Never invent a forecast or a delivered message."},
            {"role": "user", "content": json.dumps(facts)},
        ], structured=False).strip()
        allowed_numbers = set(_NUMBER_PATTERN.findall(text)) | {"5", "4", "32", "28", str(count)}
        if not explanation_text or len(explanation_text.split()) > 75 or any(
            number not in allowed_numbers for number in _NUMBER_PATTERN.findall(explanation_text)
        ):
            raise ValueError("Unverified guide output")
        return {"explanation": explanation_text, "provider": OLLAMA_MODEL, "fallback": False}
    except Exception:
        return {"explanation": text, "provider": "verified workflow", "fallback": True}


def answer(customer_id, question, conversation=None, quick=False):
    if quick:
        return _quick_answer(customer_id, question)
    question_lower = question.lower()
    if any(k in question_lower for k in ("recommend", "should we", "what should", "action", "write a message", "personalized message", "intervention")):
        context = _context(customer_id)
        result = recommend(customer_id, conversation, question)
        status = "Local LLM unavailable — deterministic recommendation mode." if result["fallback"] else f"Local agent · {result['provider']}"
        return {"answer": _render_recommendation(result, context), "recommendation": result, "status": status}

    context = _context(customer_id)
    fallback_text = _qa_fallback(context)
    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT +
                 "\nVERIFIED CONTEXT:\n" + json.dumps(context, ensure_ascii=False)}]
    for item in (conversation or [])[-6:]:
        if item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": str(item.get("content", ""))[:1200]})
    messages.append({"role": "user", "content": question[:1200]})
    try:
        text = _ollama(messages, structured=False)
        if not text.strip() or not _numbers_are_grounded(text, context):
            raise ValueError("QA answer contains unsupported numeric evidence")
        return {"answer": text, "recommendation": None, "status": f"Local agent · {OLLAMA_MODEL}"}
    except Exception:
        return {"answer": fallback_text, "recommendation": None,
                "status": "Local LLM unavailable — deterministic evidence summary."}
