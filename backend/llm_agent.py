"""Grounded Ollama provider with deterministic fallback."""
from __future__ import annotations
import json, urllib.request, urllib.error
from .config import OLLAMA_URL, OLLAMA_MODEL
from .customer_service import profile
from .forecasting import forecast
from .explainability import explanation
from .decision_policy import eligible_actions, deterministic_recommendation, CATALOG

SYSTEM=("You are a Customer Retention Decision Agent for a subscription learning platform. "
"Use only the supplied structured evidence. Never invent churn probabilities, behavior, forecasts, SHAP evidence, history, or preferences. "
"Copy numeric evidence exactly. Separate measured facts from your recommendation. If evidence is insufficient, say so. "
"Choose an action only from eligible_actions. Return only valid JSON with keys action, reason, message. "
"Do not include hidden reasoning or claim any action was executed.")

def _context(customer_id):
    p=profile(customer_id); fc=forecast(customer_id); sh=explanation(customer_id)
    return {"customer_id":customer_id,"churn":{"probability":round(p["churn_probability"],4),"risk":p["risk"]},"forecast":{"current_weekly_usage_minutes":round(fc["current_weekly_usage_minutes"],1),"week_4_minutes":round(fc["week_4_minutes"],1),"expected_change_pct":round(fc["expected_change_pct"],1)},"shap_evidence":[{"feature":v["feature"],"effect":round(v["value"],5)} for v in sh["values"][:5]],"preferences":p["preferences"],"eligible_actions":eligible_actions(customer_id)}

def recommend(customer_id,conversation=None,user_prompt="Recommend an appropriate retention action."):
    fallback=deterministic_recommendation(customer_id)
    context=_context(customer_id)
    messages=[{"role":"system","content":SYSTEM+"\nVERIFIED CONTEXT:\n"+json.dumps(context,ensure_ascii=False)}]
    for item in (conversation or [])[-6:]:
        if item.get("role") in ("user","assistant"): messages.append({"role":item["role"],"content":str(item.get("content",""))[:1200]})
    messages.append({"role":"user","content":user_prompt[:1200]})
    try:
        payload=json.dumps({"model":OLLAMA_MODEL,"messages":messages,"stream":False,"format":"json","options":{"temperature":0.15}}).encode()
        req=urllib.request.Request(OLLAMA_URL+"/api/chat",data=payload,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=45) as response: raw=json.loads(response.read().decode())
        result=json.loads(raw["message"]["content"])
        action=result.get("action")
        allowed=context["eligible_actions"]
        if action not in allowed: return fallback
        reason=str(result.get("reason", ""))[:1200]; message=str(result.get("message", ""))[:1200]
        # Reject unsupported numeric claims instead of presenting unverified output as evidence.
        import re
        known={str(v) for v in context["churn"].values()} | {str(v) for v in context["forecast"].values()}
        known.update({str(round(context["churn"]["probability"]*100))})
        numeric=re.findall(r"(?<!\w)-?\d+(?:\.\d+)?%?",reason+" "+message)
        if any(n.rstrip("%").strip(".") not in {x.rstrip("%").strip(".") for x in known} for n in numeric): return fallback
        return {"action":action,"action_label":CATALOG[action],"reason":reason or fallback["reason"],"message":message or fallback["message"],"provider":OLLAMA_MODEL,"fallback":False,"eligible_actions":allowed}
    except Exception:
        return fallback

def answer(customer_id,question,conversation=None):
    q=question.lower()
    if any(k in q for k in ("recommend","should we","what should","action","write a message","personalized message")):
        result=recommend(customer_id,conversation,question)
        status="Local LLM unavailable — deterministic recommendation mode." if result["fallback"] else f"Local agent · {result['provider']}"
        return {"answer":f"{result['reason']}\n\nRecommended action: {result['action_label']}.\n\n{result['message']}","recommendation":result,"status":status}
    p=profile(customer_id); fc=forecast(customer_id); sh=explanation(customer_id)
    fallback_text=(f"The churn classifier estimates {p['churn_probability']:.0%} churn probability and assigns {p['risk']} risk. "
                   f"The strongest local SHAP contributions are {', '.join(v['label'] for v in sh['values'][:3])}. "
                   f"The selected forecast model projects weekly usage from {fc['current_weekly_usage_minutes']:.0f} minutes now to {fc['week_4_minutes']:.0f} minutes in week 4 ({fc['expected_change_pct']:+.0f}%).")
    try:
        context=_context(customer_id)
        payload=json.dumps({"model":OLLAMA_MODEL,"messages":[{"role":"system","content":SYSTEM+"\nVERIFIED CONTEXT:\n"+json.dumps(context,ensure_ascii=False)},{"role":"user","content":question[:1200]}],"stream":False}).encode()
        req=urllib.request.Request(OLLAMA_URL+"/api/chat",data=payload,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=45) as response: text=json.loads(response.read().decode())["message"]["content"]
        import re
        allowed_nums={str(round(p["churn_probability"]*100)),str(round(fc["current_weekly_usage_minutes"])),str(round(fc["week_4_minutes"])),str(round(fc["expected_change_pct"]))}
        nums=re.findall(r"(?<!\w)-?\d+(?:\.\d+)?%?",text)
        if any(n.rstrip("%").strip(".") not in allowed_nums for n in nums): raise ValueError("Unsupported numeric output")
        return {"answer":text,"recommendation":None,"status":f"Local agent · {OLLAMA_MODEL}"}
    except Exception:
        return {"answer":fallback_text,"recommendation":None,"status":"Local LLM unavailable — deterministic recommendation mode."}
