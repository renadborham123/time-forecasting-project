from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import FRONTEND_DIR
from .customer_service import customers, profile, load_state
from .forecasting import forecast
from .explainability import explanation
from .segmentation import segment_customers
from .churn import classify_customers
from .decision_policy import eligible_actions, CATALOG
from .llm_agent import answer, recommend
from .schemas import ChatRequest, ActionRequest

app=FastAPI(title="ChurnScope AI",description="Academic customer churn intelligence demo",version="1.0.0")
actions=[]

@app.on_event("startup")
def startup(): load_state()

@app.get("/")
def index(): return FileResponse(FRONTEND_DIR/"index.html")

app.mount("/static",StaticFiles(directory=FRONTEND_DIR),name="static")

@app.get("/api/customers")
def get_customers(): return customers()

@app.post("/api/segment")
def segment(): return segment_customers()

@app.post("/api/classify")
def classify(): return classify_customers()

@app.get("/api/customer/{customer_id}")
def get_customer(customer_id:str):
    try:return profile(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found")

@app.get("/api/customer/{customer_id}/forecast")
def get_forecast(customer_id:str):
    try:return forecast(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found")

@app.get("/api/customer/{customer_id}/explanation")
def get_explanation(customer_id:str):
    try:return explanation(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found")

@app.post("/api/customer/{customer_id}/chat")
def chat(customer_id:str,body:ChatRequest):
    try: profile(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found")
    return answer(customer_id,body.question,body.conversation)

@app.post("/api/customer/{customer_id}/recommend")
def get_recommendation(customer_id:str):
    try:return recommend(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found")

@app.get("/api/customer/{customer_id}/eligible-actions")
def get_eligible_actions(customer_id:str):
    try:return {"customer_id":customer_id,"eligible_actions":eligible_actions(customer_id)}
    except KeyError:raise HTTPException(404,"Customer not found")

@app.post("/api/customer/{customer_id}/execute-action")
def execute_action(customer_id:str,body:ActionRequest):
    try: allowed=eligible_actions(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found")
    if body.action not in allowed: raise HTTPException(400,"Action is not eligible for this customer")
    record={"customer_id":customer_id,"action":body.action,"label":CATALOG[body.action],"status":"scheduled (simulated)"}
    actions.append(record)
    return {"message":"Action successfully scheduled. (Simulation only.)","record":record,"history":actions[-20:]}

@app.get("/api/model-metrics")
def model_metrics(): return load_state()[4]

@app.get("/api/action-history")
def action_history(): return actions
