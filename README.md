# ChurnScope AI

**Forecast → Classify → Explain → Act**

ChurnScope AI is an academic demonstration for a subscription learning platform. It uses fixed-seed synthetic data to explore learner behavior, estimate a defined churn outcome, forecast usage, explain classifier scores, and select policy-eligible retention suggestions.

## What is predicted?

- **Classification:** Will this learner churn in the next four weeks? We observe behavior through week 28 and predict whether cancellation occurs during weeks 29–32. Classifier features use observations through week 28 only; the customer-level target is stored separately in `data/customer_churn_targets.csv`.
- **Forecast:** What will weekly usage look like over the next four weeks? The app projects weeks 33–36 from the available 32-week history. Walk-forward evaluation separately uses cutoffs 20 → 21–24, 24 → 25–28, and 28 → 29–32.
- **Clustering:** Which learners show similar historical behavior? K-Means provides complete, stable coverage for visualization and behavioral profiling. DBSCAN is retained as a density-based comparison.
- **SHAP:** Which features most influenced the classifier's prediction? Contributions explain model output; they are not percentages, causes, or proof of causality.
- **Agent:** What intervention is reasonable based on verified evidence and learner preferences? The agent chooses only from a deterministic eligible-action list. Simulated execution sends no real messages.

Clustering, classification, forecasting, and LLM reasoning are separate tasks. The grouped map illustrates segment membership; the optional behavior-similarity view uses PCA coordinates. Risk appearance comes from the churn classifier.

## Quick start

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8000](http://localhost:8000). `run.py` starts FastAPI and serves the frontend from the same process. If model artifacts are absent, it first trains and exports the shared pipeline.

To regenerate fixed-seed data and models:

```powershell
python scripts/generate_data.py
python scripts/export_models.py
```

## Demo walkthrough

1. **Load Data:** explicitly fetch the system's synthetic sample: 300 learners with 32 weeks of activity. The dots appear in a randomly scattered view. Reloading the sample changes the visual placement, not the saved dataset or evaluation.
2. **Segment:** run the saved K-Means model. Dots receive segment colors progressively, then move into irregular outlined clouds. Select a group to highlight it. Cloud spacing illustrates membership; **Behavior similarity** uses the actual PCA coordinates instead.
3. **Classify:** run the saved churn classifier. Group positions remain fixed while dots receive risk colors. Select a dot to choose a learner for forecasting.
4. **Forecast:** follow the simple history → evaluated method → next four weeks story. The saved method winner is fitted to the selected learner. Review weeks 33–36, uncertainty, optional SHAP contributions, and preferences. C001–C004 provide contrasting example histories.
5. **Ask & act:** use the centered chat to explain the forecast, understand risk, or choose a next step. Suggested questions answer instantly from verified evidence and policy; custom questions use Ollama with a deterministic fallback. Recommendations and drafts appear inside the conversation.

The single primary button advances the workflow. Three named guides follow the relevant UI: the green **Data agent** handles loading and segmentation, the blue **Risk agent** explains classification, and the pink **Forecast agent** covers forecasting and the conversation. Their soft round characters blink, glance, and smile; each stage opens a matching tinted popup automatically. Close and reopen it using the character. Popup actions invoke the same workflow button. The optional LLM rewrites verified stage context; unavailable or unverified output falls back to an explicitly labeled workflow explanation. Model activity indicators distinguish inference from visualizing its result. Reduced-motion preferences skip dot and character animations.

**Simulate action** records an eligible in-memory event only; it never sends messages. **Model details** opens evaluation separately from the workflow.

## Dataset and methodology

- `data/customer_weekly_history.csv`: 300 learners × 32 weeks of sessions, active days, usage, lessons, labs, searches, support requests, inactivity, and subscription age. Each subscription starts at one age and increases weekly.
- `data/customer_profiles.csv`: separately generated preferences and engagement profile.
- `data/customer_churn_targets.csv`: one target per customer, derived from a synthetic cancellation event during weeks 29–32.
- `data/generated/customer_predictions.csv`: fitted probabilities, behavioral segments, and 2D map coordinates.
- Churn propensity is generated from behavior available through the week-28 cutoff with overlapping stochastic outcomes. Weeks 29–32 are never used to build classifier features.
- A customer-level holdout compares Logistic Regression and Random Forest using accuracy, precision, recall, F1, and ROC-AUC. The selected classifier is refit on all week-28 snapshots for the demo; risk thresholds are centralized in `backend/config.py`.
- K-Means cluster labels use observed cluster profiles: recent sessions and usage, trends, volatility, inactivity, active days, and lesson completion. The silhouette score is reported as evidence of overlap, not hidden. DBSCAN may mark many points as noise. K-Means is preferred for full map coverage and stable profiling, not claimed as objectively superior.
- Forecast candidates are naive, damped Exponential Smoothing, ARIMA(1,1,0), and Ridge lag regression. They are compared with chronological walk-forward validation. MAE selects the model; RMSE and sMAPE provide secondary evidence. MAPE is unstable near zero usage and is not the selection metric. Results are in `models/metrics.json`.
- Customer forecasts use a robust fallback chain: selected global model, Exponential Smoothing, then naive. The API reports both the global selection and the method actually used for that customer.
- The 90th percentile absolute walk-forward error is an empirical forecast band, not a formally calibrated prediction interval.

## Evidence and agent behavior

The QA prompt answers evidence questions in natural language. A separate recommendation prompt requires JSON with `action`, `reason`, and `message`. The action must come from the eligible action catalog. Numeric evidence is checked against the classifier, forecast, SHAP contributions, recorded profile, and action policy so a catalog phrase such as “7-day trial” remains valid.

Preferences affect action eligibility. Declining engagement and matching content preferences can enable personalized content or a learning path; repeated support requests prioritize human support and suppress promotions; stable learners receive no contact by default. SHAP describes model contributions and cannot establish causes. Recommendation wording remains separate from verified evidence in the interface.

The optional local provider uses Ollama:

```powershell
ollama pull gpt-oss:20b
```

Defaults are `OLLAMA_URL=http://localhost:11434` and `OLLAMA_MODEL=gpt-oss:20b`. Copy `.env.example` to `.env` to change the endpoint, model, or `CHURNSCOPE_PORT` (default 8000). Ollama is optional; deterministic summaries and recommendations remain available.

## Academic notebook

Open [notebooks/churnscope_pipeline.ipynb](notebooks/churnscope_pipeline.ipynb) in Jupyter and run top to bottom. Its presentation sequence is problem definition, synthetic data, cutoff/target, EDA, feature engineering, segmentation, classification, forecasting, SHAP, preferences, agent decisions, limitations, and exported artifacts. It calls the same exporter used by FastAPI.

## Important limitations

- Synthetic data, generated from a small population, cannot establish performance on real learners.
- Behavioral clusters overlap; low silhouette scores indicate weak separation.
- Forecast error remains non-trivial, particularly for low-activity or changing series. MAPE is especially unstable near zero usage.
- SHAP describes the model and is not causal.
- Agent recommendations and action execution are simulated. No real messages are sent.
- A local LLM may still produce language errors; evidence is shown for review and unsupported output falls back to deterministic behavior.

## API

- `GET /api/customers`
- `POST /api/segment`
- `POST /api/classify`
- `POST /api/pipeline-guide` (`stage` 0–4, optional `completed` and `customer_id`)
- `GET /api/customer/{customer_id}`
- `GET /api/customer/{customer_id}/forecast`
- `GET /api/customer/{customer_id}/explanation`
- `POST /api/customer/{customer_id}/chat`
- `POST /api/customer/{customer_id}/recommend`
- `GET /api/customer/{customer_id}/eligible-actions`
- `POST /api/customer/{customer_id}/execute-action` (simulated)
- `GET /api/model-metrics`

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs).

Chat accepts `question`, optional `conversation`, and optional `quick` (default `false`). `quick: true` answers example evidence/action questions from customer-specific model outputs and the eligibility policy without calling the local LLM.
