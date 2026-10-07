# ChurnScope AI

**Forecast → Classify → Explain → Act**

ChurnScope AI is an academic demonstration for a subscription-based learning platform. It identifies learners with changing engagement, estimates their cancellation risk, forecasts four weeks of usage, explains a model score with SHAP, and grounds a local AI retention agent in that evidence and recorded preferences.

The data is synthetic and generated with a fixed seed. This is a classroom demo, not a production retention system.

## Quick start

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8000](http://localhost:8000). `run.py` trains and exports the models on first launch if the artifacts are absent, then starts the FastAPI server and serves the frontend from that same process.

## Demo walkthrough

1. Press **Segment customers** to animate the 300 learners into behavior groups.
2. Press **Run classification** to add model-derived risk tiers and update the counters.
3. Use the filters or the demo customer selector. C001–C004 are stable, gradually disengaging, high-risk, and recovering example histories.
4. Inspect the selected learner’s churn score, next-four-week usage forecast, SHAP contributions, and preference profile.
5. Ask the retention agent why the learner is at risk or what action is suitable. The deterministic fallback works without Ollama.
6. **Execute action** only records a simulated in-memory event; it never sends email or messages.

## Academic notebook

Open [notebooks/churnscope_pipeline.ipynb](notebooks/churnscope_pipeline.ipynb) in Jupyter and run the cells from top to bottom. The notebook explains the problem, explores data, builds features, compares clustering and classifiers, evaluates walk-forward forecasts, inspects SHAP values, and calls the shared exporter. The app loads the exact `models/pipeline.joblib` file produced by that exporter; API requests do not train substitute models.

To regenerate source data and model artifacts from the fixed seed:

```powershell
python scripts/generate_data.py
python scripts/export_models.py
```

## Dataset and methodology

- `data/customer_weekly_history.csv`: 300 learners × 32 weeks, including sessions, active days, usage minutes, lessons, labs, searches, support requests, inactivity, subscription age, and cancellation target.
- `data/customer_profiles.csv`: separately generated learning preferences and engagement profile.
- `data/generated/customer_predictions.csv`: model probabilities, behavioral segment, and 2D map coordinates.
- Behavior patterns include stable, gradual decline, sudden decline, irregular, dormant, and recovering engagement. Outcomes overlap and depend on engagement history; demographic and profile fields do not determine churn.
- The label represents cancellation in the four-week window after the final observed week. Classifier inputs summarize only observed records. The split is by customer to prevent weekly-row leakage.
- K-Means provides complete, stable groups for the map. DBSCAN is compared as a density-based alternative and may mark many customers as noise. PCA is used only for the 2D visualization.
- Logistic Regression and Random Forest are evaluated on a customer-level holdout using accuracy, precision, recall, F1, and ROC-AUC. Risk thresholds are centralized in `backend/config.py`.
- Naive and damped-trend exponential smoothing are evaluated with three walk-forward cutoffs. The selected method minimizes validation MAE. Forecast bounds use the 90th percentile absolute walk-forward error; they are approximate empirical bounds, not a formal calibrated interval.
- SHAP values describe the fitted classifier’s output and are not causal explanations.

## Evidence vs AI reasoning

- **ML model** → churn probability and risk tier.
- **Forecast model** → future weekly usage and change.
- **SHAP** → contributions to the churn prediction.
- **Customer data** → preferences and historical engagement profile.
- **GPT-OSS** → an allowed action choice and personalized wording from supplied evidence.

The local agent must not invent probability, history, forecast, SHAP values, or preferences. It may choose only from a deterministic eligibility list. Unsupported actions or numerical claims fall back to a deterministic recommendation. Review the cited model evidence alongside generated wording.

## Ollama setup

Install and start Ollama separately, then make the configured model available locally:

```powershell
ollama pull gpt-oss:20b
```

Defaults are `OLLAMA_URL=http://localhost:11434` and `OLLAMA_MODEL=gpt-oss:20b`. Copy `.env.example` to `.env` and edit it if you use another local endpoint, model, or app port. `CHURNSCOPE_PORT` defaults to 8000. Ollama is optional; the app displays **“Local LLM unavailable — deterministic recommendation mode.”** when it cannot use the local model.

## Architecture

```text
scripts/generate_data.py ──> data/*.csv
scripts/export_models.py ──> models/pipeline.joblib + metrics.json
notebooks/churnscope_pipeline.ipynb ──> shared training/export pipeline
run.py ──> FastAPI backend ──> vanilla HTML/CSS/JS frontend
                         └──> optional local Ollama provider
```

The backend stays intentionally small: no account system, database, microservices, container stack, cloud deployment, or real message delivery.

## API

- `GET /api/customers`
- `POST /api/segment`
- `POST /api/classify`
- `GET /api/customer/{customer_id}`
- `GET /api/customer/{customer_id}/forecast`
- `GET /api/customer/{customer_id}/explanation`
- `POST /api/customer/{customer_id}/chat`
- `POST /api/customer/{customer_id}/recommend`
- `POST /api/customer/{customer_id}/execute-action` (simulated)
- `GET /api/model-metrics`

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs).

## Limitations

The population is synthetic, small, and designed for an interpretable academic walkthrough. Holdout scores measure this generated population and do not establish performance on real learners. Forecast errors are high for low-activity series, and the uncertainty band is empirical. The local LLM can still make language errors; evidence remains visible so recommendations can be reviewed.
