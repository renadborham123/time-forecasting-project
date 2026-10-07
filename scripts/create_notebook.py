"""Create the presentation-ready academic notebook."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# ChurnScope AI — Forecast → Classify → Explain → Act

**Academic project notebook · Subscription-based learning platform**

This notebook is the reproducible ML and evidence pipeline behind the demo. The prediction date is after week 32; the target represents cancellation in the following four-week window. Every feature uses records available through week 32 only.

## A — Problem definition

We want to segment learners by behavior, estimate individual churn probability, forecast future usage, explain the classifier output, and select a suitable retention action.

- **Clustering ≠ classification:** clustering groups similar activity profiles without churn labels; classification estimates an individual cancellation probability.
- **Forecasting** estimates future weekly usage from the observed timeline.
- **SHAP** attributes a fitted classifier’s output to features; it does not show causal effects.
- **LLM agent** reasons over verified model outputs and recorded preferences to suggest an allowed intervention. It does not create scientific evidence.

Cancellation outcomes overlap across behavior patterns. Demographics and preferences do not determine the label.""")
code("""from pathlib import Path
import sys, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Markdown
ROOT=Path.cwd()
if not (ROOT/'scripts').exists(): ROOT=Path.cwd().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
sns.set_theme(style='whitegrid',palette='deep')
pd.set_option('display.max_columns',30)
print('Project root:',ROOT)""")
md("""## B — Data exploration

Generate 300 reproducible learner profiles and 32 weeks of activity each. The six patterns represent stable, gradually declining, sudden decline, irregular, dormant, and recovering usage. The fixed random seed is 42.""")
code("""from scripts.generate_data import generate
history,profiles=generate()
print('Weekly history:',history.shape,'| Profiles:',profiles.shape)
display(history.head()); display(profiles.head())
print('Missing weekly values:',int(history.isna().sum().sum()))
display(history.describe().round(2).T)
display(history.groupby('customer_id').cancelled.max().value_counts().rename(index={0:'did not cancel',1:'cancelled'}).to_frame('customers'))""")
code("""weekly=history.groupby('week')[['weekly_sessions','usage_minutes','active_days','completed_lessons','completed_labs']].mean()
fig,axes=plt.subplots(2,3,figsize=(15,7))
for ax,col in zip(axes.flat,weekly.columns):
    ax.plot(weekly.index,weekly[col],color='#248c75',lw=2); ax.set_title(col.replace('_',' ').title()); ax.set_xlabel('Week')
axes.flat[-1].axis('off'); fig.suptitle('Population activity trends'); plt.tight_layout()""")
code("""fig,axes=plt.subplots(2,2,figsize=(13,7))
names={'C001':'Stable core user','C002':'Gradually disengaging','C003':'Critical high-risk pattern','C004':'Recovering user'}
for ax,cid in zip(axes.flat,names):
    g=history[history.customer_id==cid]; ax.plot(g.week,g.usage_minutes,color='#278e79',lw=2); ax.set(title=names[cid],xlabel='Week',ylabel='Usage minutes')
plt.tight_layout()""")
md("""## C — Behavioral feature engineering

Customer features use recent and prior four-week windows, an eight-week volatility window, and lag/rolling features at the end of the observed series. No post-prediction data enters X.""")
code("""from scripts.export_models import engineer_features,FEATURES
features=engineer_features(history)
display(features[['customer_id','recent_sessions','sessions_trend','recent_usage_minutes','usage_trend','days_since_last_activity','activity_lag_1','rolling_mean_4','cancelled']].head())
assert features.customer_id.nunique()==300 and set(FEATURES).issubset(features.columns)
assert 'cancelled' not in FEATURES, 'Target leakage detected'""")
md("""## D — Customer clustering and segmentation

The shared exporter standardizes behavior features, compares K-Means and DBSCAN, and uses PCA only for the two-dimensional similarity map. Segment names are assigned after inspecting activity summaries. The phrase “at risk” is reserved for the supervised classifier.""")
md("""## E — Churn classification

Logistic Regression is the baseline; Random Forest is the nonlinear comparison. The train/test split is by customer, so a learner cannot appear in both sets. Models are evaluated on the held-out customers before the chosen model is refit on the full dataset for the application.""")
code("""from scripts.export_models import train_export
train_export()  # same exporter used by FastAPI
metrics=json.loads((ROOT/'models/metrics.json').read_text(encoding='utf-8'))
bundle=joblib.load(ROOT/'models/pipeline.joblib')
X=features[bundle['features']].replace([np.inf,-np.inf],0).fillna(0)
coords=bundle['embedding'].transform(bundle['scaler'].transform(X))
clusters=features.copy(); clusters['cluster']=bundle['kmeans'].predict(bundle['scaler'].transform(X))
display(pd.DataFrame(metrics['clustering'],index=[0]).T)
display(clusters.groupby('cluster').agg(customers=('customer_id','count'),recent_sessions=('recent_sessions','mean'),usage_trend=('usage_trend','mean'),recent_usage=('recent_usage_minutes','mean'),inactivity=('days_since_last_activity','mean')).round(2))
fig,ax=plt.subplots(figsize=(9,5))
for k in sorted(clusters.cluster.unique()):
    mask=clusters.cluster.to_numpy()==k; ax.scatter(coords[mask,0],coords[mask,1],s=25,alpha=.7,label=bundle['cluster_names'][int(k)])
ax.set(title='PCA projection · visualization only',xlabel='Embedding dimension 1',ylabel='Embedding dimension 2'); ax.legend(frameon=False,ncol=2); plt.tight_layout()""")
code("""report=metrics['classification']
display(pd.DataFrame(report['models']).T.round(3))
display(Markdown(f\"**Selected:** {report['selected_model']} · customer-level holdout of {report['holdout_size']} learners. Recall is emphasized because the demo aims to identify learners before cancellation.\"))
prob=bundle['classifier'].predict_proba(X)[:,1]
display(pd.DataFrame({'customer_id':features.customer_id,'churn_probability':prob,'risk_tier':pd.cut(prob,[-.001,.30,.60,.80,1.001],labels=['STABLE','WATCHLIST','AT RISK','CRITICAL'])}).head())""")
md("""## F — Four-week time-series forecasting

Naive last-value and damped-trend exponential smoothing forecasts are compared by walk-forward validation at weeks 20, 24, and 28. Each fold predicts the following four weeks from earlier observations only. MAE selects the method used by the app; RMSE and MAPE are also reported. MAPE omits near-zero actuals.""")
code("""display(pd.DataFrame(metrics['forecasting']['metrics']).T.round(2))
print('Selected forecasting method:',metrics['forecasting']['selected_model'])
from backend.forecasting import forecast as customer_forecast
fc=customer_forecast('C003'); display(pd.DataFrame(fc['history'][-8:]+fc['forecast']))
fig,ax=plt.subplots(figsize=(10,4)); hx=[x['week'] for x in fc['history']]; hy=[x['usage_minutes'] for x in fc['history']]
fx=[hx[-1]]+[x['week'] for x in fc['forecast']]; fy=[hy[-1]]+[x['usage_minutes'] for x in fc['forecast']]
ax.plot(hx,hy,color='#218a74',label='History'); ax.plot(fx,fy,'--',color='#647ef2',label='Forecast')
ax.fill_between([x['week'] for x in fc['forecast']],[x['lower'] for x in fc['forecast']],[x['upper'] for x in fc['forecast']],color='#8498f5',alpha=.14)
ax.axvline(hx[-1],color='gray',ls=':'); ax.set(title='C003 · weekly usage forecast',xlabel='Week',ylabel='Minutes'); ax.legend(frameon=False); plt.tight_layout()""")
md("""## G — SHAP explanation

The service calculates customer-level SHAP values from the exported classifier. Positive contributions increase the fitted model score; negative contributions reduce it. SHAP is an explanation of model behavior, not a causal estimate.""")
code("""from backend.explainability import explanation
e=explanation('C003'); display(pd.DataFrame(e['values']))
chart=pd.DataFrame(e['values']).sort_values('value')
plt.figure(figsize=(8,4)); plt.barh(chart.label,chart.value,color=['#718fe8' if x<0 else '#b184ed' for x in chart.value]); plt.axvline(0,color='gray',lw=.8)
plt.title(f\"C003 · SHAP contributions · model probability {e['probability']:.1%}\"); plt.xlabel('Contribution to model output'); plt.tight_layout()""")
md("""## H — Customer preferences and grounded AI agent

Preferences come from the separate profile table. The backend supplies churn probability/tier, forecast values, SHAP evidence, preferences, and an eligibility-filtered action list. GPT-OSS may select only an eligible action. If Ollama is unavailable, or output fails validation, the application uses deterministic recommendation mode.

| Evidence source | Allowed role |
|---|---|
| Churn classifier | Probability and risk tier |
| Forecasting model | Future weekly usage |
| SHAP | Model feature contributions |
| Customer profile | Recorded preferences |
| GPT-OSS | Eligible action choice and personalized wording |

Action execution is simulated in memory; it never sends a message.""")
code("""from backend.customer_service import profile
from backend.decision_policy import eligible_actions,deterministic_recommendation
p=profile('C003'); display(pd.Series(p['preferences'],name='Recorded profile'))
print('Model risk:',p['risk'],f\"({p['churn_probability']:.1%})\")
print('Eligible actions:',eligible_actions('C003')); display(pd.Series(deterministic_recommendation('C003')))""")
md("""## I — Exported artifacts and leakage checks

`models/pipeline.joblib` is the exact artifact loaded by FastAPI: selected classifier, feature schema, scaler, K-Means/PCA models, segment names, SHAP background, and forecasting method. `models/metrics.json` contains the evaluations. CSVs and models reproduce with seed 42.

The prediction label is separate from feature construction; profile preferences and age group are excluded from classifier inputs. No training occurs inside API routes.""")

nb.cells=C
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.validate(nb)
(ROOT/'notebooks').mkdir(exist_ok=True)
nbf.write(nb,ROOT/'notebooks/churnscope_pipeline.ipynb')
print('Notebook created')
