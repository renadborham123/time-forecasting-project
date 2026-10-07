"""Train the shared academic pipeline and export the artifacts used by FastAPI."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_data import generate
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, silhouette_score, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
FEATURES = ["avg_sessions","recent_sessions","sessions_trend","avg_usage_minutes","recent_usage_minutes","usage_trend","avg_active_days","active_days_trend","lesson_completion_rate","lab_usage_rate","days_since_last_activity","engagement_volatility","subscription_age_weeks","avg_completed_lessons","recent_completed_lessons","avg_completed_labs","recent_support_requests","activity_lag_1","activity_lag_2","activity_lag_4","rolling_mean_4","rolling_std_4"]


def engineer_features(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in history.sort_values("week").groupby("customer_id", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        r, prev = g.tail(4), g.iloc[-8:-4]
        def slope(s): return float(np.polyfit(np.arange(len(s)), np.asarray(s, dtype=float), 1)[0]) if len(s) > 1 else 0.0
        usage = g.usage_minutes.astype(float)
        rows.append({"customer_id":cid,"avg_sessions":g.weekly_sessions.mean(),"recent_sessions":r.weekly_sessions.mean(),"sessions_trend":r.weekly_sessions.mean()-prev.weekly_sessions.mean(),"avg_usage_minutes":usage.mean(),"recent_usage_minutes":r.usage_minutes.mean(),"usage_trend":r.usage_minutes.mean()-prev.usage_minutes.mean(),"avg_active_days":g.active_days.mean(),"active_days_trend":r.active_days.mean()-prev.active_days.mean(),"lesson_completion_rate":g.completed_lessons.sum()/max(1,g.weekly_sessions.sum()),"lab_usage_rate":g.completed_labs.sum()/max(1,g.completed_lessons.sum()),"days_since_last_activity":r.days_since_last_activity.mean(),"engagement_volatility":usage.tail(8).std(ddof=0),"subscription_age_weeks":g.subscription_age_weeks.iloc[-1],"avg_completed_lessons":g.completed_lessons.mean(),"recent_completed_lessons":r.completed_lessons.mean(),"avg_completed_labs":g.completed_labs.mean(),"recent_support_requests":r.support_requests.sum(),"activity_lag_1":g.weekly_sessions.iloc[-1],"activity_lag_2":g.weekly_sessions.iloc[-2],"activity_lag_4":g.weekly_sessions.iloc[-4],"rolling_mean_4":g.weekly_sessions.tail(4).mean(),"rolling_std_4":g.weekly_sessions.tail(4).std(ddof=0),"cancelled":int(g.cancelled.iloc[-1])})
    return pd.DataFrame(rows)


def forecast_series(values, method):
    values = np.asarray(values, dtype=float)
    if method == "naive": return np.repeat(values[-1], 4)
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # A recent training window lets the forecast reflect changing engagement regimes.
    values = values[-16:]
    fit = ExponentialSmoothing(values, trend="add", damped_trend=True, initialization_method="estimated").fit(optimized=True)
    return np.maximum(0, np.asarray(fit.forecast(4), dtype=float))


def score_forecasting(history):
    rows = []
    for _, g in history.groupby("customer_id"):
        values = g.sort_values("week").usage_minutes.to_numpy(float)
        for cut in (20, 24, 28):
            if len(values) <= cut + 3: continue
            actual = values[cut:cut+4]
            for method in ("naive", "exp_smoothing"):
                try: pred = forecast_series(values[:cut], method)
                except Exception: pred = np.repeat(values[cut-1], 4)
                rows.append({"method":method,"actual":actual,"pred":pred})
    metrics={}
    for method in ("naive", "exp_smoothing"):
        pairs=[r for r in rows if r["method"]==method]
        y=np.concatenate([r["actual"] for r in pairs]); p=np.concatenate([r["pred"] for r in pairs])
        metrics[method]={"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),"mape":float(np.mean(np.abs((y-p)[y>1]/y[y>1]))*100)}
    selected=min(metrics,key=lambda k:metrics[k]["mae"])
    residuals=[]
    for row in rows:
        if row["method"]==selected: residuals.extend((row["actual"]-row["pred"]).tolist())
    spread=float(np.quantile(np.abs(residuals),.9)) if residuals else 0
    return selected, metrics, spread


def train_export():
    history_path=ROOT/"data/customer_weekly_history.csv"
    profile_path=ROOT/"data/customer_profiles.csv"
    if history_path.exists() and profile_path.exists():
        history=pd.read_csv(history_path); profiles=pd.read_csv(profile_path)
    else: history,profiles=generate()
    features=engineer_features(history)
    X=features[FEATURES].replace([np.inf,-np.inf],0).fillna(0); y=features.cancelled.astype(int)
    if y.nunique()<2: raise ValueError("Synthetic data needs both churn outcomes; regenerate with the fixed seed.")
    train_idx,test_idx=train_test_split(np.arange(len(y)),test_size=.25,random_state=42,stratify=y)
    candidates={"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",C=.8))]),"Random Forest":RandomForestClassifier(n_estimators=240,min_samples_leaf=3,max_features=.8,class_weight="balanced_subsample",random_state=42,n_jobs=-1)}
    scores={}
    for name,model in candidates.items():
        model.fit(X.iloc[train_idx],y.iloc[train_idx]); prob=model.predict_proba(X.iloc[test_idx])[:,1]; pred=(prob>=.5).astype(int)
        scores[name]={"accuracy":float(accuracy_score(y.iloc[test_idx],pred)),"precision":float(precision_score(y.iloc[test_idx],pred,zero_division=0)),"recall":float(recall_score(y.iloc[test_idx],pred,zero_division=0)),"f1":float(f1_score(y.iloc[test_idx],pred,zero_division=0)),"roc_auc":float(roc_auc_score(y.iloc[test_idx],prob))}
    selected_name=max(scores,key=lambda n:(scores[n]["recall"]*.45+scores[n]["roc_auc"]*.4+scores[n]["f1"]*.15))
    # Refit the selected candidate on all customer snapshots after a customer-level holdout evaluation.
    classifier=candidates[selected_name]
    classifier.fit(X,y)
    scaler=StandardScaler().fit(X)
    scaled=scaler.transform(X)
    km=KMeans(n_clusters=5,random_state=42,n_init=20).fit(scaled)
    db=DBSCAN(eps=2.15,min_samples=5).fit(scaled)
    db_labels=db.labels_
    db_nonnoise=db_labels!=-1
    try: db_sil=float(silhouette_score(scaled[db_nonnoise],db_labels[db_nonnoise])) if db_nonnoise.sum()>3 and len(set(db_labels[db_nonnoise]))>1 else None
    except Exception: db_sil=None
    k_sil=float(silhouette_score(scaled,km.labels_))
    embedding=PCA(n_components=2,random_state=42).fit(scaled)
    coords=embedding.transform(scaled)
    # Name the clusters only after inspecting their behavior summaries.
    frame=features.copy(); frame["cluster"]=km.labels_
    stats=frame.groupby("cluster").agg(recent_sessions=("recent_sessions","mean"),usage_trend=("usage_trend","mean"),recent_usage=("recent_usage_minutes","mean"),inactivity=("days_since_last_activity","mean")).to_dict("index")
    by_engagement=sorted(stats,key=lambda c:(stats[c]["recent_sessions"],stats[c]["recent_usage"]))
    names={by_engagement[0]:"Low Activity",by_engagement[1]:"Fading Engagement",by_engagement[2]:"Irregular Users",by_engagement[3]:"Core Engaged",by_engagement[4]:"Power Users"}
    labels=km.labels_
    probabilities=classifier.predict_proba(X)[:,1]
    selected_forecast,forecast_metrics,spread=score_forecasting(history)
    shap_background=X.sample(min(60,len(X)),random_state=42)
    artifact_dir=ROOT/"models"; artifact_dir.mkdir(exist_ok=True)
    (ROOT/"data/generated").mkdir(parents=True,exist_ok=True)
    forecast_dir=artifact_dir/"forecast_artifacts"; forecast_dir.mkdir(parents=True,exist_ok=True)
    bundle={"features":FEATURES,"classifier":classifier,"classifier_name":selected_name,"scaler":scaler,"kmeans":km,"embedding":embedding,"cluster_names":names,"cluster_stats":stats,"shap_background":shap_background,"forecast_method":selected_forecast,"forecast_spread":spread}
    joblib.dump(bundle,artifact_dir/"pipeline.joblib",compress=3)
    joblib.dump(km,artifact_dir/"clustering_model.joblib",compress=3)
    joblib.dump(embedding,artifact_dir/"embedding_model.joblib",compress=3)
    joblib.dump(classifier,artifact_dir/"churn_classifier.joblib",compress=3)
    joblib.dump(scaler,artifact_dir/"churn_preprocessor.joblib",compress=3)
    joblib.dump(shap_background,artifact_dir/"shap_background.joblib",compress=3)
    (forecast_dir/"selection.json").write_text(json.dumps({"selected_model":selected_forecast,"walk_forward_metrics":forecast_metrics,"interval_abs_error_p90":spread},indent=2),encoding="utf-8")
    pred=pd.DataFrame({"customer_id":features.customer_id,"churn_probability":probabilities,"cluster":labels,"segment":[names[x] for x in labels],"embedding_x":coords[:,0],"embedding_y":coords[:,1]})
    pred.to_csv(ROOT/"data/generated/customer_predictions.csv",index=False)
    (artifact_dir/"metrics.json").write_text(json.dumps({"classification":{"selected_model":selected_name,"holdout_size":len(test_idx),"customer_split":True,"models":scores},"clustering":{"selected_model":"K-Means","kmeans_clusters":5,"kmeans_silhouette":k_sil,"dbscan_clusters":len(set(db_labels))-(1 if -1 in db_labels else 0),"dbscan_noise_points":int((db_labels==-1).sum()),"dbscan_silhouette":db_sil,"selection_reason":"K-Means gives every customer a stable segment for the map; DBSCAN is compared as a density-based alternative."},"forecasting":{"selected_model":selected_forecast,"walk_forward_cutoffs":[20,24,28],"metrics":forecast_metrics,"interval_abs_error_p90":spread},"dataset":{"customers":len(profiles),"weekly_records":len(history),"churn_rate":float(y.mean()),"weeks_per_customer":int(history.groupby('customer_id').size().median())}},indent=2),encoding="utf-8")
    print(f"Exported pipeline for {len(profiles)} customers. Churn rate={y.mean():.1%}, selected classifier={selected_name}, forecast={selected_forecast}.")


if __name__=="__main__": train_export()
