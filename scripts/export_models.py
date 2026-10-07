"""Train and export the shared, temporally valid ChurnScope pipeline."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_data import PREDICTION_CUTOFF, generate
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, silhouette_score, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "avg_sessions", "recent_sessions", "sessions_trend", "avg_usage_minutes",
    "recent_usage_minutes", "usage_trend", "avg_active_days", "active_days_trend",
    "lesson_completion_rate", "lab_usage_rate", "days_since_last_activity",
    "engagement_volatility", "subscription_age_weeks", "avg_completed_lessons",
    "recent_completed_lessons", "avg_completed_labs", "recent_support_requests",
    "activity_lag_1", "activity_lag_2", "activity_lag_4", "rolling_mean_4",
    "rolling_std_4",
]
FORECAST_METHODS = ("naive", "exp_smoothing", "arima", "lag_regression")


def _slope(values) -> float:
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) > 1 else 0.0


def engineer_features(history: pd.DataFrame, cutoff: int = PREDICTION_CUTOFF) -> pd.DataFrame:
    """Build one classifier row per customer from observations through cutoff."""
    observed = history[history.week <= cutoff].sort_values(["customer_id", "week"])
    rows = []
    for cid, g in observed.groupby("customer_id", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        if g.empty:
            continue
        recent, prior = g.tail(4), g.iloc[-8:-4]
        if prior.empty:
            prior = recent
        sessions = g.weekly_sessions.astype(float)
        usage = g.usage_minutes.astype(float)
        rows.append({
            "customer_id": cid,
            "avg_sessions": sessions.mean(),
            "recent_sessions": recent.weekly_sessions.mean(),
            "sessions_trend": recent.weekly_sessions.mean() - prior.weekly_sessions.mean(),
            "avg_usage_minutes": usage.mean(),
            "recent_usage_minutes": recent.usage_minutes.mean(),
            "usage_trend": recent.usage_minutes.mean() - prior.usage_minutes.mean(),
            "usage_slope_recent": _slope(usage.tail(8)),
            "sessions_slope_recent": _slope(sessions.tail(8)),
            "avg_active_days": g.active_days.mean(),
            "active_days_trend": recent.active_days.mean() - prior.active_days.mean(),
            "lesson_completion_rate": g.completed_lessons.sum() / max(1, sessions.sum()),
            "lab_usage_rate": g.completed_labs.sum() / max(1, g.completed_lessons.sum()),
            "days_since_last_activity": recent.days_since_last_activity.mean(),
            "engagement_volatility": usage.tail(8).std(ddof=0),
            "subscription_age_weeks": int(g.subscription_age_weeks.iloc[-1]),
            "avg_completed_lessons": g.completed_lessons.mean(),
            "recent_completed_lessons": recent.completed_lessons.mean(),
            "avg_completed_labs": g.completed_labs.mean(),
            "recent_support_requests": recent.support_requests.sum(),
            "activity_lag_1": sessions.iloc[-1],
            "activity_lag_2": sessions.iloc[-2] if len(sessions) >= 2 else sessions.iloc[-1],
            "activity_lag_4": sessions.iloc[-4] if len(sessions) >= 4 else sessions.iloc[0],
            "rolling_mean_4": sessions.tail(4).mean(),
            "rolling_std_4": sessions.tail(4).std(ddof=0),
        })
    return pd.DataFrame(rows)


def _lag_features(values):
    window = np.asarray(values[-4:], dtype=float)
    if len(window) < 4:
        raise ValueError("Lag regression needs at least four observations")
    return np.r_[window[::-1], window.mean(), window.std(ddof=0), _slope(window)]


def _lag_regression(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 8:
        raise ValueError("Lag regression needs at least eight observations")
    X = np.asarray([_lag_features(values[:i]) for i in range(4, len(values))])
    y = values[4:]
    model = Ridge(alpha=2.0).fit(X, y)
    history = list(values)
    result = []
    for _ in range(4):
        prediction = float(model.predict(_lag_features(history).reshape(1, -1))[0])
        prediction = max(0.0, prediction)
        result.append(prediction)
        history.append(prediction)
    return np.asarray(result)


def forecast_series(values, method):
    """Return four forecasts; errors are allowed so callers can apply fallbacks."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Forecast input must contain finite observations")
    if method == "naive":
        return np.repeat(max(0.0, values[-1]), 4)
    if method == "lag_regression":
        return _lag_regression(values)
    recent = values[-16:]
    if method == "exp_smoothing":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(recent, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit(optimized=True)
        return np.maximum(0, np.asarray(fit.forecast(4), dtype=float))
    if method == "arima":
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(recent, order=(1, 1, 0), enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        return np.maximum(0, np.asarray(fit.forecast(4), dtype=float))
    raise ValueError(f"Unknown forecast method: {method}")


def _forecast_with_fallback(values, method):
    errors = {}
    for candidate in dict.fromkeys((method, "exp_smoothing", "naive")):
        try:
            forecast = forecast_series(values, candidate)
            return forecast, candidate, errors
        except Exception as exc:  # one degenerate customer must not take down the API
            errors[candidate] = type(exc).__name__
    raise RuntimeError(f"Naive fallback failed: {errors}")


def _smape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    denom = np.abs(actual) + np.abs(predicted)
    values = np.divide(200 * np.abs(actual - predicted), denom,
                       out=np.zeros_like(denom), where=denom > 0)
    return float(np.mean(values))


def score_forecasting(history):
    """Evaluate candidates using chronological four-step walk-forward folds."""
    rows = {name: [] for name in FORECAST_METHODS}
    fallback_counts = {name: 0 for name in FORECAST_METHODS}
    for _, group in history.groupby("customer_id", sort=True):
        values = group.sort_values("week").usage_minutes.to_numpy(float)
        for cut in (20, 24, 28):
            if len(values) < cut + 4:
                continue
            train, actual = values[:cut], values[cut:cut + 4]
            scale = float(np.mean(np.abs(np.diff(train)))) if len(train) > 1 else 0.0
            for method in FORECAST_METHODS:
                try:
                    pred = forecast_series(train, method)
                except Exception:
                    pred, _, _ = _forecast_with_fallback(train, "exp_smoothing")
                    fallback_counts[method] += 1
                rows[method].append((actual, pred, scale))
    metrics = {}
    for method, pairs in rows.items():
        actual = np.concatenate([p[0] for p in pairs])
        predicted = np.concatenate([p[1] for p in pairs])
        nonzero = np.abs(actual) > 1e-8
        mase_parts = [np.abs(p[0] - p[1]) / p[2] for p in pairs if p[2] > 1e-8]
        metrics[method] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "smape": _smape(actual, predicted),
            "mase": float(np.mean(np.concatenate(mase_parts))) if mase_parts else None,
            "mape": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None,
            "fallback_folds": fallback_counts[method],
            "evaluation_points": int(len(actual)),
        }
    # Selection is transparent: MAE is primary. RMSE and sMAPE are reported
    # beside it to expose large errors and near-zero sensitivity.
    selected = min(metrics, key=lambda name: (metrics[name]["mae"], metrics[name]["rmse"], metrics[name]["smape"]))
    residuals = []
    for _, group in history.groupby("customer_id", sort=True):
        values = group.sort_values("week").usage_minutes.to_numpy(float)
        for cut in (20, 24, 28):
            if len(values) >= cut + 4:
                pred, _, _ = _forecast_with_fallback(values[:cut], selected)
                residuals.extend((values[cut:cut + 4] - pred).tolist())
    spread = float(np.quantile(np.abs(residuals), .9)) if residuals else 0.0
    return selected, metrics, spread


def _standardized(statistics, column):
    vals = statistics[column].astype(float)
    scale = float(vals.std(ddof=0))
    return (vals - vals.mean()) / (scale if scale > 1e-9 else 1.0)


def name_clusters(features: pd.DataFrame, cluster_ids):
    frame = features.copy()
    frame["cluster"] = np.asarray(cluster_ids)
    stats = frame.groupby("cluster").agg(
        customers=("customer_id", "count"),
        recent_sessions=("recent_sessions", "mean"),
        recent_usage_minutes=("recent_usage_minutes", "mean"),
        usage_trend=("usage_trend", "mean"),
        sessions_trend=("sessions_trend", "mean"),
        active_days_trend=("active_days_trend", "mean"),
        usage_slope_recent=("usage_slope_recent", "mean"),
        sessions_slope_recent=("sessions_slope_recent", "mean"),
        engagement_volatility=("engagement_volatility", "mean"),
        days_since_last_activity=("days_since_last_activity", "mean"),
        active_days=("avg_active_days", "mean"),
        lesson_completion_rate=("lesson_completion_rate", "mean"),
    )
    usage_session = _standardized(stats, "recent_usage_minutes") + _standardized(stats, "recent_sessions")
    remaining = set(stats.index)
    power = int(usage_session.loc[sorted(remaining)].idxmax()); remaining.remove(power)
    low = int(usage_session.loc[sorted(remaining)].idxmin()); remaining.remove(low)
    decline = (_standardized(stats, "usage_trend") + _standardized(stats, "sessions_trend")
               + _standardized(stats, "active_days_trend"))
    fading = int(decline.loc[sorted(remaining)].idxmin()); remaining.remove(fading)
    irregular = int(_standardized(stats, "engagement_volatility").loc[sorted(remaining)].idxmax())
    remaining.remove(irregular)
    core = int(next(iter(remaining)))
    names = {power: "Power Users", core: "Core Engaged", fading: "Fading Engagement",
             low: "Low Activity", irregular: "Irregular Users"}
    rationale = {}
    for cluster, label in names.items():
        s = stats.loc[cluster]
        rationale[str(cluster)] = {
            "label": label,
            "rule": {
                "Power Users": "highest standardized recent usage and sessions",
                "Low Activity": "lowest standardized recent usage and sessions",
                "Fading Engagement": "most negative combined usage, session, and active-day trends among remaining clusters",
                "Irregular Users": "highest engagement volatility among remaining clusters",
                "Core Engaged": "remaining cluster after behavioral extremes are assigned",
            }[label],
            "profile": {key: float(value) for key, value in s.items() if key != "customers"},
        }
    return names, stats, rationale


def train_export():
    history_path = ROOT / "data/customer_weekly_history.csv"
    profile_path = ROOT / "data/customer_profiles.csv"
    target_path = ROOT / "data/customer_churn_targets.csv"
    if history_path.exists() and profile_path.exists() and target_path.exists():
        history = pd.read_csv(history_path)
        profiles = pd.read_csv(profile_path)
        targets = pd.read_csv(target_path)
    else:
        history, profiles, targets = generate()
    features = engineer_features(history)
    targets = targets.set_index("customer_id")
    labelled = features.join(targets[["will_churn_next_4_weeks"]], on="customer_id", validate="one_to_one")
    X = labelled[FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = labelled.will_churn_next_4_weeks.astype(int)
    if y.nunique() < 2:
        raise ValueError("Synthetic data needs both churn outcomes; regenerate with the fixed seed.")
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=.25, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=.8))]),
        "Random Forest": RandomForestClassifier(n_estimators=240, min_samples_leaf=3,
                                                max_features=.8, class_weight="balanced_subsample",
                                                random_state=42, n_jobs=-1),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= .5).astype(int)
        scores[name] = {
            "accuracy": float(accuracy_score(y.iloc[test_idx], pred)),
            "precision": float(precision_score(y.iloc[test_idx], pred, zero_division=0)),
            "recall": float(recall_score(y.iloc[test_idx], pred, zero_division=0)),
            "f1": float(f1_score(y.iloc[test_idx], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y.iloc[test_idx], prob)),
        }
    selected_name = max(scores, key=lambda n: (scores[n]["recall"] * .45 + scores[n]["roc_auc"] * .4 + scores[n]["f1"] * .15))
    classifier = candidates[selected_name].fit(X, y)

    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(scaled)
    db = DBSCAN(eps=2.15, min_samples=5).fit(scaled)
    db_labels = db.labels_
    db_nonnoise = db_labels != -1
    try:
        db_sil = float(silhouette_score(scaled[db_nonnoise], db_labels[db_nonnoise])) if db_nonnoise.sum() > 3 and len(set(db_labels[db_nonnoise])) > 1 else None
    except Exception:
        db_sil = None
    k_sil = float(silhouette_score(scaled, km.labels_))
    embedding = PCA(n_components=2, random_state=42).fit(scaled)
    coords = embedding.transform(scaled)
    names, stats, rationale = name_clusters(features, km.labels_)

    probabilities = classifier.predict_proba(X)[:, 1]
    selected_forecast, forecast_metrics, spread = score_forecasting(history)
    shap_background = X.sample(min(60, len(X)), random_state=42)
    artifact_dir = ROOT / "models"
    artifact_dir.mkdir(exist_ok=True)
    (ROOT / "data/generated").mkdir(parents=True, exist_ok=True)
    forecast_dir = artifact_dir / "forecast_artifacts"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "features": FEATURES, "classifier": classifier, "classifier_name": selected_name,
        "scaler": scaler, "kmeans": km, "embedding": embedding,
        "cluster_names": names,
        "cluster_stats": {str(k): {key: (int(v) if key == "customers" else float(v)) for key, v in row.items()} for k, row in stats.to_dict("index").items()},
        "cluster_naming_rationale": rationale,
        "shap_background": shap_background,
        "forecast_method": selected_forecast, "forecast_spread": spread,
    }
    joblib.dump(bundle, artifact_dir / "pipeline.joblib", compress=3)
    joblib.dump(km, artifact_dir / "clustering_model.joblib", compress=3)
    joblib.dump(embedding, artifact_dir / "embedding_model.joblib", compress=3)
    joblib.dump(classifier, artifact_dir / "churn_classifier.joblib", compress=3)
    joblib.dump(scaler, artifact_dir / "churn_preprocessor.joblib", compress=3)
    joblib.dump(shap_background, artifact_dir / "shap_background.joblib", compress=3)
    (forecast_dir / "selection.json").write_text(json.dumps({
        "selected_model": selected_forecast,
        "selection_rule": "lowest walk-forward MAE, with RMSE and sMAPE reported as secondary evidence",
        "walk_forward_metrics": forecast_metrics,
        "interval_abs_error_p90": spread,
        "fallback_order": list(dict.fromkeys((selected_forecast, "exp_smoothing", "naive"))),
    }, indent=2), encoding="utf-8")
    pred = pd.DataFrame({
        "customer_id": features.customer_id, "churn_probability": probabilities,
        "risk_tier": ["STABLE" if x < .30 else "WATCHLIST" if x < .60 else "AT RISK" if x < .80 else "CRITICAL" for x in probabilities],
        "cluster": km.labels_,
        "segment": [names[int(x)] for x in km.labels_],
        "embedding_x": coords[:, 0], "embedding_y": coords[:, 1],
    })
    pred.to_csv(ROOT / "data/generated/customer_predictions.csv", index=False)
    metrics = {
        "classification": {
            "selected_model": selected_name, "holdout_size": len(test_idx),
            "customer_split": True, "prediction_cutoff_week": PREDICTION_CUTOFF,
            "target": "will_churn_next_4_weeks", "models": scores,
        },
        "clustering": {
            "selected_model": "K-Means", "kmeans_clusters": 5,
            "kmeans_silhouette": k_sil,
            "dbscan_clusters": len(set(db_labels)) - (1 if -1 in db_labels else 0),
            "dbscan_noise_points": int((db_labels == -1).sum()), "dbscan_silhouette": db_sil,
            "selection_reason": "K-Means provides complete, stable coverage for map visualization and behavioral profiling; the silhouette score indicates overlapping groups, and DBSCAN remains a density-based comparison.",
            "cluster_stats": bundle["cluster_stats"], "cluster_naming_rationale": rationale,
        },
        "forecasting": {
            "selected_model": selected_forecast,
            "walk_forward_cutoffs": [20, 24, 28],
            "forecast_horizon": 4,
            "selection_rule": "lowest MAE; RMSE and sMAPE are secondary evidence",
            "metrics": forecast_metrics, "interval_abs_error_p90": spread,
            "fallback_order": list(dict.fromkeys((selected_forecast, "exp_smoothing", "naive"))),
        },
        "dataset": {
            "customers": len(profiles), "weekly_records": len(history),
            "churn_rate": float(y.mean()), "weeks_per_customer": int(history.groupby("customer_id").size().median()),
            "prediction_cutoff_week": PREDICTION_CUTOFF,
            "outcome_window_weeks": [29, 30, 31, 32],
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Exported {len(profiles)} cutoff snapshots. Churn rate={y.mean():.1%}, classifier={selected_name}, forecast={selected_forecast}.")


if __name__ == "__main__":
    train_export()
