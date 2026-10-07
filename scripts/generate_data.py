"""Generate fixed-seed synthetic activity, preferences, and churn outcomes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
N_CUSTOMERS = 300
N_WEEKS = 32
PREDICTION_CUTOFF = 28
OUTCOME_WEEKS = (29, 30, 31, 32)


def _churn_probability(observed: pd.DataFrame, pattern: str) -> float:
    """Generate an overlapping outcome from information available at week 28."""
    recent = observed.tail(8)
    usage_slope = float(np.polyfit(np.arange(len(recent)), recent.usage_minutes, 1)[0])
    inactivity = float(recent.days_since_last_activity.mean())
    sessions = float(recent.weekly_sessions.mean())
    logit = -1.65 - 0.16 * usage_slope + 0.025 * (inactivity - 14) + 0.22 * max(0, 5.5 - sessions)
    logit += {"sudden": 0.35, "dormant": 0.30, "recovering": -0.45}.get(pattern, 0.0)
    return float(1 / (1 + np.exp(-np.clip(logit, -20, 20))))


def generate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    topics = ["Machine Learning", "Python", "Data Analysis", "Deep Learning", "NLP"]
    contents = ["Videos", "Interactive Labs", "Projects", "Quizzes"]
    features = ["Interactive Labs", "Learning Paths", "Practice Problems", "Projects"]
    times = ["Morning", "Afternoon", "Evening", "Weekend"]
    profiles, weekly, targets = [], [], []
    pattern_names = ["stable", "gradual", "sudden", "irregular", "dormant", "recovering"]
    patterns = rng.choice(pattern_names, N_CUSTOMERS, p=[.30, .20, .14, .14, .10, .12]).tolist()
    patterns[:4] = ["stable", "gradual", "sudden", "recovering"]

    for i, pattern in enumerate(patterns):
        cid = f"C{i + 1:03d}"
        base = float(rng.uniform(5, 13))
        if i == 0: base = 15.5
        if i == 1: base = 12.5
        if i == 2: base = 13.0
        if i == 3: base = 10.0
        topic = str(rng.choice(topics))
        content = str(rng.choice(contents))
        feature = str(rng.choice(features))
        study_time = str(rng.choice(times))
        profiles.append({
            "customer_id": cid,
            "age_group": str(rng.choice(["18–24", "25–34", "35–44", "45+"])),
            "favorite_topic": topic,
            "preferred_content_type": content,
            "preferred_feature": feature,
            "preferred_study_time": study_time,
            "historical_engagement_level": "High" if base >= 10 else "Medium" if base >= 7 else "Developing",
        })
        level = rng.normal(0, .30)
        starting_age = int(rng.integers(12, 85))
        series = []
        for w in range(N_WEEKS):
            t = w / (N_WEEKS - 1)
            trend = 0.0
            if pattern == "gradual": trend = -8.2 * t
            elif pattern == "sudden" and w >= 18: trend = -10.5 * min(1, (w - 17) / 4)
            elif pattern == "dormant": trend = -5.0 * t
            elif pattern == "recovering": trend = -5.4 * min(1, t * 1.5) + 5.0 * max(0, (t - .48) / .52)
            elif pattern == "irregular": trend = 1.15 * np.sin(w * 1.7 + i)
            elif pattern == "stable": trend = .7 * t
            latent = max(.35, base + trend + level + rng.normal(0, 1.0 if pattern != "irregular" else 2.2))
            if pattern == "dormant": latent = max(.1, latent * .36)
            sessions = int(np.clip(rng.poisson(max(.2, latent)), 0, 24))
            active_days = int(np.clip(rng.binomial(7, min(.93, max(.04, sessions / 14))), 0, 7))
            measurement_noise = 8 if pattern == "stable" else 24
            minutes = round(max(0, sessions * rng.normal(28, 7) + rng.normal(0, measurement_noise)), 1)
            lessons = int(rng.poisson(max(.08, sessions * .46)))
            labs = int(rng.binomial(max(lessons, 1), .24 if content != "Interactive Labs" else .46))
            searches = int(rng.poisson(max(.1, sessions * .32)))
            support = int(rng.binomial(1, .035 + (.13 if pattern == "sudden" else .04 if pattern == "dormant" else 0)))
            days_idle = int(np.clip(rng.gamma(2.0, 5.0 / max(.12, latent / 8)), 0, 60))
            series.append({
                "customer_id": cid, "week": w + 1, "weekly_sessions": sessions,
                "active_days": active_days, "usage_minutes": minutes,
                "completed_lessons": lessons, "completed_labs": labs,
                "searches": searches, "support_requests": support,
                "days_since_last_activity": days_idle,
                "subscription_age_weeks": starting_age + w,
            })
        frame = pd.DataFrame(series)

        # Curated demo histories are shaped in weeks 21–28, the classifier's
        # observation window. Their labels still come only from the same
        # reproducible stochastic outcome mechanism as all other customers.
        if i == 0:
            frame.loc[20:27, "weekly_sessions"] = [14, 14, 14, 15, 15, 16, 16, 17]
            frame.loc[20:27, "usage_minutes"] = [390, 394, 398, 410, 420, 433, 446, 460]
            frame.loc[20:27, "active_days"] = [5, 5, 5, 5, 6, 6, 6, 6]
            frame.loc[20:27, "days_since_last_activity"] = [4, 3, 3, 2, 2, 1, 1, 0]
        elif i == 1:
            frame.loc[20:27, "weekly_sessions"] = [12, 11, 10, 9, 8, 7, 6, 5]
            frame.loc[20:27, "usage_minutes"] = [330, 305, 280, 250, 220, 190, 160, 130]
            frame.loc[20:27, "active_days"] = [6, 6, 5, 5, 4, 4, 3, 3]
        elif i == 2:
            frame.loc[20:27, "weekly_sessions"] = [12, 11, 9, 7, 5, 3, 1, 0]
            frame.loc[20:27, "usage_minutes"] = [330, 292, 258, 205, 155, 92, 35, 4]
            frame.loc[20:27, "active_days"] = [6, 6, 5, 4, 3, 2, 1, 0]
            frame.loc[20:27, "completed_lessons"] = [7, 6, 5, 4, 3, 2, 1, 0]
            frame.loc[20:27, "days_since_last_activity"] = [1, 2, 4, 7, 12, 19, 32, 52]
        elif i == 3:
            frame.loc[20:27, "weekly_sessions"] = [3, 3, 4, 5, 6, 8, 10, 12]
            frame.loc[20:27, "usage_minutes"] = [75, 72, 95, 120, 160, 215, 275, 330]
            frame.loc[20:27, "active_days"] = [2, 2, 3, 3, 4, 4, 5, 6]

        observed = frame[frame.week <= PREDICTION_CUTOFF]
        probability = _churn_probability(observed, pattern)
        churn = int(rng.random() < probability)
        event_week = int(rng.choice(OUTCOME_WEEKS)) if churn else None
        targets.append({
            "customer_id": cid,
            "prediction_cutoff_week": PREDICTION_CUTOFF,
            "outcome_window_start_week": OUTCOME_WEEKS[0],
            "outcome_window_end_week": OUTCOME_WEEKS[-1],
            "will_churn_next_4_weeks": churn,
            "cancellation_week": event_week,
            "synthetic_churn_propensity": round(probability, 6),
        })

        # Keep future outcomes in the weekly records for EDA and forecast
        # evaluation. These fields are never part of classifier features.
        frame["cancelled"] = 0
        if event_week is not None:
            frame.loc[frame.week == event_week, "cancelled"] = 1
        weekly.extend(frame.to_dict("records"))

    DATA.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(weekly)
    profile_df = pd.DataFrame(profiles)
    targets_df = pd.DataFrame(targets)
    history.to_csv(DATA / "customer_weekly_history.csv", index=False)
    profile_df.to_csv(DATA / "customer_profiles.csv", index=False)
    targets_df.to_csv(DATA / "customer_churn_targets.csv", index=False)
    return history, profile_df, targets_df


if __name__ == "__main__":
    h, p, t = generate()
    print(f"Generated {len(p)} profiles, {len(h)} weekly records, and {len(t)} cutoff labels at {DATA}")
