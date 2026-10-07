"""Generate reproducible synthetic learning-platform activity and preferences."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
N_CUSTOMERS = 300
N_WEEKS = 32


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    topics = ["Machine Learning", "Python", "Data Analysis", "Deep Learning", "NLP"]
    contents = ["Videos", "Interactive Labs", "Projects", "Quizzes"]
    features = ["Interactive Labs", "Learning Paths", "Practice Problems", "Projects"]
    times = ["Morning", "Afternoon", "Evening", "Weekend"]
    profiles, weekly = [], []
    pattern_names = ["stable", "gradual", "sudden", "irregular", "dormant", "recovering"]
    patterns = rng.choice(pattern_names, N_CUSTOMERS, p=[.30, .20, .14, .14, .10, .12]).tolist()
    patterns[:4] = ["stable", "gradual", "sudden", "recovering"]
    for i in range(N_CUSTOMERS):
        cid = f"C{i+1:03d}"
        pattern = patterns[i]
        base = float(rng.uniform(5, 13))
        if i == 0: base = 15.5
        if i == 1: base = 12.5
        if i == 2: base = 13.0
        if i == 3: base = 10.0
        topic, content, feature, study_time = rng.choice(topics), rng.choice(contents), rng.choice(features), rng.choice(times)
        profiles.append({"customer_id":cid,"age_group":rng.choice(["18–24","25–34","35–44","45+"]),"favorite_topic":topic,"preferred_content_type":content,"preferred_feature":feature,"preferred_study_time":study_time,"historical_engagement_level": "High" if base >= 10 else "Medium" if base >= 7 else "Developing"})
        level = rng.normal(0, .30)
        series = []
        for w in range(N_WEEKS):
            t = w / (N_WEEKS - 1)
            trend = 0.0
            if pattern == "gradual": trend = -8.2 * t
            elif pattern == "sudden" and w >= 22: trend = -10.5 * min(1, (w - 21) / 4)
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
            support = int(rng.binomial(1, .035 + (.13 if pattern == "sudden" else 0)))
            days_idle = int(np.clip(rng.gamma(2.1, 4.2) + (28 if sessions == 0 else max(0, 18 - sessions * 1.2)), 0, 90))
            series.append({"customer_id":cid,"week":w+1,"weekly_sessions":sessions,"active_days":active_days,"usage_minutes":minutes,"completed_lessons":lessons,"completed_labs":labs,"searches":searches,"support_requests":support,"days_since_last_activity":days_idle,"subscription_age_weeks":int(rng.integers(12, 105)),"cancelled":0})
        frame = pd.DataFrame(series)
        recent = frame.tail(8)
        slope = float(np.polyfit(np.arange(len(recent)), recent.usage_minutes, 1)[0])
        inactivity = float(recent.days_since_last_activity.mean())
        # Cancellation is sampled from an engagement-driven propensity with overlapping outcomes.
        logit = -2.05 - .27 * slope + .018 * (inactivity - 18) + .17 * max(0, 7 - recent.weekly_sessions.mean())
        if pattern == "sudden": logit += .48
        if pattern == "recovering": logit -= .52
        p_cancel = 1 / (1 + np.exp(-logit))
        cancelled = int(rng.random() < p_cancel)
        if cancelled:
            frame.loc[frame.index[-1], "cancelled"] = 1
        # Keep curated demo personas reproducible through behavior only, never forced labels.
        if i == 0:  # stable core demo history with steady improvement, not a prescribed model result
            frame.loc[frame.index[-8:], "weekly_sessions"] = [14, 14, 14, 15, 15, 16, 16, 17]
            frame.loc[frame.index[-8:], "usage_minutes"] = [390, 394, 398, 410, 420, 433, 446, 460]
            frame.loc[frame.index[-8:], "active_days"] = [5, 5, 5, 5, 6, 6, 6, 6]
            frame.loc[frame.index[-8:], "days_since_last_activity"] = [4, 3, 3, 2, 2, 1, 1, 0]
            recent = frame.tail(8)
            slope = float(np.polyfit(np.arange(8), recent.usage_minutes, 1)[0])
            logit = -2.05 - .27*slope + .018*(recent.days_since_last_activity.mean()-18) + .17*max(0,7-recent.weekly_sessions.mean())
            frame.loc[frame.index[-1], "cancelled"] = int(rng.random() < 1/(1+np.exp(-logit)))
        if i == 2:  # a designed high-to-low engagement history; label remains sampled from behavior
            frame.loc[frame.index[-8:], "weekly_sessions"] = [12, 11, 9, 7, 5, 3, 1, 0]
            frame.loc[frame.index[-8:], "usage_minutes"] = [330, 292, 258, 205, 155, 92, 35, 4]
            frame.loc[frame.index[-8:], "active_days"] = [6, 6, 5, 4, 3, 2, 1, 0]
            frame.loc[frame.index[-8:], "completed_lessons"] = [7, 6, 5, 4, 3, 2, 1, 0]
            frame.loc[frame.index[-8:], "days_since_last_activity"] = [1, 2, 4, 7, 12, 19, 32, 52]
            slope = float(np.polyfit(np.arange(8), frame.tail(8).usage_minutes, 1)[0])
            recent = frame.tail(8)
            logit = -2.05 - .27*slope + .018*(recent.days_since_last_activity.mean()-18) + .17*max(0,7-recent.weekly_sessions.mean()) + .48
            frame.loc[frame.index[-1], "cancelled"] = int(rng.random() < 1/(1+np.exp(-logit)))
        weekly.extend(frame.to_dict("records"))
    DATA.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(weekly)
    profile_df = pd.DataFrame(profiles)
    history.to_csv(DATA / "customer_weekly_history.csv", index=False)
    profile_df.to_csv(DATA / "customer_profiles.csv", index=False)
    return history, profile_df


if __name__ == "__main__":
    h, p = generate()
    print(f"Generated {len(p)} profiles and {len(h)} weekly records at {DATA}")
