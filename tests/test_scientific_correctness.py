"""Scientific, temporal, and agent-grounding regression checks."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.config import risk_label
from backend.customer_service import profile
from backend.decision_policy import eligible_actions
from backend.explainability import explanation
from backend.forecasting import forecast
from backend.llm_agent import (
    QA_SYSTEM_PROMPT, RECOMMENDATION_SYSTEM_PROMPT, _context,
    _numbers_are_grounded, answer, recommend,
)
from backend.main import app
from scripts.export_models import FEATURES, _forecast_with_fallback, engineer_features

ROOT = Path(__file__).resolve().parents[1]


class TemporalCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = pd.read_csv(ROOT / "data/customer_weekly_history.csv")
        cls.targets = pd.read_csv(ROOT / "data/customer_churn_targets.csv")

    def test_subscription_age_increases_each_week_for_every_customer(self):
        ordered = self.history.sort_values(["customer_id", "week"])
        increments = ordered.groupby("customer_id").subscription_age_weeks.diff().dropna()
        self.assertEqual(len(increments), 300 * 31)
        self.assertTrue((increments == 1).all())

    def test_classifier_features_do_not_change_when_future_window_changes(self):
        baseline = engineer_features(self.history).sort_values("customer_id").reset_index(drop=True)
        changed = self.history.copy()
        future = changed.week > 28
        for col in ("weekly_sessions", "active_days", "usage_minutes", "completed_lessons",
                    "completed_labs", "searches", "support_requests", "days_since_last_activity",
                    "subscription_age_weeks", "cancelled"):
            changed.loc[future, col] = changed.loc[future, col].fillna(0) + 10_000
        after = engineer_features(changed).sort_values("customer_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline, after)

    def test_target_is_customer_level_and_absent_from_classifier_schema(self):
        self.assertEqual(self.targets.customer_id.nunique(), 300)
        self.assertEqual(set(self.targets.will_churn_next_4_weeks.unique()), {0, 1})
        self.assertTrue((self.targets.prediction_cutoff_week == 28).all())
        self.assertTrue((self.targets.outcome_window_start_week == 29).all())
        self.assertTrue((self.targets.outcome_window_end_week == 32).all())
        features = engineer_features(self.history)
        self.assertNotIn("will_churn_next_4_weeks", features.columns)
        self.assertNotIn("cancelled", features.columns)
        self.assertNotIn("synthetic_churn_propensity", features.columns)
        self.assertFalse(set(FEATURES) & {"will_churn_next_4_weeks", "cancelled"})
        self.assertEqual(features.subscription_age_weeks.max(), self.history.loc[self.history.week == 28, "subscription_age_weeks"].max())

    def test_cutoff_features_capture_demo_behavior_without_exact_probabilities(self):
        f = engineer_features(self.history).set_index("customer_id")
        self.assertGreater(f.loc["C001", "usage_trend"], 0)
        self.assertLess(f.loc["C002", "usage_trend"], 0)
        self.assertLess(f.loc["C003", "usage_trend"], 0)
        self.assertGreater(f.loc["C004", "usage_trend"], 0)

    def test_exported_forecast_comparison_is_complete_and_selection_uses_mae(self):
        metrics = json.loads((ROOT / "models/metrics.json").read_text(encoding="utf-8"))["forecasting"]
        expected = {"naive", "exp_smoothing", "arima", "lag_regression"}
        self.assertEqual(set(metrics["metrics"]), expected)
        for result in metrics["metrics"].values():
            self.assertTrue({"mae", "rmse", "smape"}.issubset(result))
        winner = min(metrics["metrics"], key=lambda name: metrics["metrics"][name]["mae"])
        self.assertEqual(metrics["selected_model"], winner)

    def test_cluster_labels_and_rationales_are_unique_and_profile_based(self):
        import joblib
        bundle = joblib.load(ROOT / "models/pipeline.joblib")
        metrics = json.loads((ROOT / "models/metrics.json").read_text(encoding="utf-8"))["clustering"]
        labels = list(bundle["cluster_names"].values())
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(set(labels)), 5)
        self.assertEqual(set(labels), {"Power Users", "Core Engaged", "Fading Engagement", "Low Activity", "Irregular Users"})
        self.assertEqual(set(metrics["cluster_naming_rationale"]), set(bundle["cluster_stats"]))
        self.assertIn("overlapping", metrics["selection_reason"])

    def test_risk_threshold_boundaries(self):
        self.assertEqual(risk_label(0.0), "STABLE")
        self.assertEqual(risk_label(0.2999), "STABLE")
        self.assertEqual(risk_label(0.30), "WATCHLIST")
        self.assertEqual(risk_label(0.60), "AT RISK")
        self.assertEqual(risk_label(0.80), "CRITICAL")
        self.assertEqual(risk_label(1.0), "CRITICAL")


class ForecastCorrectnessTests(unittest.TestCase):
    def test_api_forecast_has_four_points_and_reports_global_and_used_models(self):
        result = forecast("C003")
        self.assertEqual(len(result["forecast"]), 4)
        self.assertIn(result["model_used"], {"naive", "exp_smoothing", "arima", "lag_regression"})
        self.assertIn(result["selected_global_model"], {"naive", "exp_smoothing", "arima", "lag_regression"})
        self.assertEqual(result["method"], result["model_used"])

    def test_forecast_falls_back_from_failing_models_to_naive(self):
        def broken(series, method):
            if method == "naive":
                return np.repeat(float(series[-1]), 4)
            raise ValueError("degenerate series")
        with patch("scripts.export_models.forecast_series", side_effect=broken):
            result, used, errors = _forecast_with_fallback(np.array([0.0, 0.0]), "arima")
        self.assertEqual(used, "naive")
        self.assertEqual(result.tolist(), [0.0] * 4)
        self.assertIn("arima", errors)
        self.assertIn("exp_smoothing", errors)

    def test_forecast_service_survives_low_activity_short_history(self):
        hist = pd.DataFrame({"customer_id": ["LOW", "LOW"], "week": [1, 2], "usage_minutes": [0.0, 0.0]})
        bundle = {"forecast_method": "arima", "forecast_spread": 0.0}
        metrics = {"forecasting": {"selected_model": "arima", "metrics": {}}}
        with patch("backend.forecasting._get_customer", return_value=(bundle, hist, None, None, metrics, None)):
            with patch("scripts.export_models.forecast_series", side_effect=lambda x, m: np.repeat(float(x[-1]), 4) if m == "naive" else (_ for _ in ()).throw(ValueError("fit failed"))):
                result = forecast("LOW")
        self.assertEqual(len(result["forecast"]), 4)
        self.assertEqual(result["model_used"], "naive")
        self.assertEqual(result["selected_global_model"], "arima")
        self.assertTrue(result["fallback"])


class PolicyAndAgentTests(unittest.TestCase):
    @staticmethod
    def fake_profile(tier="CRITICAL", **overrides):
        features = {"recent_support_requests": 0, "usage_trend": -40.0, "sessions_trend": -3.0, "usage_slope_recent": -5.0}
        features.update(overrides.pop("features", {}))
        prefs = {"favorite_topic": "Machine Learning", "preferred_content_type": "Interactive Labs",
                 "preferred_feature": "Interactive Labs", "historical_engagement_level": "High"}
        prefs.update(overrides.pop("preferences", {}))
        return {"risk": tier, "features": features, "preferences": prefs, "churn_probability": .87}

    def test_stable_customer_does_not_receive_aggressive_action(self):
        with patch("backend.decision_policy.profile", return_value=self.fake_profile("STABLE")):
            actions = eligible_actions("S")
        self.assertIn("NO_ACTION", actions)
        self.assertNotIn("PREMIUM_FEATURE_TRIAL", actions)
        self.assertNotIn("ONE_FREE_MONTH", actions)
        self.assertNotIn("RE_ENGAGEMENT_MESSAGE", actions)

    def test_repeated_support_requests_prioritize_human_support(self):
        p = self.fake_profile("CRITICAL", features={"recent_support_requests": 3})
        with patch("backend.decision_policy.profile", return_value=p):
            actions = eligible_actions("S")
        self.assertEqual(actions[0], "HUMAN_SUPPORT_OUTREACH")
        self.assertNotIn("PREMIUM_FEATURE_TRIAL", actions)
        self.assertNotIn("ONE_FREE_MONTH", actions)

    def test_preferred_interactive_labs_enables_matching_trial_for_declining_learner(self):
        p = self.fake_profile("AT RISK")
        with patch("backend.decision_policy.profile", return_value=p):
            actions = eligible_actions("S")
        self.assertIn("PREMIUM_FEATURE_TRIAL", actions)

    def test_forecast_decline_can_enable_preference_matched_trial(self):
        p = self.fake_profile("AT RISK", features={"usage_trend": -2.0, "sessions_trend": 0.0, "usage_slope_recent": 0.0})
        with patch("backend.decision_policy.profile", return_value=p):
            with patch("backend.forecasting.forecast", return_value={"expected_change_pct": -35.0}):
                actions = eligible_actions("S")
        self.assertIn("PREMIUM_FEATURE_TRIAL", actions)

    def test_qa_prompt_is_natural_language_without_json_requirement(self):
        with patch("backend.llm_agent._ollama", return_value="The classifier estimates risk from observed engagement evidence.") as call:
            result = answer("C001", "Why is this learner at risk?")
        self.assertIn("naturally", QA_SYSTEM_PROMPT)
        self.assertNotIn("Return only valid JSON", QA_SYSTEM_PROMPT)
        self.assertIsNone(result["recommendation"])
        self.assertIn("classifier", result["answer"].lower())
        self.assertFalse(call.call_args.kwargs["structured"])

    def test_recommendation_requires_structured_valid_eligible_action(self):
        allowed = eligible_actions("C003")
        response = json.dumps({"action": allowed[0], "reason": "This matches the verified eligibility policy.", "message": "Consider this option when convenient."})
        with patch("backend.llm_agent._ollama", return_value=response) as call:
            result = recommend("C003")
        self.assertIn("exactly the keys action, reason, and message", RECOMMENDATION_SYSTEM_PROMPT)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["action"], allowed[0])
        self.assertEqual(set(call.call_args.args[0][0]), {"role", "content"})
        self.assertTrue(call.call_args.kwargs["structured"])

    def test_model_cannot_return_action_outside_eligibility(self):
        response = json.dumps({"action": "UNLISTED_ACTION", "reason": "Unsupported action", "message": "Try this."})
        with patch("backend.llm_agent._ollama", return_value=response):
            result = recommend("C003")
        self.assertTrue(result["fallback"])
        self.assertIn(result["action"], result["eligible_actions"])

    def test_numeric_grounding_allows_catalog_trial_duration(self):
        context = _context("C003")
        self.assertTrue(_numbers_are_grounded("Offer a 7-day premium feature trial.", context))
        self.assertFalse(_numbers_are_grounded("Promise a 90-day trial.", context))

    def test_qa_numeric_grounding_uses_source_verified_values(self):
        context = _context("C001")
        self.assertTrue(_numbers_are_grounded("The learner is in the 25–34 age group.", context))
        self.assertTrue(_numbers_are_grounded("The learner is in the 25-34 age group.", context))
        self.assertFalse(_numbers_are_grounded("The chance is 91%.", context))

    def test_shap_values_are_customer_specific(self):
        one, three = explanation("C001"), explanation("C003")
        self.assertNotEqual([x["value"] for x in one["values"]], [x["value"] for x in three["values"]])
        self.assertEqual(one["method"], "SHAP")

    def test_customer_qa_context_does_not_reuse_previous_customer_conversation(self):
        captured = []
        with patch("backend.llm_agent._ollama", side_effect=lambda messages, structured=False: captured.append(messages) or "The classifier uses verified behavior."):
            answer("C001", "Why is risk changing?")
            answer("C002", "Show my evidence.")
        self.assertIn('"customer_id": "C002"', captured[1][0]["content"])
        self.assertNotIn("Why is risk changing?", captured[1][0]["content"])

    def test_chat_customer_switch_resets_frontend_conversation(self):
        source = (ROOT / "frontend/chat.js").read_text(encoding="utf-8")
        self.assertIn("function reset(customer){history=[]", source)
        self.assertIn("Chat.reset(id)", (ROOT / "frontend/app.js").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
