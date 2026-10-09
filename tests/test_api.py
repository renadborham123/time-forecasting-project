"""API-level checks for the project demo and its model evidence."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app

class ChurnScopeApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None)

    def test_homepage_and_static_frontend_are_served(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/static/app.js').status_code,200)

    def test_population_and_segments_are_data_backed(self):
        population=self.client.get('/api/customers').json()
        self.assertEqual(len(population),300)
        segmented=self.client.post('/api/segment').json()
        self.assertEqual(segmented['count'],300)
        self.assertEqual(segmented['clusters'],5)

    def test_demo_customers_follow_designed_behavior_patterns(self):
        expected={'C001':'STABLE','C003':'CRITICAL','C004':'STABLE'}
        for customer_id,risk in expected.items():
            with self.subTest(customer_id=customer_id):
                result=self.client.get(f'/api/customer/{customer_id}')
                self.assertEqual(result.status_code,200)
                self.assertEqual(result.json()['risk'],risk)

    def test_forecast_and_shap_are_customer_specific_outputs(self):
        forecast=self.client.get('/api/customer/C003/forecast').json()
        explanation=self.client.get('/api/customer/C003/explanation').json()
        self.assertEqual(forecast['method'],'exp_smoothing')
        self.assertEqual(len(forecast['forecast']),4)
        self.assertLess(forecast['week_4_minutes'],forecast['current_weekly_usage_minutes'])
        self.assertEqual(explanation['method'],'SHAP')
        self.assertGreater(len(explanation['values']),0)

    def test_action_endpoint_enforces_eligibility_and_simulates(self):
        rejected=self.client.post('/api/customer/C001/execute-action',json={'action':'PREMIUM_FEATURE_TRIAL'})
        self.assertEqual(rejected.status_code,400)
        accepted=self.client.post('/api/customer/C001/execute-action',json={'action':'NO_ACTION'})
        self.assertEqual(accepted.status_code,200)
        self.assertEqual(accepted.json()['record']['status'],'scheduled (simulated)')

    def test_instant_questions_use_customer_evidence_without_calling_llm(self):
        with patch('backend.llm_agent._ollama') as llm:
            result=self.client.post('/api/customer/C003/chat',json={
                'question':'Why is this learner at risk?', 'quick':True,
            })
        self.assertEqual(result.status_code,200)
        llm.assert_not_called()
        body=result.json()
        self.assertIn('C003',body['answer'])
        self.assertIn('CRITICAL',body['answer'])
        self.assertIn('not causes',body['answer'])
        self.assertIsNone(body['recommendation'])

    def test_instant_drafts_respect_customer_action_eligibility(self):
        with patch('backend.llm_agent._ollama') as llm:
            stable=self.client.post('/api/customer/C001/chat',json={
                'question':'Write a personalized message', 'quick':True,
            }).json()
            critical=self.client.post('/api/customer/C003/chat',json={
                'question':'Write a personalized message', 'quick':True,
            }).json()
        llm.assert_not_called()
        self.assertEqual(stable['recommendation']['action'],'NO_ACTION')
        self.assertEqual(stable['recommendation']['message'],'')
        self.assertIn('No retention message',stable['answer'])
        action=critical['recommendation']
        self.assertIn(action['action'],action['eligible_actions'])
        self.assertTrue(action['message'])
        self.assertIn('No message has been sent',critical['answer'])

    def test_stage_actions_run_the_exported_models(self):
        from backend.customer_service import load_state
        bundle = load_state()[0]
        with patch.object(bundle['kmeans'], 'predict', wraps=bundle['kmeans'].predict) as segmentation:
            result = self.client.post('/api/segment').json()
        segmentation.assert_called_once()
        self.assertEqual(result['model'], 'K-Means')
        with patch.object(bundle['classifier'], 'predict_proba', wraps=bundle['classifier'].predict_proba) as classifier:
            result = self.client.post('/api/classify').json()
        classifier.assert_called_once()
        self.assertEqual(sum(result['counts'].values()), 300)

    def test_guide_is_grounded_and_validates_stages_and_customers(self):
        from backend.llm_agent import pipeline_guide
        pipeline_guide.cache_clear()
        with patch('backend.llm_agent._ollama', side_effect=RuntimeError('offline')):
            before = self.client.post('/api/pipeline-guide', json={'stage':1}).json()
            after = self.client.post('/api/pipeline-guide', json={'stage':3, 'completed':True, 'customer_id':'C003'}).json()
        self.assertTrue(before['fallback'])
        self.assertIn('K-Means', before['explanation'])
        self.assertIn('C003', after['explanation'])
        self.assertIn('-70%', after['explanation'])
        self.assertEqual(self.client.post('/api/pipeline-guide', json={'stage':5}).status_code, 422)
        self.assertEqual(self.client.post('/api/pipeline-guide', json={'stage':3, 'customer_id':'MISSING'}).status_code, 404)

    def test_guide_rejects_invented_numbers_and_accepts_grounded_provider_text(self):
        from backend.llm_agent import pipeline_guide
        pipeline_guide.cache_clear()
        with patch('backend.llm_agent._ollama', return_value='There are 9000 learners.'):
            self.assertTrue(pipeline_guide(0)['fallback'])
        pipeline_guide.cache_clear()
        with patch('backend.llm_agent._ollama', return_value='Load the sample learners, then look for their patterns.'):
            self.assertFalse(pipeline_guide(0)['fallback'])
        pipeline_guide.cache_clear()

    def test_instant_forecast_tells_the_selected_model_story(self):
        with patch('backend.llm_agent._ollama') as llm:
            result = self.client.post('/api/customer/C003/chat', json={'question':'Explain the usage forecast', 'quick':True}).json()
        llm.assert_not_called()
        self.assertIn('Exponential Smoothing', result['answer'])
        self.assertIn('32 weeks', result['answer'])
        self.assertIn('estimate, not a promise', result['answer'])

if __name__=='__main__':
    unittest.main()
