"""API-level checks for the project demo and its model evidence."""
import unittest
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

if __name__=='__main__':
    unittest.main()
