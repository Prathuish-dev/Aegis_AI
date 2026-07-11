import os
import unittest
import tempfile
import yaml
from datetime import datetime
from unittest.mock import MagicMock
from src.monitoring.metric_tracker import MetricTracker
from src.monitoring.models import MetricType, AnomalyEvent
from src.healing.audit_logger import AuditLogger
from src.healing.fix_executor import FixExecutor
from src.healing.verifier import Verifier
from src.agent.state import AegisAgentState
from src.agent.graph import build_debugging_agent

class TestHealingIntegration(unittest.TestCase):
    def setUp(self):
        # Create temp environment
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "integration_events.db")
        self.config_path = os.path.join(self.temp_dir.name, "integration_settings.yaml")
        
        # Write default settings
        default_settings = {
            "thresholds": {
                "accuracy": {
                    "operator": "delta_pct",
                    "threshold": -0.05,
                    "severity": "high",
                    "window_seconds": 3600
                }
            }
        }
        with open(self.config_path, "w") as f:
            yaml.safe_dump(default_settings, f)
            
        # Initialize components
        self.audit_logger = AuditLogger(config_path=self.config_path, db_path=self.db_path)
        self.metric_tracker = MetricTracker(config_path=self.config_path)
        self.verifier = Verifier(audit_logger=self.audit_logger)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_healing_loop_integration(self):
        # System details
        system_id = "churn_model_v1"
        metric_type = MetricType.ACCURACY
        
        # 1. Record baseline metric (0.90 accuracy)
        self.metric_tracker.record_metric(
            system_id=system_id,
            metric_type=metric_type,
            value=0.90,
            baseline_value=0.90,
            trigger_alerts=False
        )
        
        # 2. Record dropped metric (0.80 accuracy, drop = -11% > -5% threshold) -> triggers anomaly event
        events = self.metric_tracker.record_metric(
            system_id=system_id,
            metric_type=metric_type,
            value=0.80,
            baseline_value=0.90,
            trigger_alerts=True
        )
        
        self.assertEqual(len(events), 1)
        anomaly = events[0]
        self.assertEqual(anomaly.metric_type, metric_type)
        
        # 3. Simulate Agent Graph Diagnosis
        # Let's compile the graph and mock the LLM response
        class IntegrationMockLLM:
            def invoke(self, *args, **kwargs):
                class Content:
                    content = '{"root_cause": "data drift in churn model features", "failure_category": "model_issue", "fix_recommendation": "trigger model retraining", "confidence_score": 0.90}'
                return Content()
                
        app = build_debugging_agent()
        config = {
            "configurable": {
                "thread_id": "integration_thread",
                "llm": IntegrationMockLLM()
            }
        }
        
        agent_input = AegisAgentState(
            anomaly_description=f"Anomaly detected on {anomaly.metric_type} with value {anomaly.current_value}",
            system_id=system_id,
            failure_category="",
            raw_metrics={"current_value": anomaly.current_value, "baseline": anomaly.baseline_value},
            retrieved_context=[],
            analysis_steps=[],
            root_cause="",
            fix_recommendation="",
            confidence_score=0.0,
            requires_human_review=False,
            iteration_count=0,
            messages=[]
        )
        
        agent_output = app.invoke(agent_input, config=config)
        
        # Assert agent diagnosed correctly
        self.assertEqual(agent_output["failure_category"], "model_issue")
        self.assertEqual(agent_output["root_cause"], "data drift in churn model features")
        self.assertEqual(agent_output["fix_recommendation"], "trigger model retraining")
        self.assertFalse(agent_output["requires_human_review"]) # Confidence 90 >= 80 threshold
        
        # 4. Execute the Corrective Action using FixExecutor (Auto mode)
        executor = FixExecutor(mode="auto", audit_logger=self.audit_logger)
        
        fix_plan = {
            "action": "alert_retrain",
            "system_id": system_id,
            "metric_details": agent_output["raw_metrics"]
        }
        
        execution_result = executor.execute(fix_plan)
        
        self.assertEqual(execution_result["status"], "success")
        self.assertTrue(execution_result["alert_sent"])
        
        # 5. Verify healing recovery using Verifier
        # Record new healthy metrics (0.89 accuracy, drop is -1.1% which is within threshold)
        self.metric_tracker.record_metric(
            system_id=system_id,
            metric_type=metric_type,
            value=0.89,
            baseline_value=0.90,
            trigger_alerts=False
        )
        
        healed = self.verifier.verify(
            system_id=system_id,
            metric_type=metric_type,
            metric_tracker=self.metric_tracker,
            baseline_value=0.90,
            check_interval_seconds=0.1,
            max_wait_seconds=1.0
        )
        
        self.assertTrue(healed)
        
        # 6. Verify sqlite Audit Logs contains all entries (executor log, verifier log)
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 2)
        
        # Newest first
        self.assertEqual(history[0]["action"], "post_fix_verify")
        self.assertEqual(history[0]["outcome"], "healed")
        self.assertEqual(history[0]["details"]["status"], "resolved")
        
        self.assertEqual(history[1]["action"], "alert_retrain")
        self.assertEqual(history[1]["outcome"], "success")
        self.assertEqual(history[1]["details"]["system_id"], system_id)

if __name__ == "__main__":
    unittest.main()
