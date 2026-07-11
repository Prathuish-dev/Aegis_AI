import os
import unittest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock
from src.healing.fix_executor import FixExecutor
from src.healing.audit_logger import AuditLogger

class TestFixExecutor(unittest.TestCase):
    def setUp(self):
        # Create temp files for DB and configs to isolate tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_events.db")
        self.config_path = os.path.join(self.temp_dir.name, "test_settings.yaml")
        
        # Initialize isolated AuditLogger
        self.audit_logger = AuditLogger(config_path=self.config_path, db_path=self.db_path)
        
        # Create mock LLM
        self.mock_llm = MagicMock()
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_mode_resolution(self):
        # Should raise ValueError on invalid mode
        with self.assertRaises(ValueError):
            FixExecutor(mode="invalid_mode", audit_logger=self.audit_logger)
            
        # Defaults to suggest if env is missing
        if "FIX_MODE" in os.environ:
            del os.environ["FIX_MODE"]
        executor = FixExecutor(audit_logger=self.audit_logger)
        self.assertEqual(executor.mode, "suggest")
        
        # Uses env variable if present
        os.environ["FIX_MODE"] = "auto"
        executor_env = FixExecutor(audit_logger=self.audit_logger)
        self.assertEqual(executor_env.mode, "auto")
        del os.environ["FIX_MODE"]

    def test_suggest_mode(self):
        executor = FixExecutor(mode="suggest", audit_logger=self.audit_logger)
        fix_plan = {"action": "alert_retrain", "system_id": "sys_01"}
        
        result = executor.execute(fix_plan)
        
        self.assertEqual(result["status"], "suggested")
        self.assertEqual(result["plan"], fix_plan)
        
        # Verify action was logged in audit history
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action"], "suggest_alert_retrain")
        self.assertEqual(history[0]["outcome"], "suggested")

    def test_semi_auto_mode_no_approval(self):
        executor = FixExecutor(mode="semi-auto", audit_logger=self.audit_logger)
        fix_plan = {"action": "alert_retrain", "system_id": "sys_01"}
        
        result = executor.execute(fix_plan, approved=False)
        
        self.assertEqual(result["status"], "pending_approval")
        self.assertEqual(result["plan"], fix_plan)
        
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action"], "request_approval_alert_retrain")
        self.assertEqual(history[0]["outcome"], "pending_approval")

    def test_semi_auto_mode_with_approval(self):
        executor = FixExecutor(mode="semi-auto", audit_logger=self.audit_logger)
        fix_plan = {"action": "alert_retrain", "system_id": "sys_01"}
        
        result = executor.execute(fix_plan, approved=True)
        
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["alert_sent"])
        
        history = self.audit_logger.get_audit_history()
        # Should have log from the applied fix
        self.assertEqual(history[0]["action"], "alert_retrain")
        self.assertEqual(history[0]["outcome"], "success")

    def test_auto_mode(self):
        executor = FixExecutor(mode="auto", audit_logger=self.audit_logger)
        fix_plan = {"action": "alert_retrain", "system_id": "sys_01"}
        
        result = executor.execute(fix_plan)
        
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["alert_sent"])

    def test_action_rewrite_prompt_fallback(self):
        executor = FixExecutor(mode="auto", audit_logger=self.audit_logger)
        target_prompt_file = os.path.join(self.temp_dir.name, "prompts", "test_prompt.txt")
        
        fix_plan = {
            "action": "rewrite_prompt",
            "original_prompt": "Translate: {text}",
            "failure_reason": "Sometimes outputs explanation",
            "filepath": target_prompt_file
        }
        
        result = executor.execute(fix_plan)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("Aegis Auto-Correction Guardrails", result["optimized_prompt"])
        
        # Verify file was written
        self.assertTrue(os.path.exists(target_prompt_file))
        with open(target_prompt_file, "r") as f:
            content = f.read()
            self.assertIn("Translate: {text}", content)
            self.assertIn("Sometimes outputs explanation", content)

    def test_action_rewrite_prompt_with_llm(self):
        # Configure mock LLM return value
        class MockResponse:
            content = "This is a super optimized prompt output by LLM."
        self.mock_llm.invoke.return_value = MockResponse()
        
        executor = FixExecutor(mode="auto", audit_logger=self.audit_logger, llm=self.mock_llm)
        target_prompt_file = os.path.join(self.temp_dir.name, "prompts", "test_prompt_llm.txt")
        
        fix_plan = {
            "action": "rewrite_prompt",
            "original_prompt": "Original prompt text",
            "failure_reason": "Low quality output",
            "filepath": target_prompt_file
        }
        
        result = executor.execute(fix_plan)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["optimized_prompt"], "This is a super optimized prompt output by LLM.")
        
        # Verify mock LLM was invoked
        self.mock_llm.invoke.assert_called_once()

    def test_action_update_config(self):
        # Create initial yaml config
        initial_config = {
            "thresholds": {
                "drift_psi": 0.20,
                "accuracy": 0.80
            }
        }
        with open(self.config_path, "w") as f:
            yaml.safe_dump(initial_config, f)
            
        executor = FixExecutor(mode="auto", audit_logger=self.audit_logger)
        
        fix_plan = {
            "action": "update_config",
            "filepath": self.config_path,
            "config_key": "thresholds.drift_psi",
            "config_value": 0.35
        }
        
        result = executor.execute(fix_plan)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["config_value"], 0.35)
        
        # Reload and verify yaml was modified
        with open(self.config_path, "r") as f:
            updated_config = yaml.safe_load(f)
            self.assertEqual(updated_config["thresholds"]["drift_psi"], 0.35)
            self.assertEqual(updated_config["thresholds"]["accuracy"], 0.80) # Unchanged

if __name__ == "__main__":
    unittest.main()
