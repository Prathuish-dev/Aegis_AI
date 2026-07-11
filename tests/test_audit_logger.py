import os
import shutil
import tempfile
import unittest
import yaml
from datetime import datetime
from src.healing.audit_logger import AuditLogger

class TestAuditLogger(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test database and config
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_audit.db")
        self.config_path = os.path.join(self.test_dir, "test_settings.yaml")
        
        # Create a test settings.yaml
        settings = {
            "database": {
                "log_db_path": self.db_path
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(settings, f)
            
        self.audit_logger = AuditLogger(config_path=self.config_path, db_path=self.db_path)

    def tearDown(self):
        # Clean up temporary directory and files
        shutil.rmtree(self.test_dir)

    def test_table_initialization(self):
        # Verify database file is created
        self.assertTrue(os.path.exists(self.db_path))

    def test_log_action_and_get_history(self):
        # Log a sample action
        agent_id = "agent_d"
        action = "rewrite_prompt"
        outcome = "success"
        details = {"original_prompt": "Hello", "optimized_prompt": "Hello world"}
        
        self.audit_logger.log_action(agent_id, action, outcome, details)
        
        # Retrieve history
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        
        record = history[0]
        self.assertEqual(record["agent_id"], agent_id)
        self.assertEqual(record["action"], action)
        self.assertEqual(record["outcome"], outcome)
        self.assertEqual(record["details"], details)
        self.assertTrue(isinstance(record["timestamp"], str))
        self.assertEqual(record["id"], 1)

    def test_get_history_limit_and_ordering(self):
        # Log multiple actions
        for i in range(10):
            self.audit_logger.log_action(
                agent_id=f"agent_{i}",
                action="test_action",
                outcome="success",
                details={"index": i}
            )
            
        # Get history with limit 5
        history = self.audit_logger.get_audit_history(limit=5)
        self.assertEqual(len(history), 5)
        
        # Verify ordering (newest first, i.e. index 9, 8, 7, 6, 5)
        self.assertEqual(history[0]["agent_id"], "agent_9")
        self.assertEqual(history[0]["details"]["index"], 9)
        self.assertEqual(history[4]["agent_id"], "agent_5")
        self.assertEqual(history[4]["details"]["index"], 5)

    def test_log_action_without_details(self):
        self.audit_logger.log_action(
            agent_id="agent_a",
            action="init_system",
            outcome="completed"
        )
        
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        self.assertIsNone(history[0]["details"])

if __name__ == '__main__':
    unittest.main()
