import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from src.healing.verifier import Verifier
from src.healing.audit_logger import AuditLogger
from src.monitoring.metric_tracker import MetricTracker
from src.monitoring.models import MetricType, AnomalyEvent

class TestVerifier(unittest.TestCase):
    def setUp(self):
        # Setup temp DB and configurations
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_verify_events.db")
        self.config_path = os.path.join(self.temp_dir.name, "test_settings.yaml")
        
        self.audit_logger = AuditLogger(config_path=self.config_path, db_path=self.db_path)
        self.verifier = Verifier(audit_logger=self.audit_logger)
        
        # Mock MetricTracker
        self.mock_tracker = MagicMock(spec=MetricTracker)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_health_healthy(self):
        # Empty list of breach events means healthy
        self.mock_tracker.check_thresholds.return_value = []
        is_healthy = self.verifier.check_health("sys_01", MetricType.LATENCY, self.mock_tracker)
        self.assertTrue(is_healthy)
        self.mock_tracker.check_thresholds.assert_called_once_with("sys_01", MetricType.LATENCY, None)

    def test_check_health_unhealthy(self):
        # Breach events present means unhealthy
        mock_event = MagicMock(spec=AnomalyEvent)
        self.mock_tracker.check_thresholds.return_value = [mock_event]
        is_healthy = self.verifier.check_health("sys_01", MetricType.LATENCY, self.mock_tracker)
        self.assertFalse(is_healthy)

    @patch("time.sleep")
    @patch("time.time")
    def test_verify_immediate_success(self, mock_time, mock_sleep):
        # Mock time progression
        mock_time.side_effect = [100.0, 100.5]
        
        # System is healthy immediately
        self.mock_tracker.check_thresholds.return_value = []
        
        healed = self.verifier.verify(
            system_id="sys_01",
            metric_type=MetricType.LATENCY,
            metric_tracker=self.mock_tracker,
            check_interval_seconds=1.0,
            max_wait_seconds=5.0
        )
        
        self.assertTrue(healed)
        mock_sleep.assert_not_called()
        
        # Check audit log
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action"], "post_fix_verify")
        self.assertEqual(history[0]["outcome"], "healed")

    @patch("time.sleep")
    @patch("time.time")
    def test_verify_delayed_success(self, mock_time, mock_sleep):
        # 1st call: returns an event (unhealthy), time = 100.0
        # 2nd call: returns no events (healthy), time = 101.5
        mock_time.side_effect = [100.0, 101.5, 101.5]
        
        mock_event = MagicMock(spec=AnomalyEvent)
        self.mock_tracker.check_thresholds.side_effect = [[mock_event], []]
        
        healed = self.verifier.verify(
            system_id="sys_01",
            metric_type=MetricType.LATENCY,
            metric_tracker=self.mock_tracker,
            check_interval_seconds=1.0,
            max_wait_seconds=5.0
        )
        
        self.assertTrue(healed)
        mock_sleep.assert_called_once_with(1.0)
        self.assertEqual(self.mock_tracker.check_thresholds.call_count, 2)
        
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["outcome"], "healed")

    @patch("time.sleep")
    @patch("time.time")
    def test_verify_timeout_unresolved(self, mock_time, mock_sleep):
        # Simulate time exceeding max_wait_seconds
        # 1st loop: time = 100.0 (elapsed = 0.0) -> unhealthy
        # 2nd loop: time = 106.0 (elapsed = 6.0 > max_wait) -> loop terminates
        mock_time.side_effect = [100.0, 106.0]
        
        mock_event = MagicMock(spec=AnomalyEvent)
        self.mock_tracker.check_thresholds.return_value = [mock_event]
        
        healed = self.verifier.verify(
            system_id="sys_01",
            metric_type=MetricType.LATENCY,
            metric_tracker=self.mock_tracker,
            check_interval_seconds=1.0,
            max_wait_seconds=5.0
        )
        
        self.assertFalse(healed)
        mock_sleep.assert_called_once_with(1.0)
        
        history = self.audit_logger.get_audit_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["outcome"], "unresolved")

if __name__ == "__main__":
    unittest.main()
