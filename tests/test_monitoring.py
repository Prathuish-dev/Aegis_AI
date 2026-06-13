import unittest
import os
import shutil
import tempfile
import yaml
from datetime import datetime, timedelta
from src.monitoring.models import MetricType, FailureCategory, AnomalyEvent
from src.monitoring.log_collector import LogCollector
from src.monitoring.metric_tracker import MetricTracker

class TestMonitoringSystem(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test database and config
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_events.db")
        self.config_path = os.path.join(self.test_dir, "test_settings.yaml")
        
        # Create a test settings.yaml file
        settings = {
            "database": {
                "log_db_path": self.db_path
            },
            "thresholds": {
                "accuracy": {
                    "metric_type": "accuracy",
                    "operator": "delta_pct",
                    "threshold": -0.05,
                    "severity": "high",
                    "window_seconds": 60
                },
                "latency": {
                    "metric_type": "latency",
                    "operator": "gt",
                    "threshold": 2000.0,
                    "severity": "medium",
                    "window_seconds": 60
                },
                "error_rate": {
                    "metric_type": "error_rate",
                    "operator": "gt",
                    "threshold": 0.02,
                    "severity": "critical",
                    "window_seconds": 60
                }
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(settings, f)
            
        self.log_collector = LogCollector(db_path=self.db_path)
        self.metric_tracker = MetricTracker(config_path=self.config_path, log_collector=self.log_collector)

    def tearDown(self):
        # Close DB connections and cleanup files
        if hasattr(self, 'log_collector') and self.log_collector:
            self.log_collector.close()
        shutil.rmtree(self.test_dir)

    def test_log_collector_ingest_and_retrieve(self):
        log_entry = {
            "event_id": "evt_test_001",
            "system_id": "sys_main",
            "metric_type": "accuracy",
            "value": 0.92,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log_collector.ingest_log(log_entry)
        
        retrieved = self.log_collector.get_logs(system_id="sys_main", metric_type="accuracy")
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["event_id"], "evt_test_001")
        self.assertEqual(retrieved[0]["value"], 0.92)

    def test_metric_tracker_sliding_window_calculations(self):
        system_id = "sys_main"
        base_time = datetime.utcnow()
        
        # Record accuracy metrics (average should be 0.90)
        self.metric_tracker.record_metric(system_id, MetricType.ACCURACY, 0.88, base_time - timedelta(seconds=10), trigger_alerts=False)
        self.metric_tracker.record_metric(system_id, MetricType.ACCURACY, 0.92, base_time, trigger_alerts=False)
        
        avg = self.metric_tracker.calculate_sliding_average(system_id, MetricType.ACCURACY)
        self.assertAlmostEqual(avg, 0.90)
        
        # Record latency metrics (P95 of [1000, 1200, 1500, 2100])
        for val in [1000, 1200, 1500, 2100]:
            self.metric_tracker.record_metric(system_id, MetricType.LATENCY, val, base_time, trigger_alerts=False)
            
        p95 = self.metric_tracker.calculate_p95(system_id, MetricType.LATENCY)
        self.assertTrue(p95 >= 1500 and p95 <= 2100)

    def test_metric_tracker_clean_window(self):
        system_id = "sys_main"
        base_time = datetime.utcnow()
        
        # In settings.yaml, window is 60 seconds.
        # Record a metric 70 seconds ago, and another one now.
        self.metric_tracker.record_metric(system_id, MetricType.ERROR_RATE, 0.05, base_time - timedelta(seconds=70), trigger_alerts=False)
        self.metric_tracker.record_metric(system_id, MetricType.ERROR_RATE, 0.01, base_time, trigger_alerts=False)
        
        # Get active values in sliding window (the 70s ago one should be cleaned up)
        vals = self.metric_tracker.get_window_values(system_id, MetricType.ERROR_RATE)
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0], 0.01)

    def test_accuracy_drop_breach(self):
        # We test accuracy delta_pct drop.
        # Baseline = 0.95. Threshold = -0.05 (5% drop allowed).
        # We record 0.88 (a drop from 0.95: (0.88 - 0.95)/0.95 = -0.073 = -7.3% drop, which is a breach)
        system_id = "sys_main"
        events = self.metric_tracker.record_metric(
            system_id=system_id,
            metric_type=MetricType.ACCURACY,
            value=0.88,
            timestamp=datetime.utcnow(),
            baseline_value=0.95,
            trigger_alerts=True
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].metric_type, MetricType.ACCURACY)
        self.assertEqual(events[0].severity, "high")
        self.assertTrue(events[0].current_value < 0.90)

    def test_latency_breach_and_callback(self):
        system_id = "sys_main"
        callback_triggered = []
        
        def test_callback(event):
            callback_triggered.append(event)
            
        self.metric_tracker.register_callback(test_callback)
        
        # Record a latency value that breaches P95 threshold of 2000ms
        events = self.metric_tracker.record_metric(
            system_id=system_id,
            metric_type=MetricType.LATENCY,
            value=2500.0,
            timestamp=datetime.utcnow(),
            trigger_alerts=True
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(len(callback_triggered), 1)
        self.assertEqual(callback_triggered[0].event_id, events[0].event_id)
        self.assertEqual(callback_triggered[0].severity, "medium")

    def test_poll_from_db(self):
        # Ingest logs directly to SQLite
        self.log_collector.ingest_log({
            "event_id": "e1",
            "system_id": "sys_db",
            "metric_type": "error_rate",
            "value": 0.03,
            "timestamp": (datetime.utcnow() - timedelta(seconds=10)).isoformat()
        })
        self.log_collector.ingest_log({
            "event_id": "e2",
            "system_id": "sys_db",
            "metric_type": "error_rate",
            "value": 0.04,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Poll DB
        self.metric_tracker.poll_from_db(system_id="sys_db")
        
        # Verify stats populated in tracker
        stats = self.metric_tracker.get_stats("sys_db", MetricType.ERROR_RATE)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["mean"], 0.035)

if __name__ == '__main__':
    unittest.main()
