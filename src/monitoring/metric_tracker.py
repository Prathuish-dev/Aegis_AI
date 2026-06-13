import os
import yaml
import datetime as dt
from datetime import datetime
from src.monitoring.models import MetricType, AnomalyEvent

class MetricTracker:
    def __init__(self, config_path: str = "config/settings.yaml", log_collector=None):
        self.config_path = config_path
        self.log_collector = log_collector
        self.history = {}  # (system_id, metric_type) -> list of (timestamp, value)
        self.callbacks = []
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from settings.yaml if it exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
        return {}

    def get_threshold_config(self, metric_type: MetricType) -> dict:
        """Get the threshold configuration for a specific metric type."""
        thresholds = self.config.get("thresholds", {})
        metric_key = metric_type.value if hasattr(metric_type, "value") else str(metric_type)
        if metric_key in thresholds:
            return thresholds[metric_key]
        
        # Default fallback threshold settings if not configured
        defaults = {
            "accuracy": {"operator": "delta_pct", "threshold": -0.05, "severity": "high", "window_seconds": 3600},
            "latency": {"operator": "gt", "threshold": 2000.0, "severity": "medium", "window_seconds": 3600},
            "error_rate": {"operator": "gt", "threshold": 0.02, "severity": "critical", "window_seconds": 3600}
        }
        return defaults.get(metric_key, {"operator": "gt", "threshold": float("inf"), "severity": "medium", "window_seconds": 3600})

    def register_callback(self, callback_func):
        """Register a callback function to be invoked when an anomaly event is triggered."""
        self.callbacks.append(callback_func)

    def record_metric(
        self,
        system_id: str,
        metric_type: MetricType,
        value: float,
        timestamp: datetime = None,
        baseline_value: float = None,
        trigger_alerts: bool = True
    ) -> list[AnomalyEvent]:
        """Record a new metric value, clean up the sliding window, and check thresholds."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Ensure metric_type is of type MetricType enum
        if isinstance(metric_type, str):
            metric_type = MetricType(metric_type)

        key = (system_id, metric_type)
        if key not in self.history:
            self.history[key] = []

        self.history[key].append((timestamp, float(value)))
        
        # Sort and clean sliding window
        self._clean_window(system_id, metric_type)

        # Check thresholds if alerting is enabled
        if trigger_alerts:
            events = self.check_thresholds(system_id, metric_type, baseline_value)
            for event in events:
                # Ingest anomaly event into database if log_collector is available
                if self.log_collector:
                    try:
                        event_dict = event.model_dump() if hasattr(event, "model_dump") else event.dict()
                        # Ensure timestamp is serialized as a string
                        event_dict["timestamp"] = event.timestamp.isoformat()
                        self.log_collector.ingest_log(event_dict)
                    except Exception as e:
                        print(f"Error persisting anomaly event: {e}")

                # Invoke registered callbacks
                for cb in self.callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        print(f"Error in metric tracker callback: {e}")
            return events

        return []

    def _clean_window(self, system_id: str, metric_type: MetricType):
        """Remove metrics older than the configured window size from history."""
        key = (system_id, metric_type)
        if key not in self.history or not self.history[key]:
            return
            
        config = self.get_threshold_config(metric_type)
        window_seconds = config.get("window_seconds", 3600)
        
        # Sort by timestamp chronologically
        self.history[key] = sorted(self.history[key], key=lambda x: x[0])
        
        latest_time = self.history[key][-1][0]
        cutoff_time = latest_time - dt.timedelta(seconds=window_seconds)
        
        self.history[key] = [item for item in self.history[key] if item[0] >= cutoff_time]

    def get_window_values(self, system_id: str, metric_type: MetricType) -> list[float]:
        """Get all values in the current sliding window."""
        key = (system_id, metric_type)
        if key not in self.history or not self.history[key]:
            return []
        return [val for _, val in self.history[key]]

    def calculate_sliding_average(self, system_id: str, metric_type: MetricType) -> float:
        """Calculate the average of values in the current sliding window."""
        values = self.get_window_values(system_id, metric_type)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def calculate_p95(self, system_id: str, metric_type: MetricType) -> float:
        """Calculate the 95th percentile value in the current sliding window."""
        values = self.get_window_values(system_id, metric_type)
        if not values:
            return 0.0
        try:
            import numpy as np
            return float(np.percentile(values, 95))
        except ImportError:
            # Fallback pure-Python percentile calculation
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * 0.95)
            return float(sorted_vals[min(idx, len(sorted_vals) - 1)])

    def get_stats(self, system_id: str, metric_type: MetricType) -> dict:
        """Retrieve current statistics for the sliding window."""
        values = self.get_window_values(system_id, metric_type)
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        
        metric_key = metric_type.value if hasattr(metric_type, "value") else str(metric_type)
        stats = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        if metric_key == "latency":
            stats["p95"] = self.calculate_p95(system_id, metric_type)
        return stats

    def check_thresholds(
        self,
        system_id: str,
        metric_type: MetricType,
        baseline_value: float = None
    ) -> list[AnomalyEvent]:
        """Evaluate current window statistics against thresholds and return AnomalyEvents if breached."""
        metric_key = metric_type.value if hasattr(metric_type, "value") else str(metric_type)
        config = self.get_threshold_config(metric_type)
        
        window_vals = self.get_window_values(system_id, metric_type)
        if not window_vals:
            return []
            
        operator = config.get("operator", "gt")
        threshold = config.get("threshold", 0.0)
        severity = config.get("severity", "medium")
        
        # Calculate current statistics for comparison
        if metric_key == "latency":
            current_value = self.calculate_p95(system_id, metric_type)
        else:
            current_value = self.calculate_sliding_average(system_id, metric_type)
            
        breached = False
        
        if operator == "gt":
            breached = current_value >= threshold
        elif operator == "lt":
            breached = current_value <= threshold
        elif operator == "delta_pct":
            # For delta_pct, baseline_value is needed
            if baseline_value is None:
                baseline_value = config.get("baseline", 1.0)
                
            if baseline_value != 0:
                delta_pct = (current_value - baseline_value) / baseline_value
                # If threshold is e.g. -0.05, drop >= 5% means delta_pct <= -0.05
                target_threshold = -abs(threshold)
                breached = delta_pct <= target_threshold
            else:
                breached = False
                
        if breached:
            import uuid
            latest_time = self.history[(system_id, metric_type)][-1][0]
            
            event = AnomalyEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                system_id=system_id,
                metric_type=metric_type,
                current_value=float(current_value),
                baseline_value=float(baseline_value if baseline_value is not None else threshold),
                timestamp=latest_time,
                raw_context={
                    "operator": operator,
                    "threshold": threshold,
                    "severity": severity,
                    "window_values_count": len(window_vals),
                    "last_values": window_vals[-5:]
                },
                severity=severity
            )
            return [event]
            
        return []

    def poll_from_db(self, system_id: str = None):
        """Poll historical metrics from LogCollector SQLite database and populate tracker history."""
        if not self.log_collector:
            return
            
        logs = self.log_collector.get_logs(system_id=system_id)
        
        for log in logs:
            try:
                sys_id = log.get("system_id")
                metric_str = log.get("metric_type")
                
                # Retrieve numeric value
                value = log.get("value")
                if value is None:
                    value = log.get("current_value")
                
                if value is not None and sys_id and metric_str:
                    try:
                        metric_type = MetricType(metric_str)
                    except ValueError:
                        continue
                        
                    timestamp_str = log.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    else:
                        timestamp = datetime.utcnow()
                        
                    # Record metric without triggering callbacks or re-saving to DB
                    self.record_metric(
                        system_id=sys_id,
                        metric_type=metric_type,
                        value=float(value),
                        timestamp=timestamp,
                        trigger_alerts=False
                    )
            except Exception as e:
                print(f"Error parsing log during DB polling: {e}")
