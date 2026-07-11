from typing import List, Dict, Optional

class ModelPerformanceDetector:
    """Detector for identifying model performance degradation over a sliding window."""

    def __init__(self, window_hours: int = 24, drop_threshold: float = 0.05):
        """Initializes the ModelPerformanceDetector with a sliding window size and drop threshold."""
        self.window_hours = window_hours
        self.drop_threshold = drop_threshold

    def check_performance_drop(self, metrics_history: List[Dict]) -> Optional[Dict]:
        """Compares the latest performance metrics to the baseline (the first metric in the window).
        
        Args:
            metrics_history: List of metric dicts ordered by timestamp (oldest first).
                             Each dict should contain keys like 'accuracy', 'f1', 'auc', and 'timestamp'.
                             
        Returns:
            A dictionary containing details of metric drops if any drop exceeds the threshold,
            otherwise None.
        """
        if len(metrics_history) < 2:
            return None

        # Compare current (latest) to baseline (first in window)
        baseline = metrics_history[0]
        current = metrics_history[-1]

        drops = {}
        for metric in ["accuracy", "f1", "auc"]:
            if metric in baseline and metric in current:
                if baseline[metric] is None or current[metric] is None:
                    continue
                
                delta = current[metric] - baseline[metric]
                # If the delta is a negative change worse than the threshold
                if delta < -self.drop_threshold:
                    drops[metric] = {
                        "baseline": float(baseline[metric]),
                        "current": float(current[metric]),
                        "drop": float(abs(delta)),
                        "severity": "high" if abs(delta) > 0.15 else "medium"
                    }

        return drops if drops else None
