from typing import List, Dict, Any, Optional

class ModelPerformanceDetector:
    """Detector for model performance degradation over a sliding window."""

    def __init__(self, window_hours: int = 24, drop_threshold: float = 0.05):
        """Initializes the ModelPerformanceDetector.
        
        Args:
            window_hours: Sliding window size in hours (for configuration reference).
            drop_threshold: Threshold change value to alert on (default: 0.05).
        """
        self.window_hours = window_hours
        self.drop_threshold = drop_threshold

    def check_performance_drop(self, metrics_history: List[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Compares current performance to baseline within the metrics history.
        
        Args:
            metrics_history: Chronological list of metric dictionary records.
                             Example: [{'timestamp': ..., 'accuracy': 0.85, 'f1': 0.80}]
                             
        Returns:
            Dictionary containing metrics that dropped beyond the threshold, or None if no drop.
        """
        if len(metrics_history) < 2:
            return None

        # Compare current (latest) to baseline (earliest)
        baseline = metrics_history[0]
        current = metrics_history[-1]

        drops = {}
        target_metrics = ["accuracy", "f1", "auc"]

        for metric in target_metrics:
            if metric in baseline and metric in current:
                val_baseline = baseline[metric]
                val_current = current[metric]
                
                if val_baseline is not None and val_current is not None:
                    delta = val_current - val_baseline
                    # A drop is a negative change
                    if delta < -self.drop_threshold:
                        drops[metric] = {
                            "baseline": float(val_baseline),
                            "current": float(val_current),
                            "drop": float(abs(delta)),
                            "severity": "high" if abs(delta) > 0.15 else "medium"
                        }

        return drops if drops else None
