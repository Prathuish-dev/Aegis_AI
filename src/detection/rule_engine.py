from dataclasses import dataclass
from typing import List, Union, Optional
import os
import yaml
from src.monitoring.models import MetricType

@dataclass
class DetectionRule:
    """Dataclass representing a failure detection rule."""
    name: str
    metric_type: Union[str, MetricType]
    operator: str  # "lt", "gt", "delta_pct"
    threshold: float
    severity: str
    window_seconds: int = 3600

DEFAULT_RULES = [
    DetectionRule("accuracy_drop", "accuracy", "delta_pct", -0.05, "high"),
    DetectionRule("high_latency", "latency", "gt", 2000.0, "medium"),
    DetectionRule("error_spike", "error_rate", "gt", 0.02, "critical"),
]

def load_rules_from_yaml(yaml_path: str) -> List[DetectionRule]:
    """Loads failure detection rules from a settings.yaml configuration file."""
    if not os.path.exists(yaml_path):
        return DEFAULT_RULES
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        thresholds = config.get("thresholds", {})
        if not thresholds:
            return DEFAULT_RULES
        
        rules = []
        for name, params in thresholds.items():
            rules.append(
                DetectionRule(
                    name=name,
                    metric_type=params.get("metric_type", name),
                    operator=params.get("operator", "gt"),
                    threshold=float(params.get("threshold", 0.0)),
                    severity=params.get("severity", "medium"),
                    window_seconds=int(params.get("window_seconds", 3600))
                )
            )
        return rules
    except Exception:
        return DEFAULT_RULES

class RuleEngine:
    """Engine responsible for evaluating system metrics against defined thresholds."""
    
    def __init__(self, rules: Optional[List[DetectionRule]] = None, config_path: Optional[str] = None):
        """Initializes the RuleEngine with rules or loads them from a configuration file.
        
        If both rules and config_path are None, it attempts to load from the default
        config/settings.yaml file in the project. If that fails, it falls back to DEFAULT_RULES.
        """
        if rules is not None:
            self.rules = rules
        elif config_path is not None:
            self.rules = load_rules_from_yaml(config_path)
        else:
            # Try to resolve default settings.yaml path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_path = os.path.join(base_dir, "config", "settings.yaml")
            self.rules = load_rules_from_yaml(default_path)

    def evaluate(self, metric_type: Union[str, MetricType], value: float, baseline: Optional[float] = None) -> List[DetectionRule]:
        """Returns all rules that were triggered for a given metric type."""
        triggered = []
        
        # Standardize metric type representation (as string)
        target_metric = metric_type.value if isinstance(metric_type, MetricType) else str(metric_type)
        
        for rule in self.rules:
            rule_metric = rule.metric_type.value if isinstance(rule.metric_type, MetricType) else str(rule.metric_type)
            if rule_metric == target_metric:
                if self._check(rule, value, baseline):
                    triggered.append(rule)
        return triggered

    def _check(self, rule: DetectionRule, value: float, baseline: Optional[float] = None) -> bool:
        """Evaluates a single rule's operator against the observed value and optional baseline.
        
        Operators supported:
            - 'lt': True if value < threshold
            - 'gt': True if value > threshold
            - 'delta_pct': True if the percentage change is less than the threshold (which is a negative value for drops).
                           Calculated as (value - baseline) / baseline. If baseline is None or 0, it falls back
                           to treating 'value' directly as the pre-calculated delta percentage.
        """
        if rule.operator == "lt":
            return value < rule.threshold
        elif rule.operator == "gt":
            return value > rule.threshold
        elif rule.operator == "delta_pct":
            if baseline is not None:
                if baseline == 0:
                    return False
                delta = (value - baseline) / baseline
            else:
                delta = value
            return delta < rule.threshold
        return False
