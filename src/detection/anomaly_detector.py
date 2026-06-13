from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from src.monitoring.models import AnomalyEvent, MetricType, FailureCategory
from src.detection.rule_engine import RuleEngine
from src.detection.drift_detector import DriftDetector
from src.detection.data_issue_detector import DataIssueDetector
from src.detection.model_performance_detector import ModelPerformanceDetector
from src.detection.relevance_scorer import RelevanceScorer
from src.detection.system_issue_detector import SystemIssueDetector

class AnomalyDetector:
    """Unified facade class for failures, anomalies, and drift detection across the system."""

    def __init__(self, system_id: str = "default-system", config_path: Optional[str] = None):
        """Initializes the AnomalyDetector facade.
        
        Args:
            system_id: Identifier for the current monitored pipeline/system.
            config_path: Custom configuration path for threshold rules.
        """
        self.system_id = system_id
        self.rule_engine = RuleEngine(config_path=config_path)
        self.drift_detector = DriftDetector()
        self.data_issue_detector = DataIssueDetector()
        self.model_performance_detector = ModelPerformanceDetector()
        self.relevance_scorer = RelevanceScorer()
        self.system_issue_detector = SystemIssueDetector()

    def check_system_log(self, log_message: str) -> Optional[AnomalyEvent]:
        """Scans a raw log message for system/infrastructure issues.
        
        Args:
            log_message: The raw text of a log line.
            
        Returns:
            An AnomalyEvent if a system issue is matched, else None.
        """
        match = self.system_issue_detector.detect_from_log(log_message)
        if match:
            raw_context = {
                "failure_category": FailureCategory.SYSTEM_ISSUE.value,
                "failure_subcategory": match["type"],
                "suggested_fix": match["suggested_fix"],
                "log_snippet": match["log_snippet"],
                "context_for_rag": f"System issue detected: {match['type']} (severity: {match['severity']}). Log snippet: {match['log_snippet']}"
            }
            # System issues don't have standard numerical values, so use 1.0/0.0 representation
            return AnomalyEvent(
                event_id=str(uuid.uuid4()),
                system_id=self.system_id,
                metric_type=MetricType.ERROR_RATE,
                current_value=1.0,
                baseline_value=0.0,
                timestamp=datetime.utcnow(),
                raw_context=raw_context,
                severity=match["severity"]
            )
        return None

    def check_metric_threshold(self, metric_type: Union[str, MetricType], value: float, baseline: Optional[float] = None) -> List[AnomalyEvent]:
        """Evaluates a current metric value and baseline against defined threshold rules.
        
        Args:
            metric_type: Name or MetricType enum of the evaluated metric.
            value: Current observed value of the metric.
            baseline: Optional baseline value for percentage difference rules.
            
        Returns:
            A list of triggered AnomalyEvents (if any).
        """
        triggered_rules = self.rule_engine.evaluate(metric_type, value, baseline)
        events = []
        for rule in triggered_rules:
            # Map metric name to MetricType enum safely
            target_metric_str = metric_type.value if isinstance(metric_type, MetricType) else str(metric_type)
            m_type = MetricType.ERROR_RATE
            for mt in MetricType:
                if mt.value == target_metric_str:
                    m_type = mt
                    break

            raw_context = {
                "failure_category": FailureCategory.MODEL_ISSUE.value if m_type in [MetricType.ACCURACY, MetricType.LATENCY] else FailureCategory.SYSTEM_ISSUE.value,
                "failure_subcategory": rule.name,
                "evidence": {
                    "metric_name": rule.metric_type,
                    "current_value": value,
                    "baseline_value": baseline if baseline is not None else 0.0,
                    "threshold": rule.threshold
                },
                "context_for_rag": f"Metric {rule.metric_type} triggered rule {rule.name} (severity: {rule.severity}). Observed: {value}, Threshold: {rule.threshold}"
            }
            events.append(
                AnomalyEvent(
                    event_id=str(uuid.uuid4()),
                    system_id=self.system_id,
                    metric_type=m_type,
                    current_value=value,
                    baseline_value=baseline if baseline is not None else 0.0,
                    timestamp=datetime.utcnow(),
                    raw_context=raw_context,
                    severity=rule.severity
                )
            )
        return events

    def check_model_performance(self, metrics_history: List[Dict[str, Any]]) -> List[AnomalyEvent]:
        """Analyzes historical metrics for sliding-window performance degradation.
        
        Args:
            metrics_history: Chronological list of metric records.
            
        Returns:
            List of AnomalyEvents representing performance drops.
        """
        drops = self.model_performance_detector.check_performance_drop(metrics_history)
        events = []
        if drops:
            for metric, details in drops.items():
                raw_context = {
                    "failure_category": FailureCategory.MODEL_ISSUE.value,
                    "failure_subcategory": f"{metric}_drop",
                    "evidence": {
                        "metric_name": metric,
                        "current_value": details["current"],
                        "baseline_value": details["baseline"],
                        "change": -details["drop"]
                    },
                    "context_for_rag": f"Model performance dropped for metric {metric} from baseline {details['baseline']:.4f} to current {details['current']:.4f} (drop of {details['drop']:.4f})."
                }
                events.append(
                    AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        system_id=self.system_id,
                        metric_type=MetricType.ACCURACY,
                        current_value=details["current"],
                        baseline_value=details["baseline"],
                        timestamp=datetime.utcnow(),
                        raw_context=raw_context,
                        severity=details["severity"]
                    )
                )
        return events

    def check_llm_relevance(self, query: str, response: str, threshold: float = 0.5) -> Optional[AnomalyEvent]:
        """Checks LLM response relevance against query.
        
        Args:
            query: The user input query.
            response: The generated response from the LLM.
            threshold: Minimum relevance similarity score (default: 0.5).
            
        Returns:
            An AnomalyEvent if low relevance is detected, else None.
        """
        score = self.relevance_scorer.score(query, response)
        if score < threshold:
            raw_context = {
                "failure_category": FailureCategory.PROMPT_ISSUE.value,
                "failure_subcategory": "low_relevance",
                "evidence": {
                    "metric_name": "llm_relevance",
                    "current_value": score,
                    "baseline_value": threshold
                },
                "context_for_rag": f"LLM relevance score of {score:.4f} fell below threshold {threshold:.4f}. Query: '{query}', Response: '{response}'"
            }
            return AnomalyEvent(
                event_id=str(uuid.uuid4()),
                system_id=self.system_id,
                metric_type=MetricType.LLM_RELEVANCE,
                current_value=score,
                baseline_value=threshold,
                timestamp=datetime.utcnow(),
                raw_context=raw_context,
                severity="high" if score < (threshold - 0.2) else "medium"
            )
        return None

    def check_data_drift(self, reference_df: pd.DataFrame, production_df: pd.DataFrame) -> List[AnomalyEvent]:
        """Checks dataset features for statistical covariate distribution drift.
        
        Args:
            reference_df: Baseline historical dataframe.
            production_df: Current production dataframe.
            
        Returns:
            List of AnomalyEvents representing drifted features.
        """
        drift_results = self.data_issue_detector.detect_covariate_drift(reference_df, production_df)
        events = []
        for col, res in drift_results.items():
            if res["drift"]:
                raw_context = {
                    "failure_category": FailureCategory.DATA_ISSUE.value,
                    "failure_subcategory": "covariate_drift",
                    "evidence": {
                        "feature_name": col,
                        "test_type": res["test"],
                        "statistic": res["statistic"],
                        "p_value": res["p_value"]
                    },
                    "context_for_rag": f"Data drift detected in feature '{col}' using {res['test']} test (statistic={res['statistic']:.4f}, p-value={res['p_value']:.4f})."
                }
                events.append(
                    AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        system_id=self.system_id,
                        metric_type=MetricType.ERROR_RATE,  # Fallback
                        current_value=res["p_value"],
                        baseline_value=0.05,  # Alpha threshold
                        timestamp=datetime.utcnow(),
                        raw_context=raw_context,
                        severity="high" if res["p_value"] < 0.01 else "medium"
                    )
                )
        return events

    def check_schema_drift(self, reference_schema: Dict[str, Any], production_schema: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Checks schemas for missing or new columns and type mismatches.
        
        Args:
            reference_schema: Schema specification for baseline data.
            production_schema: Schema specification for production data.
            
        Returns:
            An AnomalyEvent if schema drift is found, else None.
        """
        drift = self.data_issue_detector.detect_schema_drift(reference_schema, production_schema)
        if drift["missing_columns"] or drift["new_columns"] or drift["type_changes"]:
            raw_context = {
                "failure_category": FailureCategory.DATA_ISSUE.value,
                "failure_subcategory": "schema_drift",
                "evidence": drift,
                "context_for_rag": f"Schema drift detected. Missing columns: {drift['missing_columns']}. New columns: {drift['new_columns']}. Type changes: {drift['type_changes']}."
            }
            return AnomalyEvent(
                event_id=str(uuid.uuid4()),
                system_id=self.system_id,
                metric_type=MetricType.ERROR_RATE,
                current_value=1.0,
                baseline_value=0.0,
                timestamp=datetime.utcnow(),
                raw_context=raw_context,
                severity="critical" if drift["missing_columns"] else "medium"
            )
        return None
