import uuid
from datetime import datetime
from typing import List, Dict, Union, Optional
import pandas as pd

from src.monitoring.models import AnomalyEvent, MetricType, FailureCategory
from src.detection.rule_engine import RuleEngine
from src.detection.drift_detector import DriftDetector
from src.detection.data_issue_detector import DataIssueDetector
from src.detection.model_performance_detector import ModelPerformanceDetector
from src.detection.relevance_scorer import RelevanceScorer
from src.detection.system_issue_detector import SystemIssueDetector

class AnomalyDetector:
    """Facade class integrating all Aegis AI failure detection modules.
    
    Provides unified APIs to scan metrics, dataframes, logs, and LLM outputs,
    returning standard AnomalyEvent schemas.
    """

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        drift_detector: Optional[DriftDetector] = None,
        data_detector: Optional[DataIssueDetector] = None,
        performance_detector: Optional[ModelPerformanceDetector] = None,
        relevance_scorer: Optional[RelevanceScorer] = None,
        system_detector: Optional[SystemIssueDetector] = None
    ):
        self.rule_engine = rule_engine or RuleEngine()
        self.drift_detector = drift_detector or DriftDetector()
        self.data_detector = data_detector or DataIssueDetector()
        self.performance_detector = performance_detector or ModelPerformanceDetector()
        self.relevance_scorer = relevance_scorer or RelevanceScorer()
        self.system_detector = system_detector or SystemIssueDetector()

    def check_metric(
        self,
        system_id: str,
        metric_type: Union[str, MetricType],
        value: float,
        baseline: Optional[float] = None
    ) -> List[AnomalyEvent]:
        """Evaluates numerical metrics against rule thresholds using the RuleEngine.
        
        Returns:
            A list of triggered AnomalyEvents.
        """
        triggered_rules = self.rule_engine.evaluate(metric_type, value, baseline)
        events = []
        
        metric_str = metric_type.value if isinstance(metric_type, MetricType) else str(metric_type)
        try:
            m_enum = MetricType(metric_str)
        except ValueError:
            m_enum = MetricType.ACCURACY  # fallback

        for rule in triggered_rules:
            # Map metric types to failure categories
            category = FailureCategory.SYSTEM_ISSUE.value
            if rule.metric_type == "accuracy":
                category = FailureCategory.MODEL_ISSUE.value
            elif rule.metric_type == "llm_relevance":
                category = FailureCategory.PROMPT_ISSUE.value

            events.append(
                AnomalyEvent(
                    event_id=str(uuid.uuid4()),
                    system_id=system_id,
                    metric_type=m_enum,
                    current_value=float(value),
                    baseline_value=float(baseline) if baseline is not None else 0.0,
                    timestamp=datetime.utcnow(),
                    severity=rule.severity,
                    raw_context={
                        "failure_category": category,
                        "failure_subcategory": rule.name,
                        "rule_operator": rule.operator,
                        "rule_threshold": rule.threshold,
                        "evidence": {
                            "metric_name": rule.metric_type,
                            "current_value": float(value),
                            "baseline_value": float(baseline) if baseline is not None else None,
                            "threshold": rule.threshold
                        },
                        "context_for_rag": f"Threshold rule '{rule.name}' triggered for metric '{rule.metric_type}'. "
                                           f"Current value: {value}, Baseline: {baseline}, Threshold: {rule.threshold}."
                    }
                )
            )
        return events

    def check_log(self, system_id: str, log_message: str) -> Optional[AnomalyEvent]:
        """Scans a log line for system infrastructure failures.
        
        Returns:
            An AnomalyEvent if a failure pattern matches, else None.
        """
        issue = self.system_detector.detect_from_log(log_message)
        if issue:
            return AnomalyEvent(
                event_id=str(uuid.uuid4()),
                system_id=system_id,
                metric_type=MetricType.ERROR_RATE,
                current_value=1.0,
                baseline_value=0.0,
                timestamp=datetime.utcnow(),
                severity=issue["severity"],
                raw_context={
                    "failure_category": FailureCategory.SYSTEM_ISSUE.value,
                    "failure_subcategory": issue["type"],
                    "suggested_fix": issue["suggested_fix"],
                    "log_snippet": issue["log_snippet"],
                    "context_for_rag": f"System infrastructure failure '{issue['type']}' detected from log snippet. "
                                       f"Suggested auto-fix: {issue['suggested_fix']}."
                }
            )
        return None

    def check_data_drift(
        self,
        system_id: str,
        reference_df: pd.DataFrame,
        production_df: pd.DataFrame
    ) -> List[AnomalyEvent]:
        """Evaluates covariate drift across all features in reference/production datasets.
        
        Returns:
            A list of AnomalyEvents for each feature showing statistically significant drift.
        """
        drift_results = self.data_detector.detect_covariate_drift(reference_df, production_df)
        events = []
        for col, info in drift_results.items():
            if info.get("drift", False):
                events.append(
                    AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        system_id=system_id,
                        metric_type=MetricType.ACCURACY,  # fallback as there's no data metric enum
                        current_value=float(info["p_value"]),
                        baseline_value=0.05,  # default alpha significance level
                        timestamp=datetime.utcnow(),
                        severity="high" if info["p_value"] < 0.01 else "medium",
                        raw_context={
                            "failure_category": FailureCategory.DATA_ISSUE.value,
                            "failure_subcategory": "covariate_drift",
                            "feature_name": col,
                            "test_type": info["test"],
                            "statistic": info["statistic"],
                            "p_value": info["p_value"],
                            "context_for_rag": f"Data drift detected in feature '{col}' using '{info['test']}' test "
                                               f"(p-value: {info['p_value']:.4f})."
                        }
                    )
                )
        return events

    def check_schema_drift(
        self,
        system_id: str,
        reference_schema: Dict[str, str],
        production_schema: Dict[str, str]
    ) -> Optional[AnomalyEvent]:
        """Checks for differences in column schemas (types, new, or missing columns).
        
        Returns:
            An AnomalyEvent if schema drift exists, else None.
        """
        drift = self.data_detector.detect_schema_drift(reference_schema, production_schema)
        if drift["missing_columns"] or drift["new_columns"] or drift["type_changes"]:
            # If columns are missing, classify as critical severity, otherwise medium
            severity = "critical" if drift["missing_columns"] else "medium"
            total_violations = len(drift["missing_columns"]) + len(drift["new_columns"]) + len(drift["type_changes"])
            
            return AnomalyEvent(
                event_id=str(uuid.uuid4()),
                system_id=system_id,
                metric_type=MetricType.ACCURACY,  # fallback
                current_value=float(total_violations),
                baseline_value=0.0,
                timestamp=datetime.utcnow(),
                severity=severity,
                raw_context={
                    "failure_category": FailureCategory.DATA_ISSUE.value,
                    "failure_subcategory": "schema_drift",
                    "missing_columns": drift["missing_columns"],
                    "new_columns": drift["new_columns"],
                    "type_changes": drift["type_changes"],
                    "context_for_rag": f"Schema drift detected. Missing columns: {drift['missing_columns']}, "
                                       f"New columns: {drift['new_columns']}, Type changes: {drift['type_changes']}."
                }
            )
        return None

    def check_performance(self, system_id: str, metrics_history: List[Dict]) -> List[AnomalyEvent]:
        """Checks for degradation in model performance metrics.
        
        Returns:
            A list of AnomalyEvents for each performance drop detected.
        """
        drops = self.performance_detector.check_performance_drop(metrics_history)
        events = []
        if drops:
            for metric, info in drops.items():
                try:
                    m_enum = MetricType(metric)
                except ValueError:
                    m_enum = MetricType.ACCURACY

                events.append(
                    AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        system_id=system_id,
                        metric_type=m_enum,
                        current_value=float(info["current"]),
                        baseline_value=float(info["baseline"]),
                        timestamp=datetime.utcnow(),
                        severity=info["severity"],
                        raw_context={
                            "failure_category": FailureCategory.MODEL_ISSUE.value,
                            "failure_subcategory": f"{metric}_drop",
                            "drop_amount": float(info["drop"]),
                            "context_for_rag": f"Model performance degradation detected for metric '{metric}'. "
                                               f"Current: {info['current']:.4f}, Baseline: {info['baseline']:.4f} "
                                               f"(drop of {info['drop']:.4f})."
                        }
                    )
                )
        return events

    def check_relevance(
        self,
        system_id: str,
        query: str,
        response: str,
        threshold: float = 0.5
    ) -> Optional[AnomalyEvent]:
        """Checks the semantic relevance between a query and an LLM response.
        
        Returns:
            An AnomalyEvent if the relevance is below the threshold, else None.
        """
        score = self.relevance_scorer.score(query, response)
        if score < threshold:
            return AnomalyEvent(
                event_id=str(uuid.uuid4()),
                system_id=system_id,
                metric_type=MetricType.LLM_RELEVANCE,
                current_value=score,
                baseline_value=threshold,
                timestamp=datetime.utcnow(),
                severity="high",
                raw_context={
                    "failure_category": FailureCategory.PROMPT_ISSUE.value,
                    "failure_subcategory": "low_relevance",
                    "query": query,
                    "response": response,
                    "context_for_rag": f"LLM Quality degradation: Low relevance score of {score:.4f} "
                                       f"(threshold: {threshold:.2f}) between query and response."
                }
            )
        return None
