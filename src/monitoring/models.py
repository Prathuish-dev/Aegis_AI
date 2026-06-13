from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    LLM_RELEVANCE = "llm_relevance"

class FailureCategory(str, Enum):
    DATA_ISSUE = "data_issue"
    MODEL_ISSUE = "model_issue"
    PROMPT_ISSUE = "prompt_issue"
    RETRIEVAL_ISSUE = "retrieval_issue"
    SYSTEM_ISSUE = "system_issue"

class AnomalyEvent(BaseModel):
    event_id: str
    system_id: str
    metric_type: MetricType
    current_value: float
    baseline_value: float
    timestamp: datetime
    raw_context: dict
    severity: str  # "low", "medium", "high", "critical"
