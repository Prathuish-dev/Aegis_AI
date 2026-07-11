from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class AegisAgentState(TypedDict):
    """
    Represents the state of the Aegis AI self-healing agent.
    
    This state is passed through the LangGraph state machine during the
    anomaly diagnosis and self-healing process.
    """
    # Input fields
    anomaly_description: str
    system_id: str
    failure_category: str
    raw_metrics: Dict[str, Any]

    # Processing fields
    retrieved_context: List[str]
    analysis_steps: List[str]

    # Output fields
    root_cause: str
    fix_recommendation: str
    confidence_score: float
    requires_human_review: bool

    # Control flow and memory
    iteration_count: int
    messages: Annotated[List[BaseMessage], operator.add]
