from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AegisAgentState
from src.agent.nodes import (
    retrieve_context_node,
    analyze_anomaly_node,
    classify_failure_node,
    generate_fix_node,
    human_review_node,
    apply_fix_node
)

def route_after_fix(state: AegisAgentState) -> str:
    """
    Conditional router to decide whether to pause for human review
    or proceed directly to apply the fix.
    
    Args:
        state: The current AegisAgentState.
        
    Returns:
        The name of the next node to route to ("human_review" or "apply_fix").
    """
    if state.get("requires_human_review", True):
        return "human_review"
    return "apply_fix"

def build_debugging_agent():
    """
    Compiles the Aegis AI self-healing agent state graph.
    
    The graph executes retrieve_context -> analyze_anomaly -> classify_failure -> generate_fix.
    It then conditionally routes to human_review (pausing execution via checkpointer)
    or directly to apply_fix, and finally terminates at END.
    
    Returns:
        A compiled LangGraph executable application.
    """
    # Create the graph builder
    builder = StateGraph(AegisAgentState)
    
    # Add nodes
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("analyze_anomaly", analyze_anomaly_node)
    builder.add_node("classify_failure", classify_failure_node)
    builder.add_node("generate_fix", generate_fix_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("apply_fix", apply_fix_node)
    
    # Define linear control flow edges
    builder.set_entry_point("retrieve_context")
    builder.add_edge("retrieve_context", "analyze_anomaly")
    builder.add_edge("analyze_anomaly", "classify_failure")
    builder.add_edge("classify_failure", "generate_fix")
    
    # Define conditional routing from generate_fix
    builder.add_conditional_edges(
        "generate_fix",
        route_after_fix,
        {
            "human_review": "human_review",
            "apply_fix": "apply_fix"
        }
    )
    
    # Outgoing edge from human_review (once resumed) to apply_fix
    builder.add_edge("human_review", "apply_fix")
    
    # Terminate after apply_fix
    builder.add_edge("apply_fix", END)
    
    # Initialize checkpoint memory for human-in-the-loop state preservation
    checkpointer = MemorySaver()
    
    # Compile graph with checkpointer and interrupt before human_review node
    app = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )
    
    return app
