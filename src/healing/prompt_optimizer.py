from typing import Any

class PromptOptimizer:
    """
    PromptOptimizer rewrites failing prompts using an LLM based on failure analysis.
    """
    def __init__(self, llm: Any):
        """
        Initialize the PromptOptimizer with an LLM.
        
        Args:
            llm: A LangChain-compatible LLM instance that supports .invoke().
        """
        self.llm = llm

    def optimize(self, original_prompt: str, failure_reason: str) -> str:
        """
        Rewrite a failing prompt based on the failure analysis and guidelines.
        
        Args:
            original_prompt: The prompt that failed or resulted in poor output.
            failure_reason: The description/analysis of why the prompt failed.
            
        Returns:
            The optimized/rewritten prompt as a string.
        """
        optimization_prompt = f"""You are an expert prompt engineer. Your task is to rewrite a failing prompt based on the provided failure reason to make it more robust, precise, and less prone to failures, while maintaining its original core intent.

ORIGINAL PROMPT (failing):
{original_prompt}

FAILURE REASON:
{failure_reason}

Rewrite instructions:
1. Make instructions more specific, clear, and unambiguous.
2. Include explicit output format instructions (e.g. JSON schema, structure) if appropriate.
3. Add explicit guardrails and negative constraints against the identified failure.
4. Keep similar intent and context.

Return ONLY the new optimized prompt, nothing else. Do not include markdown code blocks (e.g. ```) or any preamble/postamble.
"""
        response = self.llm.invoke(optimization_prompt)
        
        # Handle both BaseMessage objects and raw string responses
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()
