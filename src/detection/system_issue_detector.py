import re
from typing import Dict, Any, Optional

SYSTEM_FAILURE_PATTERNS = {
    "api_timeout": {
        "pattern": "HTTPTimeoutError|TimeoutError|ReadTimeout",
        "severity": "high",
        "fix": "exponential_backoff"
    },
    "rate_limit": {
        "pattern": "RateLimitError|429",
        "severity": "medium",
        "fix": "implement_retry_with_backoff"
    },
    "out_of_memory": {
        "pattern": "OOMError|MemoryError|CUDA out of memory",
        "severity": "critical",
        "fix": "reduce_batch_size_or_scale_up"
    },
    "invalid_api_key": {
        "pattern": "AuthenticationError|401|Invalid API key",
        "severity": "critical",
        "fix": "rotate_api_key"
    },
    "model_not_found": {
        "pattern": "ModelNotFoundError|404|model_not_found",
        "severity": "high",
        "fix": "check_model_name_or_fallback"
    }
}

class SystemIssueDetector:
    """Detector for system or infrastructure failures parsed from raw logs."""

    def __init__(self, patterns: Optional[Dict[str, Dict[str, str]]] = None):
        """Initializes the SystemIssueDetector.
        
        Args:
            patterns: Dictionary containing regex patterns and classification settings.
                      Defaults to SYSTEM_FAILURE_PATTERNS if None.
        """
        self.patterns = patterns or SYSTEM_FAILURE_PATTERNS

    def detect_from_log(self, log_message: str) -> Optional[Dict[str, Any]]:
        """Scans a raw log message for regex matches corresponding to known failure modes.
        
        Args:
            log_message: The raw text of the log line.
            
        Returns:
            Dictionary with classification and fix recommendation if a pattern matches, else None.
        """
        if not log_message:
            return None

        for issue_type, config in self.patterns.items():
            pattern = config["pattern"]
            if re.search(pattern, log_message, re.IGNORECASE):
                return {
                    "type": issue_type,
                    "severity": config["severity"],
                    "suggested_fix": config["fix"],
                    "log_snippet": log_message[:200]
                }
        return None
