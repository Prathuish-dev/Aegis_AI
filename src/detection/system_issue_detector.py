import re
from typing import Dict, Optional

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
    """Detector for identifying infrastructure, API, and configuration issues from log snippets."""

    def detect_from_log(self, log_message: str) -> Optional[Dict]:
        """Scans a log message against predefined regex patterns to detect issues.
        
        Args:
            log_message: Raw log line or message.
            
        Returns:
            A dictionary containing issue details (type, severity, suggested_fix, log_snippet)
            if a match is found, otherwise None.
        """
        if not log_message:
            return None

        for issue_type, config in SYSTEM_FAILURE_PATTERNS.items():
            # Use IGNORECASE to make pattern matching case-insensitive and robust
            if re.search(config["pattern"], log_message, re.IGNORECASE):
                return {
                    "type": issue_type,
                    "severity": config["severity"],
                    "suggested_fix": config["fix"],
                    "log_snippet": log_message[:200]
                }
        return None
