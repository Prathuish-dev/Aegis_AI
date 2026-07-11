import time
from typing import Any, Dict
from loguru import logger
from src.healing.audit_logger import AuditLogger
from src.monitoring.metric_tracker import MetricTracker

class Verifier:
    """
    Verifier performs post-fix checks to verify if a corrective action (fix)
    successfully resolved the detected anomaly.
    """
    def __init__(self, audit_logger: AuditLogger = None):
        """
        Initialize the Verifier.
        
        Args:
            audit_logger: AuditLogger instance to record verification events. If None, initializes a default one.
        """
        self.audit_logger = audit_logger if audit_logger is not None else AuditLogger()
        logger.info("Verifier initialized.")

    def check_health(self, system_id: str, metric_type: Any, metric_tracker: MetricTracker, baseline_value: float = None) -> bool:
        """
        Evaluates system health by checking if there are any active threshold breaches.
        
        Args:
            system_id: Unique identifier for the system.
            metric_type: MetricType under review.
            metric_tracker: MetricTracker instance to check.
            baseline_value: Baseline metric value (required for delta checks).
            
        Returns:
            bool: True if the system is healthy (no active breaches), False otherwise.
        """
        events = metric_tracker.check_thresholds(system_id, metric_type, baseline_value)
        return len(events) == 0

    def verify(
        self,
        system_id: str,
        metric_type: Any,
        metric_tracker: MetricTracker,
        baseline_value: float = None,
        check_interval_seconds: float = 1.0,
        max_wait_seconds: float = 600.0
    ) -> bool:
        """
        Polls the system metrics for a specified duration to confirm recovery.
        
        Args:
            system_id: Unique identifier for the system.
            metric_type: MetricType under review.
            metric_tracker: MetricTracker instance to monitor.
            baseline_value: Baseline metric value (required for delta checks).
            check_interval_seconds: Delay between health checks.
            max_wait_seconds: Maximum time to wait for recovery.
            
        Returns:
            bool: True if the system recovered, False if it remained unresolved.
        """
        metric_key = metric_type.value if hasattr(metric_type, "value") else str(metric_type)
        logger.info(f"Starting post-fix verification for {system_id} ({metric_key}) for up to {max_wait_seconds} seconds.")
        
        start_time = time.time()
        elapsed = 0.0
        
        while elapsed < max_wait_seconds:
            is_healthy = self.check_health(system_id, metric_type, metric_tracker, baseline_value)
            
            if is_healthy:
                logger.info(f"System {system_id} successfully healed after {elapsed:.1f}s.")
                self.audit_logger.log_action(
                    agent_id="Verifier",
                    action="post_fix_verify",
                    outcome="healed",
                    details={
                        "system_id": system_id,
                        "metric_type": metric_key,
                        "elapsed_seconds": elapsed,
                        "status": "resolved"
                    }
                )
                return True
                
            time.sleep(check_interval_seconds)
            elapsed = time.time() - start_time
            
        logger.warning(f"Post-fix verification timed out for {system_id} ({metric_key}). System remains unresolved.")
        self.audit_logger.log_action(
            agent_id="Verifier",
            action="post_fix_verify",
            outcome="unresolved",
            details={
                "system_id": system_id,
                "metric_type": metric_key,
                "elapsed_seconds": elapsed,
                "status": "failed_to_recover"
            }
        )
        return False
