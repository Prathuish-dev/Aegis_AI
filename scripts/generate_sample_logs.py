import sys
import os
import json
import uuid
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.models import MetricType, FailureCategory, AnomalyEvent
from src.monitoring.log_collector import LogCollector
from src.monitoring.metric_tracker import MetricTracker

def generate_sample_logs(db_path: str = None, reset: bool = False):
    """
    Populate the SQLite database with mock logs illustrating each of the 5 failure categories.
    """
    if db_path is None:
        db_path = os.getenv("LOG_DB_PATH", "data/logs/events.db")
    
    db_file = Path(db_path)
    if reset and db_file.exists():
        print(f"Resetting database: deleting existing file at {db_file}")
        db_file.unlink()
        
    print(f"Initializing LogCollector with database at {db_file}...")
    log_collector = LogCollector(db_path=str(db_file))
    
    # 1. API Issues (llm-gateway-service)
    # Failure patterns checked: RateLimitError, AuthenticationError, ModelNotFoundError
    print("Generating API Issue logs...")
    system_api = "llm-gateway-service"
    base_time = datetime.utcnow() - timedelta(hours=6)
    
    # Normal metrics first
    for i in range(12):
        t = base_time + timedelta(minutes=i * 15)
        log_collector.ingest_log({
            "event_id": f"raw_api_{i}",
            "system_id": system_api,
            "metric_type": MetricType.ERROR_RATE.value,
            "value": 0.005,
            "timestamp": t.isoformat()
        })
    
    # API errors and error rate spike
    error_time = base_time + timedelta(hours=3)
    # Ingest system logs that will be parsed by SystemIssueDetector
    log_collector.ingest_log({
        "event_id": f"sys_api_err_{uuid.uuid4().hex[:8]}",
        "system_id": system_api,
        "metric_type": "system_log",
        "timestamp": (error_time + timedelta(minutes=5)).isoformat(),
        "message": "ERROR: RateLimitError: Rate limit exceeded for model gpt-4. Please retry in 15 seconds. (HTTP 429)",
        "level": "ERROR"
    })
    log_collector.ingest_log({
        "event_id": f"sys_api_err_{uuid.uuid4().hex[:8]}",
        "system_id": system_api,
        "metric_type": "system_log",
        "timestamp": (error_time + timedelta(minutes=10)).isoformat(),
        "message": "ERROR: AuthenticationError: Invalid API key provided. Please check credentials. (HTTP 401)",
        "level": "ERROR"
    })
    log_collector.ingest_log({
        "event_id": f"sys_api_err_{uuid.uuid4().hex[:8]}",
        "system_id": system_api,
        "metric_type": "system_log",
        "timestamp": (error_time + timedelta(minutes=15)).isoformat(),
        "message": "ERROR: ModelNotFoundError: Model 'gpt-4-custom' not found or you do not have access to it. (HTTP 404)",
        "level": "ERROR"
    })
    # Error rate spike metrics
    for i in range(5):
        t = error_time + timedelta(minutes=i * 5)
        log_collector.ingest_log({
            "event_id": f"raw_api_spike_{i}",
            "system_id": system_api,
            "metric_type": MetricType.ERROR_RATE.value,
            "value": 0.15,
            "timestamp": t.isoformat()
        })
    
    # Anomaly Event for API issues
    api_anomaly = AnomalyEvent(
        event_id=f"evt_api_{uuid.uuid4().hex[:8]}",
        system_id=system_api,
        metric_type=MetricType.ERROR_RATE,
        current_value=0.15,
        baseline_value=0.005,
        timestamp=error_time + timedelta(minutes=15),
        raw_context={
            "failure_category": FailureCategory.SYSTEM_ISSUE.value,
            "failure_subcategory": "api_issue",
            "message": "High error rate detected with multiple LLM API authentication and rate limit failures.",
            "error_types": ["RateLimitError", "AuthenticationError", "ModelNotFoundError"]
        },
        severity="critical"
    )
    log_collector.ingest_log(api_anomaly.model_dump())


    # 2. Data Drift (customer-churn-predictor)
    print("Generating Data Drift logs...")
    system_drift = "customer-churn-predictor"
    drift_time = datetime.utcnow() - timedelta(hours=4)
    
    # System logs matching data drift statistical reports
    log_collector.ingest_log({
        "event_id": f"sys_drift_{uuid.uuid4().hex[:8]}",
        "system_id": system_drift,
        "metric_type": "system_log",
        "timestamp": drift_time.isoformat(),
        "message": "DataIssue: Covariate drift detected in feature 'user_age' (KS statistic: 0.24, p-value: 0.003)",
        "level": "WARNING"
    })
    log_collector.ingest_log({
        "event_id": f"sys_drift_{uuid.uuid4().hex[:8]}",
        "system_id": system_drift,
        "metric_type": "system_log",
        "timestamp": (drift_time + timedelta(minutes=5)).isoformat(),
        "message": "DataIssue: Population Stability Index (PSI) breach for feature 'monthly_spend' (PSI: 0.31, severity: severe)",
        "level": "ERROR"
    })
    log_collector.ingest_log({
        "event_id": f"sys_drift_{uuid.uuid4().hex[:8]}",
        "system_id": system_drift,
        "metric_type": "system_log",
        "timestamp": (drift_time + timedelta(minutes=10)).isoformat(),
        "message": "DataIssue: Schema drift detected. Expected column 'signup_country' but found 'country_code' with type TEXT.",
        "level": "ERROR"
    })
    
    # Anomaly Event for Data Drift
    # Since MetricType does not contain data_drift, we map it to ACCURACY drop (often a result of drift) or use a custom raw_context
    drift_anomaly = AnomalyEvent(
        event_id=f"evt_drift_{uuid.uuid4().hex[:8]}",
        system_id=system_drift,
        metric_type=MetricType.ACCURACY,
        current_value=0.72,
        baseline_value=0.85,
        timestamp=drift_time + timedelta(minutes=10),
        raw_context={
            "failure_category": FailureCategory.DATA_ISSUE.value,
            "failure_subcategory": "data_drift",
            "message": "Feature covariate drift and PSI breach observed in input stream.",
            "drift_features": {
                "user_age": {"test": "KS", "p_value": 0.003, "stat": 0.24},
                "monthly_spend": {"test": "PSI", "psi": 0.31}
            },
            "schema_issues": {
                "missing": ["signup_country"],
                "added": ["country_code"]
            }
        },
        severity="high"
    )
    log_collector.ingest_log(drift_anomaly.model_dump())


    # 3. OOM (image-segmentation-pipeline)
    # Failure patterns checked: OOMError, MemoryError, CUDA out of memory
    print("Generating OOM logs...")
    system_oom = "image-segmentation-pipeline"
    oom_time = datetime.utcnow() - timedelta(hours=3)
    
    # CUDA OOM log entry
    log_collector.ingest_log({
        "event_id": f"sys_oom_{uuid.uuid4().hex[:8]}",
        "system_id": system_oom,
        "metric_type": "system_log",
        "timestamp": oom_time.isoformat(),
        "message": "CRITICAL: CUDA out of memory. Tried to allocate 8.00 GiB (GPU 0; 16.00 GiB total capacity; 12.50 GiB already allocated)",
        "level": "CRITICAL"
    })
    
    # System process OOM log entry
    log_collector.ingest_log({
        "event_id": f"sys_oom_{uuid.uuid4().hex[:8]}",
        "system_id": system_oom,
        "metric_type": "system_log",
        "timestamp": (oom_time + timedelta(seconds=15)).isoformat(),
        "message": "OOMError: Process terminated by OOM Killer. Memory usage exceeded system limits (32GB limit, attempted 34.2GB)",
        "level": "CRITICAL"
    })
    
    # Anomaly Event for OOM
    oom_anomaly = AnomalyEvent(
        event_id=f"evt_oom_{uuid.uuid4().hex[:8]}",
        system_id=system_oom,
        metric_type=MetricType.ERROR_RATE,
        current_value=1.0,
        baseline_value=0.0,
        timestamp=oom_time + timedelta(seconds=15),
        raw_context={
            "failure_category": FailureCategory.SYSTEM_ISSUE.value,
            "failure_subcategory": "out_of_memory",
            "message": "GPU / System memory exhaustion causing process termination.",
            "oom_details": "CUDA out of memory on GPU 0. OS terminated process."
        },
        severity="critical"
    )
    log_collector.ingest_log(oom_anomaly.model_dump())


    # 4. Timeouts (payment-processor)
    # Failure patterns checked: HTTPTimeoutError, TimeoutError, ReadTimeout
    print("Generating Timeout logs...")
    system_timeout = "payment-processor"
    timeout_time = datetime.utcnow() - timedelta(hours=2)
    
    # Normal latency metrics
    for i in range(10):
        t = timeout_time - timedelta(minutes=(10-i) * 10)
        log_collector.ingest_log({
            "event_id": f"raw_lat_{i}",
            "system_id": system_timeout,
            "metric_type": MetricType.LATENCY.value,
            "value": 150.0,
            "timestamp": t.isoformat()
        })
        
    # Latency spike and timeout logs
    log_collector.ingest_log({
        "event_id": f"sys_timeout_{uuid.uuid4().hex[:8]}",
        "system_id": system_timeout,
        "metric_type": "system_log",
        "timestamp": timeout_time.isoformat(),
        "message": "ERROR: HTTPTimeoutError: Request to payment provider timed out after 30000ms",
        "level": "ERROR"
    })
    log_collector.ingest_log({
        "event_id": f"sys_timeout_{uuid.uuid4().hex[:8]}",
        "system_id": system_timeout,
        "metric_type": "system_log",
        "timestamp": (timeout_time + timedelta(minutes=5)).isoformat(),
        "message": "ERROR: ReadTimeout: HTTPSConnectionPool(host='api.stripe.com', port=443): Read timed out (timeout=30.0)",
        "level": "ERROR"
    })
    
    # Latency metric spike
    for i in range(3):
        t = timeout_time + timedelta(minutes=i * 5)
        log_collector.ingest_log({
            "event_id": f"raw_lat_spike_{i}",
            "system_id": system_timeout,
            "metric_type": MetricType.LATENCY.value,
            "value": 4500.0,
            "timestamp": t.isoformat()
        })
        
    # Anomaly Event for Latency/Timeout
    timeout_anomaly = AnomalyEvent(
        event_id=f"evt_timeout_{uuid.uuid4().hex[:8]}",
        system_id=system_timeout,
        metric_type=MetricType.LATENCY,
        current_value=4500.0,
        baseline_value=150.0,
        timestamp=timeout_time + timedelta(minutes=10),
        raw_context={
            "failure_category": FailureCategory.SYSTEM_ISSUE.value,
            "failure_subcategory": "timeout",
            "message": "Payment API connection timed out. Latency exceeded the 2000ms threshold.",
            "timeout_exceptions": ["HTTPTimeoutError", "ReadTimeout"]
        },
        severity="high"
    )
    log_collector.ingest_log(timeout_anomaly.model_dump())


    # 5. Accuracy Drops (churn-classifier)
    print("Generating Accuracy Drop logs...")
    system_accuracy = "churn-classifier"
    acc_time = datetime.utcnow() - timedelta(hours=1)
    
    # Normal metrics (Baseline)
    for i in range(15):
        t = acc_time - timedelta(minutes=(15-i) * 10)
        log_collector.ingest_log({
            "event_id": f"raw_acc_{i}",
            "system_id": system_accuracy,
            "metric_type": MetricType.ACCURACY.value,
            "value": 0.85 + (i % 3 - 1) * 0.01, # ~0.85
            "timestamp": t.isoformat()
        })
        
    # Accuracy drop metrics
    for i in range(5):
        t = acc_time + timedelta(minutes=i * 10)
        log_collector.ingest_log({
            "event_id": f"raw_acc_drop_{i}",
            "system_id": system_accuracy,
            "metric_type": MetricType.ACCURACY.value,
            "value": 0.58 + (i % 2) * 0.01, # ~0.58 (approx 31% drop)
            "timestamp": t.isoformat()
        })
        
    # Anomaly Event for Accuracy Drop
    accuracy_anomaly = AnomalyEvent(
        event_id=f"evt_acc_{uuid.uuid4().hex[:8]}",
        system_id=system_accuracy,
        metric_type=MetricType.ACCURACY,
        current_value=0.58,
        baseline_value=0.85,
        timestamp=acc_time + timedelta(minutes=40),
        raw_context={
            "failure_category": FailureCategory.MODEL_ISSUE.value,
            "failure_subcategory": "accuracy_drop",
            "message": "Model classification accuracy dropped significantly below baseline.",
            "delta_pct": -0.31
        },
        severity="high"
    )
    log_collector.ingest_log(accuracy_anomaly.model_dump())

    log_collector.close()
    print("Successfully populated SQLite database with mock logs for all 5 failure categories!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate SQLite database with mock failure logs.")
    parser.add_argument("--db-path", type=str, default=None, help="Path to the SQLite database file.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database file before writing.")
    args = parser.parse_args()
    
    generate_sample_logs(db_path=args.db_path, reset=args.reset)
