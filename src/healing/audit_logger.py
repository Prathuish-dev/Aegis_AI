import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_path: str):
    """Context manager for SQLite connection handling."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

class AuditLogger:
    def __init__(self, config_path: str = "config/settings.yaml", db_path: str = None):
        if db_path is None:
            db_path = self._load_db_path(config_path)
            
        self.db_path = db_path
        # Ensure parent directories exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _load_db_path(self, config_path: str) -> str:
        """Load database path from config/settings.yaml or fallback to environment variables/defaults."""
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
                    db_path = config.get("database", {}).get("log_db_path")
                    if db_path:
                        return db_path
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
        
        return os.getenv("LOG_DB_PATH", "data/logs/events.db")

    def _init_db(self) -> None:
        """Create the audit_log table if it does not exist."""
        with get_db_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT
                )
            """)
            conn.commit()

    def log_action(self, agent_id: str, action: str, outcome: str, details: dict = None) -> None:
        """Record an autonomous action in the audit log table."""
        timestamp = datetime.utcnow().isoformat()
        details_json = json.dumps(details) if details is not None else None
        
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (timestamp, agent_id, action, outcome, details) VALUES (?, ?, ?, ?, ?)",
                (timestamp, agent_id, action, outcome, details_json)
            )
            conn.commit()

    def get_audit_history(self, limit: int = 100) -> list[dict]:
        """Retrieve recent audit logs, ordered by newest first."""
        with get_db_connection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, agent_id, action, outcome, details FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                details_dict = None
                if row["details"]:
                    try:
                        details_dict = json.loads(row["details"])
                    except Exception:
                        details_dict = row["details"]
                
                history.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "agent_id": row["agent_id"],
                    "action": row["action"],
                    "outcome": row["outcome"],
                    "details": details_dict
                })
            return history
