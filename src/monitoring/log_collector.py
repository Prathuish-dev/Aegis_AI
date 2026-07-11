import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime

class LogCollector:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("LOG_DB_PATH", "data/logs/events.db")
        # Ensure parent directories exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the events table if it does not exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                system_id TEXT,
                metric_type TEXT,
                log_entry TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def ingest_log(self, log_entry: dict) -> None:
        """Accept a structured log and persist it."""
        timestamp = log_entry["timestamp"]
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        # Clone log_entry and ensure timestamp is stringified for JSON serialization
        serialized_log = dict(log_entry)
        serialized_log["timestamp"] = timestamp
        
        self.conn.execute(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)",
            (
                log_entry["event_id"],
                log_entry["system_id"],
                log_entry["metric_type"],
                json.dumps(serialized_log),
                timestamp
            )
        )
        self.conn.commit()

    def get_logs(self, system_id: str = None, metric_type: str = None, limit: int = 1000) -> list[dict]:
        """Retrieve logs from SQLite, optionally filtered by system_id and metric_type."""
        query = "SELECT log_entry FROM events"
        params = []
        conditions = []
        if system_id:
            conditions.append("system_id = ?")
            params.append(system_id)
        if metric_type:
            conditions.append("metric_type = ?")
            params.append(metric_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [json.loads(row[0]) for row in cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
