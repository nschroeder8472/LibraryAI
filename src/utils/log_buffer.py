"""In-memory log buffer for the web UI logs panel."""
import logging
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional


class LogBuffer(logging.Handler):
    """Logging handler that stores recent log entries in a circular buffer.

    Attach this handler to the root logger to capture all application logs
    and expose them via the /api/logs REST endpoint.
    """

    def __init__(self, max_entries: int = 500):
        super().__init__()
        self.buffer: deque = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(
                "%H:%M:%S"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        }
        self.buffer.append(entry)

    def get_logs(
        self,
        level: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Return recent log entries, optionally filtered by level.

        Args:
            level: Filter to a specific level (e.g. "ERROR"). None returns all.
            limit: Maximum number of entries to return.

        Returns:
            List of log entry dicts (most recent last).
        """
        logs = list(self.buffer)
        if level:
            logs = [entry for entry in logs if entry["level"] == level.upper()]
        return logs[-limit:]


# Module-level singleton so both logging_config and app.py can share it.
log_buffer = LogBuffer()
