import json
import os
import uuid
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "detection_history.json")

_history: list[dict] = []


def _load_history():
    global _history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                _history = json.load(f)
        except Exception:
            _history = []
    else:
        _history = []


_load_history()


def _save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_history, f, indent=2)
    except Exception:
        # If filesystem permissions temporarily block write, in-memory copy retains records
        pass


def add_detection_record(
    matched: bool,
    status: str,
    name: str | None,
    score: float,
    threshold: float,
    image_name: str | None,
    latency_ms: float,
    message: str,
) -> dict:
    record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "matched": 1 if matched else 0,
        "status": status,
        "name": name,
        "score": round(score, 4),
        "threshold": round(threshold, 4),
        "image_name": image_name,
        "latency_ms": round(latency_ms, 1),
        "message": message,
    }
    _history.insert(0, record)
    if len(_history) > 500:
        _history.pop()
    _save_history()
    return record


def get_detection_history(limit: int = 100) -> list[dict]:
    return _history[:limit]


def clear_detection_history() -> None:
    global _history
    _history = []
    _save_history()
