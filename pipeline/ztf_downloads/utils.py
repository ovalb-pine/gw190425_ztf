import json
from pathlib import Path

def _load_json(path, default):
    """Load JSON from disk, returning default if file doesn't exist."""
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, payload):
    """Atomically save JSON to disk using temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)