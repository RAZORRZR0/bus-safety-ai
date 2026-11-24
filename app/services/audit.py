import hashlib
import json
import threading
from pathlib import Path
from typing import Any


EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
CHAIN_FILE = EVIDENCE_DIR / "audit_chain.log"
GENESIS = "0" * 64
# Appends read the last hash then write; without a lock two requests can fork the chain.
_lock = threading.Lock()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry_hash(prev_hash: str, record_type: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return _sha256(prev_hash + "|" + record_type + "|" + body)


def _last_hash() -> str:
    if not CHAIN_FILE.exists():
        return GENESIS
    lines = CHAIN_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return GENESIS
    return str(json.loads(lines[-1])["entry_hash"])


def append_audit(record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        prev_hash = _last_hash()
        row = {
            "record_type": record_type,
            "prev_hash": prev_hash,
            "entry_hash": _entry_hash(prev_hash, record_type, payload),
            "payload": payload,
        }
        with CHAIN_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return row


def verify_chain() -> dict[str, Any]:
    """Recompute every hash; report the first entry that was edited, removed or reordered."""
    if not CHAIN_FILE.exists():
        return {"exists": False, "entries": 0, "valid": True, "first_bad_entry": None}
    prev = GENESIS
    lines = CHAIN_FILE.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
            ok = row["prev_hash"] == prev and row["entry_hash"] == _entry_hash(
                prev, row["record_type"], row["payload"])
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            return {"exists": True, "entries": len(lines), "valid": False, "first_bad_entry": i}
        prev = row["entry_hash"]
    return {"exists": True, "entries": len(lines), "valid": True, "first_bad_entry": None}
