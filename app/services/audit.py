import hashlib
import json
from pathlib import Path
from typing import Any


EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
CHAIN_FILE = EVIDENCE_DIR / "audit_chain.log"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_hash() -> str:
    if not CHAIN_FILE.exists():
        return "0" * 64
    lines = CHAIN_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return "0" * 64
    try:
        obj = json.loads(lines[-1])
        return str(obj["entry_hash"])
    except Exception:
        return "0" * 64


def append_audit(record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    prev_hash = _last_hash()
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    entry_hash = _sha256(prev_hash + "|" + record_type + "|" + body)
    row = {
        "record_type": record_type,
        "prev_hash": prev_hash,
        "entry_hash": entry_hash,
        "payload": payload,
    }
    with CHAIN_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return row

