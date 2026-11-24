import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def persist_evidence(bus_id: str, raw: bytes) -> tuple[str, str]:
    digest = hash_bytes(raw)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_bus = re.sub(r"[^A-Za-z0-9_-]", "_", bus_id)[:64]  # never let an ID become a path
    filename = f"{ts}_{safe_bus}_{digest[:12]}.bin"
    path = EVIDENCE_DIR / filename
    path.write_bytes(raw)
    return str(path), digest

