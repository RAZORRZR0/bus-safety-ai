import os
import sys
import tempfile
from pathlib import Path

import pytest

# Must be set before the app (and its settings) are imported.
_tmp = tempfile.mkdtemp()
os.environ.update(
    JWT_SECRET="test-secret",
    REQUIRE_2FA="true",
    DB_URL=f"sqlite:///{_tmp}/test.db",
    YOLO_MODEL="none",            # no detector download in tests; app falls back to rules
    VIOLENCE_MODEL="models/missing.pth",
)
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # app serves dashboard/ by relative path
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def tmp_audit_chain(tmp_path, monkeypatch):
    from app.services import audit

    monkeypatch.setattr(audit, "CHAIN_FILE", tmp_path / "audit_chain.log")
