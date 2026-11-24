import os
import sys
import tempfile
from pathlib import Path

import pyotp
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

PASSWORD = "correct-horse-1"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def login_with_2fa(client, username, secret, password=PASSWORD):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    assert r.json()["twofa_required"] is True
    r = client.post("/auth/verify-2fa", json={"token": r.json()["access_token"],
                                              "code": pyotp.TOTP(secret).now()})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    """The first registered user, so the only one allowed to be admin."""
    r = client.post("/auth/register", json={"username": "admin", "password": PASSWORD, "role": "admin"})
    assert r.status_code == 200, r.text
    return login_with_2fa(client, "admin", r.json()["twofa_secret"])


@pytest.fixture(scope="session")
def operator(client, admin):
    r = client.post("/auth/register", json={"username": "operator1", "password": PASSWORD, "role": "operator"})
    assert r.status_code == 200, r.text
    return login_with_2fa(client, "operator1", r.json()["twofa_secret"])


@pytest.fixture(autouse=True)
def tmp_audit_chain(tmp_path, monkeypatch):
    from app.services import audit

    monkeypatch.setattr(audit, "CHAIN_FILE", tmp_path / "audit_chain.log")
