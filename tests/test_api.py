import pyotp


def _login(client, username, password):
    """Register-agnostic login that completes 2FA and returns an API token."""
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200
    body = r.json()
    assert body["twofa_required"] is True
    return body["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_flow_enforces_admin_rule_and_2fa(client):
    r = client.post("/auth/register", json={"username": "boss", "password": "pw1", "role": "admin"})
    assert r.status_code == 200
    secret = r.json()["twofa_secret"]

    # A second self-registered admin is refused; operators are fine.
    r = client.post("/auth/register", json={"username": "eve", "password": "pw2", "role": "admin"})
    assert r.status_code == 403
    r = client.post("/auth/register", json={"username": "op", "password": "pw3", "role": "operator"})
    assert r.status_code == 200

    assert client.post("/auth/login", json={"username": "boss", "password": "bad"}).status_code == 401

    pre_token = _login(client, "boss", "pw1")
    assert client.get("/alerts", headers=_auth(pre_token)).status_code == 403  # 2FA not done yet

    r = client.post("/auth/verify-2fa", json={"token": pre_token, "code": "000000"})
    assert r.status_code == 401
    r = client.post("/auth/verify-2fa", json={"token": pre_token, "code": pyotp.TOTP(secret).now()})
    assert r.status_code == 200
    token = r.json()["access_token"]
    assert client.get("/alerts", headers=_auth(token)).status_code == 200


def test_ingest_event_creates_alert_and_lowers_trust(client):
    secret = client.post(
        "/auth/register", json={"username": "op2", "password": "pw", "role": "operator"}
    ).json()["twofa_secret"]
    pre = _login(client, "op2", "pw")
    token = client.post(
        "/auth/verify-2fa", json={"token": pre, "code": pyotp.TOTP(secret).now()}
    ).json()["access_token"]

    event = {
        "bus_id": "DL1PC0001", "route_id": "R-42", "timeslot": "08:00-09:00",
        "reserved_seats_total": 4, "reserved_seats_occupied": 0,
        "special_category_present": False, "aggression_score": 0.95,
    }
    r = client.post("/ingest/event", json=event, headers=_auth(token))
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] == "critical"
    assert body["alert_id"] is not None

    scores = client.get("/trust/overview", headers=_auth(token)).json()
    bus = next(s for s in scores if s["scope_type"] == "bus" and s["scope_id"] == "DL1PC0001")
    assert bus["score"] == 75.0  # 100 - (15 violence + 10 critical)


def test_stream_rejects_arbitrary_paths(client):
    r = client.get("/stream/live", params={"source": "/etc/passwd"})
    assert r.status_code == 400
