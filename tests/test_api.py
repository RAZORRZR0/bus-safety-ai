from tests.conftest import PASSWORD

EVENT = {
    "bus_id": "DL1PC0001", "route_id": "R-42", "timeslot": "08:00-09:00",
    "reserved_seats_total": 4, "reserved_seats_occupied": 0,
    "special_category_present": False, "aggression_score": 0.95,
}


def test_health_and_security_headers(client):
    r = client.get("/health")
    assert r.json() == {"status": "ok", "demo": False}
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_only_first_user_can_be_admin(client, admin):
    r = client.post("/auth/register", json={"username": "eve", "password": PASSWORD, "role": "admin"})
    assert r.status_code == 403
    r = client.post("/auth/register", json={"username": "eve", "password": PASSWORD, "role": "root"})
    assert r.status_code == 400


def test_register_validates_input(client):
    assert client.post("/auth/register", json={"username": "shortpw", "password": "123"}).status_code == 422
    assert client.post("/auth/register", json={"username": "<b>x</b>", "password": PASSWORD}).status_code == 422


def test_pre_2fa_token_cannot_use_api(client, admin):
    r = client.post("/auth/login", json={"username": "admin", "password": PASSWORD})
    pre = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get("/alerts", headers=pre).status_code == 403
    r = client.post("/auth/verify-2fa", json={"token": r.json()["access_token"], "code": "000000"})
    assert r.status_code == 401


def test_login_is_rate_limited(client):
    client.post("/auth/register", json={"username": "victim", "password": PASSWORD})
    for _ in range(5):
        assert client.post("/auth/login", json={"username": "victim", "password": "wrong-pass"}).status_code == 401
    # Even the right password is refused while locked out.
    assert client.post("/auth/login", json={"username": "victim", "password": PASSWORD}).status_code == 429


def test_endpoints_need_login(client):
    for path in ("/alerts", "/events", "/trust/overview", "/model/status", "/roi/config"):
        assert client.get(path).status_code == 401, path


# ── Roles ─────────────────────────────────────────────────────────────────────

def test_operator_cannot_change_config_or_delete(client, admin, operator):
    bus = {"bus_id": "DL1PC9999", "route_id": "R-1", "capacity": 50, "model": "Tata", "year": 2022}
    assert client.post("/fleet/buses", json=bus, headers=operator).status_code == 201
    assert client.delete("/fleet/buses/DL1PC9999", headers=operator).status_code == 403
    assert client.post("/roi/config", json={"roi_x1": 0.1}, headers=operator).status_code == 403
    assert client.post("/model/switch", json={"model": "yolo26n.pt"}, headers=operator).status_code == 403
    assert client.get("/admin/audit-chain", headers=operator).status_code == 403
    assert client.delete("/fleet/buses/DL1PC9999", headers=admin).status_code == 204


def test_roi_must_be_a_valid_box(client, admin):
    r = client.post("/roi/config", json={"roi_x1": 0.8, "roi_x2": 0.2}, headers=admin)
    assert r.status_code == 400
    r = client.post("/roi/config", json={"roi_x1": "abc"}, headers=admin)
    assert r.status_code == 400
    r = client.post("/roi/config", json={"roi_x1": 0.1, "roi_y1": 0.2, "roi_x2": 0.6, "roi_y2": 0.9}, headers=admin)
    assert r.status_code == 200 and r.json()["roi_x2"] == 0.6


# ── Input validation ──────────────────────────────────────────────────────────

def test_ids_with_markup_or_paths_are_rejected(client, operator):
    for bad in ("<img src=x onerror=alert(1)>", "../../etc", "a'b"):
        r = client.post("/ingest/event", json={**EVENT, "bus_id": bad}, headers=operator)
        assert r.status_code == 422, bad
    bus = {"bus_id": "x');alert(1);//", "route_id": "R-1", "capacity": 50, "model": "Tata", "year": 2022}
    assert client.post("/fleet/buses", json=bus, headers=operator).status_code == 422
    assert client.post("/ingest/event", json={**EVENT, "aggression_score": 5}, headers=operator).status_code == 422


def test_feedback_reference_must_be_event_or_scan(client, operator):
    for ref in ("audit_chain.log", "", "../x"):
        r = client.post("/feedback", json={"event_reference_id": ref, "feedback_type": "x", "is_correct": True},
                        headers=operator)
        assert r.status_code == 422, ref


def test_oversized_upload_is_rejected(client, operator):
    big = b"\0" * (10 * 1024 * 1024 + 1)
    r = client.post("/ingest/frame", params={"bus_id": "B1", "route_id": "R1", "timeslot": "T1"},
                    files={"file": ("f.jpg", big, "image/jpeg")}, headers=operator)
    assert r.status_code == 413


def test_scan_rejects_non_video_files(client, operator):
    r = client.post("/scan/video", files={"file": ("x.html", b"<script>", "text/html")}, headers=operator)
    assert r.status_code == 400


# ── Core flow ─────────────────────────────────────────────────────────────────

def test_ingest_event_creates_alert_and_lowers_trust(client, operator):
    r = client.post("/ingest/event", json=EVENT, headers=operator)
    assert r.status_code == 200
    assert r.json()["severity"] == "critical"
    assert r.json()["alert_id"] is not None

    scores = client.get("/trust/overview", headers=operator).json()
    bus = next(s for s in scores if s["scope_type"] == "bus" and s["scope_id"] == "DL1PC0001")
    assert bus["score"] == 75.0  # 100 - (15 violence + 10 critical)


def test_audit_chain_endpoint_reports_tampering(client, admin):
    from app.services import audit

    client.post("/ingest/event", json=EVENT, headers=admin)
    client.post("/ingest/event", json=EVENT, headers=admin)
    assert client.get("/admin/audit-chain", headers=admin).json()["valid"] is True

    lines = audit.CHAIN_FILE.read_text().splitlines()
    audit.CHAIN_FILE.write_text("\n".join([lines[0].replace("critical", "low"), *lines[1:]]) + "\n")
    report = client.get("/admin/audit-chain", headers=admin).json()
    assert report["valid"] is False and report["first_bad_entry"] == 0


# ── Live stream ───────────────────────────────────────────────────────────────

def test_stream_needs_a_stream_token(client, operator):
    api_token = operator["Authorization"].split()[1]
    assert client.get("/stream/live").status_code == 422
    assert client.get("/stream/live", params={"token": api_token}).status_code == 401

    stream_token = client.post("/stream/token", headers=operator).json()["token"]
    # A stream token must not work as an API token.
    assert client.get("/alerts", headers={"Authorization": f"Bearer {stream_token}"}).status_code == 401
    # And it still only opens cameras, rtsp:// or uploaded files.
    r = client.get("/stream/live", params={"token": stream_token, "source": "/etc/passwd"})
    assert r.status_code == 400
    assert client.post("/stream/stop").status_code == 401
