import hashlib
import json
from pathlib import Path

from app.models.domain import EventIngestRequest, ViolationEvent
from app.services import audit
from app.services.trust import penalty_from_event


def _event(**kw):
    base = dict(bus_id="B", route_id="R", timeslot="T", severity="low", summary="")
    return ViolationEvent(**{**base, **kw})


def test_trust_penalties():
    assert penalty_from_event(_event()) == 0
    assert penalty_from_event(_event(seat_misuse=True, severity="medium")) == 3
    assert penalty_from_event(_event(violence=True, severity="critical")) == 25


def test_severity_rules():
    from app.api.routes import inference_service

    def sev(aggr, occupied=0, special=False):
        return inference_service.analyze(EventIngestRequest(
            bus_id="B", route_id="R", timeslot="T", reserved_seats_total=4,
            reserved_seats_occupied=occupied, special_category_present=special,
            aggression_score=aggr,
        )).severity

    assert sev(0.1) == "low"
    assert sev(0.1, occupied=1) == "medium"
    assert sev(0.1, occupied=1, special=True) == "low"
    assert sev(0.6, occupied=1) == "high"
    assert sev(0.8) == "high"
    assert sev(0.95) == "critical"


def test_audit_chain_links_and_detects_tampering():
    a = audit.append_audit("event", {"id": 1})
    b = audit.append_audit("event", {"id": 2})
    assert a["prev_hash"] == "0" * 64
    assert b["prev_hash"] == a["entry_hash"]

    def recompute(row):
        body = json.dumps(row["payload"], separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(f'{row["prev_hash"]}|{row["record_type"]}|{body}'.encode()).hexdigest()

    assert recompute(b) == b["entry_hash"]
    b["payload"]["id"] = 3
    assert recompute(b) != b["entry_hash"]


def test_evidence_filename_cannot_escape_the_folder(tmp_path, monkeypatch):
    from app.services import evidence

    monkeypatch.setattr(evidence, "EVIDENCE_DIR", tmp_path)
    path, _ = evidence.persist_evidence("../../evil", b"data")
    assert Path(path).parent == tmp_path


def test_demo_seed_takes_admin_slot_and_is_idempotent(tmp_path):
    from sqlmodel import Session, SQLModel, create_engine, select

    from app.core.security import verify_password
    from app.models.domain import Bus, User
    from app.storage.demo import DEMO_PASSWORD, seed_demo

    engine = create_engine(f"sqlite:///{tmp_path}/demo.db")
    SQLModel.metadata.create_all(engine)
    seed_demo(engine)
    seed_demo(engine)
    with Session(engine) as s:
        users = {u.username: u for u in s.exec(select(User)).all()}
        assert users["admin"].role == "admin" and users["demo"].role == "operator"
        assert verify_password(DEMO_PASSWORD, users["demo"].password_hash)
        assert len(s.exec(select(Bus)).all()) == 3
