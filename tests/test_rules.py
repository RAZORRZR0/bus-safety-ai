import hashlib
import json

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
