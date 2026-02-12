from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.domain import TrustScore, ViolationEvent


def penalty_from_event(event: ViolationEvent) -> float:
    penalty = 0.0
    if event.seat_misuse:
        penalty += 3.0
    if event.violence:
        penalty += 15.0
    if event.severity == "critical":
        penalty += 10.0
    return penalty


def upsert_score(session: Session, scope_type: str, scope_id: str, penalty: float) -> TrustScore:
    stmt = select(TrustScore).where(TrustScore.scope_type == scope_type, TrustScore.scope_id == scope_id)
    existing = session.exec(stmt).first()
    if existing is None:
        existing = TrustScore(scope_type=scope_type, scope_id=scope_id, score=100.0)
    existing.score = max(0.0, min(100.0, existing.score - penalty))
    existing.updated_at = datetime.now(timezone.utc)
    session.add(existing)
    session.commit()
    session.refresh(existing)
    return existing


def apply_trust_impact(session: Session, event: ViolationEvent) -> None:
    penalty = penalty_from_event(event)
    if penalty <= 0:
        return
    upsert_score(session, "bus", event.bus_id, penalty)
    upsert_score(session, "route", event.route_id, penalty)
    upsert_score(session, "timeslot", event.timeslot, penalty)

