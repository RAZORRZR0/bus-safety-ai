from datetime import datetime, timezone

from sqlmodel import Session

from app.models.domain import Alert, ViolationEvent


def create_alert(session: Session, event: ViolationEvent) -> Alert | None:
    if event.severity not in {"medium", "high", "critical"}:
        return None
    alert = Alert(
        violation_event_id=event.id or 0,
        bus_id=event.bus_id,
        route_id=event.route_id,
        severity=event.severity,
        status="open",
    )
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def acknowledge_alert(session: Session, alert: Alert, username: str) -> Alert:
    alert.status = "acknowledged"
    alert.acknowledged_by = username
    alert.acknowledged_at = datetime.now(timezone.utc)
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert

