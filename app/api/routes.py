import logging
import tempfile
from collections import deque
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Body, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from sqlmodel import Session, select

from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_2fa_secret,
    hash_password,
    provisioning_uri,
    verify_2fa_code,
    verify_password,
)
from app.models.domain import (
    Alert,
    EventIngestRequest,
    LoginRequest,
    RegisterRequest,
    TrustScore,
    User,
    Verify2FARequest,
    ViolationEvent,
)
from app.core.config import settings
from app.services.alerts import acknowledge_alert, create_alert
from app.services.audit import append_audit
from app.services.evidence import persist_evidence
from app.services.inference import InferenceService
from app.services.trust import apply_trust_impact
from app.storage.db import get_session

logger = logging.getLogger("transport")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

router = APIRouter()
inference_service = InferenceService()


def _extract_token(auth_header: str | None) -> str:
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return auth_header.split(" ", 1)[1].strip()


def get_current_user(
    session: Annotated[Session, Depends(get_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    token = _extract_token(authorization)
    username = decode_access_token(token)
    user = session.exec(select(User).where(User.username == username)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    # 2FA check disabled for testing
    # if not user.twofa_verified:
    #     raise HTTPException(status_code=403, detail="2FA verification required.")
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    return RedirectResponse(url="/dashboard", status_code=307)


@router.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@router.get("/model/status")
def model_status():
    return inference_service.model_status()


@router.post("/model/switch")
def model_switch(
    payload: dict,
    _: User = Depends(get_current_user),
):
    model_name = payload.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' field.")
    try:
        return inference_service.switch_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/roi/config")
def get_roi():
    roi = {
        "roi_x1": settings.roi_x1, "roi_y1": settings.roi_y1,
        "roi_x2": settings.roi_x2, "roi_y2": settings.roi_y2,
    }
    logger.debug("GET /roi/config → %s", roi)
    return roi


@router.post("/roi/config")
def set_roi(payload: dict = Body(...), _: User = Depends(get_current_user)):
    logger.info("POST /roi/config payload=%s", payload)
    try:
        for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"):
            if key in payload:
                val = float(payload[key])
                if not (0.0 <= val <= 1.0):
                    raise HTTPException(400, f"{key} must be between 0.0 and 1.0")
                setattr(settings, key, val)
                logger.debug("  set %s = %.4f", key, val)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ROI config update FAILED")
        raise HTTPException(500, f"ROI update error: {e}")
    result = get_roi()
    logger.info("ROI updated → %s", result)
    return result


# ── Unified video scan (all models) ─────────────────────────────────────────
@router.post("/scan/video")
async def scan_video(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
):
    """Upload a video clip and run ALL detection models:
    YOLO people detection, gender/seat violations, violence classification.
    """
    logger.info("POST /scan/video  filename=%s", file.filename)
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = inference_service.analyze_video_unified(tmp_path)
        logger.info("Unified scan result: %s", result)
        return result
    except Exception as e:
        logger.exception("Unified scan FAILED")
        raise HTTPException(500, str(e))
    finally:
        import os
        os.unlink(tmp_path)


@router.get("/stream/live")
def stream_live(source: str = "0"):
    """MJPEG stream with YOLO annotations. source=0 for webcam, or a file/RTSP path."""
    cam_source = int(source) if source.isdigit() else source

    def generate():
        cap = cv2.VideoCapture(cam_source)
        if not cap.isOpened():
            # Yield a single error frame
            err = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(err, f"Cannot open: {source}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, buf = cv2.imencode(".jpg", err)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            return
        
        # Buffer for violence AI (needs 16 frames sequence)
        frame_buffer = deque(maxlen=16)
        violence_score = 0.0
        frame_count = 0
        
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Maintain rolling buffer
                frame_buffer.append(frame)
                frame_count += 1
                
                # Run violence AI every 8 frames (approx 3-4 times/sec)
                # Only if we have a full sequence of 16 frames
                if len(frame_buffer) == 16 and frame_count % 8 == 0:
                    try:
                        # logical_frames = list(frame_buffer)
                        # The prediction model expects a list of frames.
                        # Using the latest 16 frames available.
                        violence_score = inference_service.classify_violence_frames(list(frame_buffer))
                    except Exception as e:
                        # Fallback or log error, keep previous score to avoid flickering
                        pass

                annotated = inference_service.analyze_frame_annotated(frame, violence_score=violence_score)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/dashboard")
def dashboard():
    return FileResponse(Path("dashboard") / "index.html")


@router.post("/auth/register")
def register(payload: RegisterRequest, session: Annotated[Session, Depends(get_session)]):
    existing = session.exec(select(User).where(User.username == payload.username)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")
    secret = generate_2fa_secret()
    user = User(
        username=payload.username,
        role=payload.role,
        password_hash=hash_password(payload.password),
        twofa_secret=secret,
        twofa_verified=False,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "twofa_secret": user.twofa_secret,
        "provisioning_uri": provisioning_uri(user.twofa_secret, user.username),
    }


@router.post("/auth/login")
def login(payload: LoginRequest, session: Annotated[Session, Depends(get_session)]):
    user = session.exec(select(User).where(User.username == payload.username)).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    user.twofa_verified = True  # Auto-verify for testing
    session.add(user)
    session.commit()
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/auth/verify-2fa")
def verify_2fa(payload: Verify2FARequest, session: Annotated[Session, Depends(get_session)]):
    username = decode_access_token(payload.token)
    user = session.exec(select(User).where(User.username == username)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    if not verify_2fa_code(user.twofa_secret, payload.code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code.")
    user.twofa_verified = True
    session.add(user)
    session.commit()
    return {"verified": True}


@router.post("/ingest/event")
def ingest_event(
    payload: EventIngestRequest,
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    result = inference_service.analyze(payload)
    event = ViolationEvent(
        bus_id=payload.bus_id,
        route_id=payload.route_id,
        timeslot=payload.timeslot,
        seat_misuse=result.seat_misuse,
        violence=result.violence,
        people_count=result.people_count,
        aggression_score=payload.aggression_score,
        severity=result.severity,
        summary=result.summary,
    )
    session.add(event)
    session.commit()
    session.refresh(event)

    alert = create_alert(session, event)
    apply_trust_impact(session, event)
    append_audit(
        "event",
        {
            "event_id": event.id,
            "bus_id": event.bus_id,
            "route_id": event.route_id,
            "severity": event.severity,
            "seat_misuse": event.seat_misuse,
            "violence": event.violence,
        },
    )

    return {
        "event_id": event.id,
        "severity": event.severity,
        "summary": event.summary,
        "people_count": event.people_count,
        "alert_id": alert.id if alert else None,
    }


@router.post("/ingest/frame")
async def ingest_frame(
    bus_id: str,
    route_id: str,
    timeslot: str,
    reserved_seats_total: int = 4,
    special_category_present: bool = False,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    npbuf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image payload.")

    frame_analysis = inference_service.analyze_frame(
        bus_id=bus_id,
        route_id=route_id,
        timeslot=timeslot,
        frame_bgr=frame,
        reserved_seats_total=max(1, reserved_seats_total),
        special_category_present=special_category_present,
    )
    result = inference_service.analyze(frame_analysis.payload)
    event = ViolationEvent(
        bus_id=frame_analysis.payload.bus_id,
        route_id=frame_analysis.payload.route_id,
        timeslot=frame_analysis.payload.timeslot,
        seat_misuse=result.seat_misuse,
        violence=result.violence,
        people_count=frame_analysis.people_count,
        aggression_score=frame_analysis.payload.aggression_score,
        severity=result.severity,
        summary=result.summary,
    )
    session.add(event)
    session.commit()
    session.refresh(event)

    alert = create_alert(session, event)
    apply_trust_impact(session, event)
    append_audit(
        "event",
        {
            "event_id": event.id,
            "bus_id": event.bus_id,
            "route_id": event.route_id,
            "severity": event.severity,
            "seat_misuse": event.seat_misuse,
            "violence": event.violence,
            "people_count": event.people_count,
        },
    )

    return {
        "event_id": event.id,
        "severity": event.severity,
        "summary": event.summary,
        "people_count": event.people_count,
        "aggression_score": event.aggression_score,
        "alert_id": alert.id if alert else None,
    }


@router.post("/ingest/frame-preview")
async def frame_preview(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
):
    """Return an annotated JPEG with YOLO bounding boxes drawn on the frame."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    npbuf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image.")
    annotated = inference_service.analyze_frame_annotated(frame)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/ingest/evidence")
async def ingest_evidence(
    bus_id: str,
    event_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    event = session.get(ViolationEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    path, digest = persist_evidence(bus_id, raw)
    event.evidence_hash = digest
    session.add(event)
    session.commit()
    append_audit(
        "evidence",
        {"event_id": event.id, "bus_id": bus_id, "path": path, "sha256": digest},
    )
    return {"event_id": event.id, "path": path, "sha256": digest}


@router.get("/alerts")
def list_alerts(
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    rows = session.exec(select(Alert).order_by(Alert.created_at.desc())).all()
    return rows


@router.post("/alerts/{alert_id}/ack")
def ack_alert(
    alert_id: int,
    session: Annotated[Session, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    alert = session.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    if alert.status == "acknowledged":
        return alert
    alert = acknowledge_alert(session, alert, user.username)
    return alert


@router.get("/trust/overview")
def trust_overview(
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    rows = session.exec(select(TrustScore).order_by(TrustScore.score.asc())).all()
    return rows


@router.get("/events")
def list_events(
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    rows = session.exec(select(ViolationEvent).order_by(ViolationEvent.created_at.desc())).all()
    return rows


@router.get("/admin/audit-chain")
def audit_chain_status(
    _: Annotated[User, Depends(require_admin)],
):
    from app.services.audit import CHAIN_FILE

    if not CHAIN_FILE.exists():
        return {"exists": False, "entries": 0}
    lines = CHAIN_FILE.read_text(encoding="utf-8").strip().splitlines()
    return {"exists": True, "entries": len(lines)}
