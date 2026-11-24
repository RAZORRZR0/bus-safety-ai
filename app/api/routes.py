import logging
import tempfile
import threading
import time
from collections import defaultdict, deque
from functools import partial

import anyio
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Body, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from sqlmodel import Session, select

from app.core.security import (
    create_access_token,
    create_stream_token,
    decode_access_token,
    generate_2fa_secret,
    hash_password,
    provisioning_uri,
    verify_2fa_code,
    verify_password,
)
from app.models.domain import (
    ID_PATTERN,
    Alert,
    Bus,
    BusCreate,
    EventIngestRequest,
    FeedbackSubmitRequest,
    LoginRequest,
    RegisterRequest,
    Route,
    RouteCreate,
    TrustScore,
    User,
    Verify2FARequest,
    ViolationEvent,
)
from app.core.config import settings
from app.services.alerts import acknowledge_alert, create_alert
from app.services.audit import append_audit, verify_chain
from app.services.evidence import persist_evidence
from app.services.inference import InferenceService
from app.services.trust import apply_trust_impact
from app.storage.db import get_session

from app.services.feedback import FeedbackService
from app.services.retrain import RetrainService

logger = logging.getLogger("transport")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

router = APIRouter()
inference_service = InferenceService()
retrain_service = RetrainService(inference_service)
feedback_service = FeedbackService(retrain_service=retrain_service)

_stream_cancel = threading.Event()
# Files the dashboard uploaded for streaming; /stream/live only opens these, cameras and rtsp:// URLs.
_uploaded_streams: set[str] = set()

MB = 1024 * 1024
MAX_IMAGE_BYTES = 10 * MB
MAX_VIDEO_BYTES = 200 * MB
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Login throttling: per-username failures inside a sliding window (in-process; enough for one server).
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 300
_login_failures: dict[str, deque] = defaultdict(deque)
_login_lock = threading.Lock()


# Model calls take seconds of CPU. Running them inside an async endpoint froze every other
# request, so they go to a worker thread. One at a time: the YOLO tracker is not thread-safe.
_inference_limiter = anyio.CapacityLimiter(1)


async def _infer(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs), limiter=_inference_limiter)


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks and stop as soon as it is too big, instead of buffering it all."""
    chunks, size = [], 0
    while chunk := await file.read(MB):
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(413, f"File too large (limit {max_bytes // MB} MB).")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "Empty upload.")
    return data


def _video_suffix(filename: str | None) -> str:
    suffix = Path(filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, f"Unsupported video type. Use one of: {', '.join(sorted(VIDEO_SUFFIXES))}.")
    return suffix


def _too_many_failures(username: str) -> bool:
    now = time.monotonic()
    with _login_lock:
        attempts = _login_failures[username]
        while attempts and now - attempts[0] > LOGIN_WINDOW_SECONDS:
            attempts.popleft()
        return len(attempts) >= LOGIN_MAX_FAILURES


def _record_failure(username: str) -> None:
    with _login_lock:
        _login_failures[username].append(time.monotonic())


def _extract_token(auth_header: str | None) -> str:
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return auth_header.split(" ", 1)[1].strip()


def get_current_user(
    session: Annotated[Session, Depends(get_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    token = _extract_token(authorization)
    claims = decode_access_token(token)
    if claims.get("scope") != "api":
        raise HTTPException(status_code=401, detail="Wrong token type.")
    user = session.exec(select(User).where(User.username == claims["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    if not claims.get("mfa"):
        raise HTTPException(status_code=403, detail="2FA verification required.")
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


@router.get("/health")
def health():
    return {"status": "ok", "demo": settings.demo_mode}


@router.get("/")
def root():
    return RedirectResponse(url="/dashboard", status_code=307)


@router.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@router.get("/model/status")
def model_status(_: Annotated[User, Depends(get_current_user)]):
    return inference_service.model_status()


@router.get("/model/retrain-status")
def retrain_status(_: Annotated[User, Depends(get_current_user)]):
    return retrain_service.status


@router.post("/model/switch")
def model_switch(
    payload: dict,
    _: User = Depends(require_admin),
):
    model_name = payload.get("model")
    violence_model_name = payload.get("violence_model")
    
    if not model_name and not violence_model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' or 'violence_model' field.")
    
    try:
        result = {}
        if model_name:
            result = inference_service.switch_model(model_name)
        if violence_model_name:
            result = inference_service.switch_violence_model(violence_model_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _roi() -> dict:
    return {
        "roi_x1": settings.roi_x1, "roi_y1": settings.roi_y1,
        "roi_x2": settings.roi_x2, "roi_y2": settings.roi_y2,
    }


@router.get("/roi/config")
def get_roi(_: Annotated[User, Depends(get_current_user)]):
    return _roi()


@router.post("/roi/config")
def set_roi(payload: dict = Body(...), _: User = Depends(require_admin)):
    logger.info("POST /roi/config payload=%s", payload)
    new = _roi()
    try:
        for key in new:
            if key in payload:
                new[key] = float(payload[key])
    except (TypeError, ValueError):
        raise HTTPException(400, "ROI values must be numbers.")
    if not all(0.0 <= v <= 1.0 for v in new.values()):
        raise HTTPException(400, "ROI values must be between 0.0 and 1.0.")
    if new["roi_x1"] >= new["roi_x2"] or new["roi_y1"] >= new["roi_y2"]:
        raise HTTPException(400, "ROI must have x1 < x2 and y1 < y2.")
    for key, val in new.items():  # apply only after the whole box is valid
        setattr(settings, key, val)
    result = _roi()
    logger.info("ROI updated → %s", result)
    return result


# ── Unified video scan (all models) ─────────────────────────────────────────
@router.post("/scan/video")
async def scan_video(
    file: UploadFile = File(...),
    bus_id: str | None = Form(None, pattern=ID_PATTERN),
    route_id: str | None = Form(None, pattern=ID_PATTERN),
    timeslot: str | None = Form(None, pattern=ID_PATTERN),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Upload a video clip and run ALL detection models:
    YOLO people detection, gender/seat violations, violence classification.
    Optionally tag the result to a registered bus/route to update trust scores.
    """
    logger.info("POST /scan/video  filename=%s  bus=%s  route=%s", file.filename, bus_id, route_id)
    from app.services.evidence import hash_bytes, EVIDENCE_DIR

    suffix = _video_suffix(file.filename)
    data = await _read_upload(file, MAX_VIDEO_BYTES)
    digest = hash_bytes(data)

    # Save as manual scan evidence in case we receive feedback
    filename = f"manual_{digest[:12]}{suffix}"
    dest_path = EVIDENCE_DIR / filename
    dest_path.write_bytes(data)

    try:
        result = await _infer(inference_service.analyze_video_unified, str(dest_path))
        result["reference_id"] = filename
        result["bus_id"] = bus_id
        result["route_id"] = route_id
        result["timeslot"] = timeslot

        # ── Create ViolationEvent + update trust if fleet context provided ──
        if bus_id and route_id and timeslot:
            event = ViolationEvent(
                bus_id=bus_id,
                route_id=route_id,
                timeslot=timeslot,
                seat_misuse=bool(result.get("seat_violations", 0) > 0),
                violence=bool(result.get("violence", False)),
                people_count=int(result.get("people_count_max", 0)),
                aggression_score=float(result.get("violence_score", 0.0)),
                severity=result.get("severity", "low"),
                summary=result.get("summary", "Manual video scan"),
                evidence_hash=digest,
            )
            session.add(event)
            session.commit()
            session.refresh(event)

            # Create alert if warranted
            alert = create_alert(session, event)
            apply_trust_impact(session, event)
            append_audit("scan_event", {
                "event_id": event.id,
                "bus_id": bus_id,
                "route_id": route_id,
                "severity": event.severity,
            })

            result["event_id"] = event.id
            result["alert_id"] = alert.id if alert else None
            logger.info("Scan created event #%d  alert=%s", event.id, alert.id if alert else None)

        logger.info("Unified scan result: %s", result)
        return result
    except Exception:
        logger.exception("Unified scan FAILED")
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(500, "Video analysis failed. Check that the file is a readable video.")



@router.post("/stream/upload")
async def upload_stream_file(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
):
    """Upload a file to be used as a stream source."""
    suffix = _video_suffix(file.filename)
    data = await _read_upload(file, MAX_VIDEO_BYTES)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    _uploaded_streams.add(tmp_path)
    
    logger.info("Stream file uploaded: %s", tmp_path)
    return {"path": tmp_path, "filename": file.filename}


@router.post("/stream/token")
def stream_token(user: Annotated[User, Depends(get_current_user)]):
    """One-minute token for /stream/live, since an <img> tag cannot send an Authorization header."""
    return {"token": create_stream_token(user.username)}


@router.get("/stream/live")
def stream_live(token: str, source: str = "0"):
    """MJPEG stream with YOLO annotations. source=0 for webcam, or a file/RTSP path."""
    if decode_access_token(token).get("scope") != "stream":
        raise HTTPException(401, "Stream token required.")
    if not (source.isdigit() or source.startswith("rtsp://") or source in _uploaded_streams):
        raise HTTPException(400, "Source must be a camera index, an rtsp:// URL or an uploaded file.")
    _stream_cancel.clear()
    cam_source = int(source) if source.isdigit() else source

    def generate():
        cap = cv2.VideoCapture(cam_source)
        if not cap.isOpened():
            err = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(err, f"Cannot open: {source}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, buf = cv2.imencode(".jpg", err)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            return

        frame_buffer = deque(maxlen=16)
        violence_score = 0.0
        frame_count = 0

        try:
            while not _stream_cancel.is_set():
                ok, frame = cap.read()
                if not ok:
                    if isinstance(cam_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok:
                        break

                # Check cancel again after (potentially slow) read
                if _stream_cancel.is_set():
                    break

                frame_buffer.append(frame)
                frame_count += 1

                # Violence AI every 8 frames — skip if cancel requested
                if len(frame_buffer) == 16 and frame_count % 8 == 0 and not _stream_cancel.is_set():
                    try:
                        violence_score = inference_service.classify_violence_frames(list(frame_buffer))
                    except Exception:
                        pass

                # Check cancel before expensive annotation
                if _stream_cancel.is_set():
                    break

                annotated = inference_service.analyze_frame_annotated(frame, violence_score=violence_score)

                # Check cancel before yielding
                if _stream_cancel.is_set():
                    break

                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        finally:
            cap.release()
            logger.info("Stream generator stopped, camera released")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/stream/stop")
def stream_stop(_: Annotated[User, Depends(get_current_user)]):
    """Signal the live stream generator to stop and release the camera."""
    logger.info("POST /stream/stop — setting cancel event")
    _stream_cancel.set()
    return {"status": "stopped"}


@router.get("/dashboard")
def dashboard():
    return FileResponse(Path("dashboard") / "index.html")


@router.post("/auth/register")
def register(payload: RegisterRequest, session: Annotated[Session, Depends(get_session)]):
    existing = session.exec(select(User).where(User.username == payload.username)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")
    if payload.role not in {"admin", "operator"}:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'operator'.")
    # Only the very first account may self-register as admin.
    if payload.role == "admin" and session.exec(select(User)).first() is not None:
        raise HTTPException(status_code=403, detail="Only the first user can register as admin.")
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
    if _too_many_failures(payload.username):
        raise HTTPException(status_code=429, detail="Too many failed logins. Try again in a few minutes.")
    user = session.exec(select(User).where(User.username == payload.username)).first()
    if not user or not verify_password(payload.password, user.password_hash):
        _record_failure(payload.username)
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    with _login_lock:
        _login_failures.pop(payload.username, None)
    needs_2fa = settings.require_2fa
    token = create_access_token(user.username, mfa=not needs_2fa)
    return {"access_token": token, "token_type": "bearer", "twofa_required": needs_2fa}


@router.post("/auth/verify-2fa")
def verify_2fa(payload: Verify2FARequest, session: Annotated[Session, Depends(get_session)]):
    claims = decode_access_token(payload.token)
    if claims.get("scope") != "api":
        raise HTTPException(status_code=401, detail="Wrong token type.")
    username = claims["sub"]
    if _too_many_failures("2fa:" + username):
        raise HTTPException(status_code=429, detail="Too many wrong codes. Log in again later.")
    user = session.exec(select(User).where(User.username == username)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    if not verify_2fa_code(user.twofa_secret, payload.code):
        _record_failure("2fa:" + username)
        raise HTTPException(status_code=401, detail="Invalid 2FA code.")
    user.twofa_verified = True
    session.add(user)
    session.commit()
    return {"verified": True, "access_token": create_access_token(user.username, mfa=True)}


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
    bus_id: Annotated[str, Query(pattern=ID_PATTERN)],
    route_id: Annotated[str, Query(pattern=ID_PATTERN)],
    timeslot: Annotated[str, Query(pattern=ID_PATTERN)],
    reserved_seats_total: Annotated[int, Query(ge=0, le=100)] = 4,
    special_category_present: bool = False,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    raw = await _read_upload(file, MAX_IMAGE_BYTES)
    npbuf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image payload.")

    frame_analysis = await _infer(
        inference_service.analyze_frame,
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
    raw = await _read_upload(file, MAX_IMAGE_BYTES)
    npbuf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image.")
    annotated = await _infer(inference_service.analyze_frame_annotated, frame)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/ingest/evidence")
async def ingest_evidence(
    bus_id: Annotated[str, Query(pattern=ID_PATTERN)],
    event_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    event = session.get(ViolationEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")
    raw = await _read_upload(file, MAX_VIDEO_BYTES)
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
    """Recompute the whole hash chain and report whether any entry was tampered with."""
    return verify_chain()


@router.post("/feedback")
def submit_feedback(
    payload: FeedbackSubmitRequest,
    session: Annotated[Session, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Submit feedback to improve ML model accuracy and collect hard examples."""
    feedback = feedback_service.process_feedback(payload, user.username, session)
    return {"status": "success", "feedback_id": feedback.id}


# ── Fleet management ─────────────────────────────────────────────────────────

@router.get("/fleet/buses")
def list_buses(
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    return session.exec(select(Bus).order_by(Bus.bus_id)).all()


@router.post("/fleet/buses", status_code=201)
def create_bus(
    payload: BusCreate,
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    existing = session.exec(select(Bus).where(Bus.bus_id == payload.bus_id)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Bus '{payload.bus_id}' already exists.")
    bus = Bus(**payload.model_dump())
    session.add(bus)
    session.commit()
    session.refresh(bus)
    return bus


@router.delete("/fleet/buses/{bus_id}", status_code=204)
def delete_bus(
    bus_id: str,
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(require_admin)],
):
    bus = session.exec(select(Bus).where(Bus.bus_id == bus_id)).first()
    if not bus:
        raise HTTPException(status_code=404, detail="Bus not found.")
    session.delete(bus)
    session.commit()


@router.get("/fleet/routes")
def list_routes(
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    return session.exec(select(Route).order_by(Route.route_id)).all()


@router.post("/fleet/routes", status_code=201)
def create_route(
    payload: RouteCreate,
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
):
    existing = session.exec(select(Route).where(Route.route_id == payload.route_id)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Route '{payload.route_id}' already exists.")
    route = Route(**payload.model_dump())
    session.add(route)
    session.commit()
    session.refresh(route)
    return route


@router.delete("/fleet/routes/{route_id}", status_code=204)
def delete_route(
    route_id: str,
    session: Annotated[Session, Depends(get_session)],
    _: Annotated[User, Depends(require_admin)],
):
    route = session.exec(select(Route).where(Route.route_id == route_id)).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found.")
    session.delete(route)
    session.commit()
