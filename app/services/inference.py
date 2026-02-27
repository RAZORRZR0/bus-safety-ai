from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.core.config import settings
from app.models.domain import EventIngestRequest
from app.services.models.advanced_violence import TemporalPyramidNetwork, Int2D3DCNN

logger = logging.getLogger("transport.inference")

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at runtime
    YOLO = None

# Optional DirectML for AMD GPUs
try:
    import torch_directml
    _DML = True
except ImportError:
    _DML = False


# ─── Violence Classifier Architecture (must match train_violence.py) ─────────
class _ViolenceClassifier(nn.Module):
    """MobileNetV2 backbone + temporal mean-pool + binary head."""
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = x.view(B, T, -1).mean(dim=1)
        return self.head(x).squeeze(-1)


@dataclass
class InferenceResult:
    seat_misuse: bool
    violence: bool
    severity: str
    summary: str
    people_count: int = 0


@dataclass
class FrameAnalysis:
    payload: EventIngestRequest
    people_count: int


AVAILABLE_MODELS = [
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
]


class InferenceService:
    """Inference layer with optional YOLO person detection and rule fallback."""

    def __init__(self):
        self._model = None
        self._lock = Lock()
        self._prev_frame_by_bus: dict[str, np.ndarray] = {}
        self._current_model_name: str = settings.yolo_model_path
        self._load_model()
        # Violence classifier
        self._violence_model = None
        self._violence_ort_session = None # ONNX Runtime session
        self._violence_device = None
        self._violence_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._load_violence_model()

    def _load_model(self) -> None:
        if YOLO is None:
            return
        try:
            self._model = YOLO(self._current_model_name)
        except Exception:
            self._model = None

    def model_status(self) -> dict[str, str | bool | list]:
        return {
            "yolo_available": self._model is not None,
            "current_model": self._current_model_name,
            "available_models": AVAILABLE_MODELS,
        }

    def switch_model(self, model_name: str) -> dict:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {AVAILABLE_MODELS}")
        self._current_model_name = model_name
        settings.yolo_model_path = model_name
        self._model = None
        self._load_model()
        return self.model_status()

    def _people_count(self, frame: np.ndarray) -> int:
        if self._model is None:
            return 0
        try:
            # COCO class id 0 = person
            results = self._model.predict(
                source=frame,
                conf=settings.yolo_conf_threshold,
                classes=[0],
                verbose=False,
            )
            if not results:
                return 0
            boxes = getattr(results[0], "boxes", None)
            return 0 if boxes is None else int(len(boxes))
        except Exception:
            return 0

    def _detect_persons_tracked(self, frame: np.ndarray):
        """Use model.track() for temporally consistent detections across frames.
        Returns the results object with tracked bounding boxes (includes IDs)."""
        if self._model is None:
            return None
        try:
            results = self._model.track(
                source=frame,
                conf=settings.yolo_conf_threshold,
                classes=[0],
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            if not results:
                return None
            return results[0]
        except Exception as e:
            logger.warning("Tracking failed, falling back to predict: %s", e)
            # Fallback to predict if tracking fails
            try:
                results = self._model.predict(
                    source=frame,
                    conf=settings.yolo_conf_threshold,
                    classes=[0],
                    verbose=False,
                )
                return results[0] if results else None
            except Exception:
                return None

    def _estimate_aggression(self, current: np.ndarray, previous: np.ndarray | None) -> float:
        """Fallback motion-heuristic aggression estimator."""
        if previous is None:
            return 0.0
        diff = cv2.absdiff(current, previous)
        motion = float(np.mean(diff)) / 255.0
        return min(1.0, motion * 2.4)

    # ── Violence classifier helpers ───────────────────────────────────────
    def _load_violence_model(self) -> None:
        """Load trained violence classifier: attempts ONNX first for speed, then PyTorch."""
        ckpt_path = Path(settings.violence_model_path)
        
        # 1. Attempt to load ONNX model for acceleration
        onnx_path = ckpt_path.with_suffix('.onnx')
        if ort is not None and onnx_path.exists():
            try:
                providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                self._violence_ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
                self._violence_device = "onnx"
                logger.info(f"Loaded ONNX violence model: {onnx_path.name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX violence model, falling back to PyTorch: {e}")

        # 2. PyTorch checkpoint fallback
        if not ckpt_path.exists():
            logger.info("Violence model not found at %s — will use motion heuristic", ckpt_path)
            return

        try:
            device = (torch_directml.device() if _DML
                      else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            # Support the advanced architectures from the research papers
            model_type = ckpt.get("model_type", "legacy")
            if model_type == "int2d3d":
                model = Int2D3DCNN()
            elif model_type == "tpn":
                model = TemporalPyramidNetwork()
            else:
                model = _ViolenceClassifier()
                
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            self._violence_model = model
            self._violence_device = device
            logger.info("Violence model loaded (%s, type=%s) — test_acc=%.3f",
                        device, model_type, ckpt.get("test_acc", -1))
        except Exception as e:
            logger.warning("Failed to load violence model: %s", e)

    def _prep_frames_for_violence(self, frames_bgr: list[np.ndarray],
                                   img_size: int = 112) -> torch.Tensor:
        """Preprocess a list of BGR frames into a (1, T, C, H, W) tensor."""
        tensors = []
        for f in frames_bgr:
            f_rgb = cv2.cvtColor(cv2.resize(f, (img_size, img_size)), cv2.COLOR_BGR2RGB)
            tensors.append(self._violence_transform(f_rgb))
        clip = torch.stack(tensors).unsqueeze(0)  # (1, T, C, H, W)
        return clip

    def classify_violence_frames(self, frames_bgr: list[np.ndarray]) -> float:
        """Run the violence classifier on a list of BGR frames."""
        if len(frames_bgr) == 0:
            return 0.0
            
        if self._violence_ort_session is None and self._violence_model is None:
            # Fallback to motion heuristic
            if len(frames_bgr) >= 2:
                return self._estimate_aggression(frames_bgr[-1], frames_bgr[0])
            return 0.0

        clip = self._prep_frames_for_violence(frames_bgr)
        
        # Accelerate inference if ONNX is available
        if self._violence_ort_session is not None:
            try:
                ort_inputs = {self._violence_ort_session.get_inputs()[0].name: clip.numpy()}
                ort_outs = self._violence_ort_session.run(None, ort_inputs)
                logit = float(ort_outs[0][0])
                score = 1.0 / (1.0 + np.exp(-logit)) # sigmoid
                logger.debug("ONNX Violence score: %.3f", score)
                return score
            except Exception as e:
                logger.warning(f"ONNX inference failed: {e}")
                # Fallthrough to PyTorch if available, though usually we only have one or the other.
                if self._violence_model is None:
                    return 0.0

        clip_t = clip.to(self._violence_device)
        with torch.no_grad():
            logit = self._violence_model(clip_t)
            score = float(torch.sigmoid(logit).item())
        logger.debug("PyTorch Violence score: %.3f", score)
        return score

    def classify_violence_video(self, video_path: str,
                                 n_frames: int = 16) -> dict:
        """Classify an entire video file. Returns {violence, confidence}."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return {"violence": False, "confidence": 0.0, "error": "Cannot read video"}

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        if not frames:
            return {"violence": False, "confidence": 0.0, "error": "No frames read"}

        confidence = self.classify_violence_frames(frames)
        return {
            "violence": confidence >= settings.violence_threshold,
            "confidence": round(confidence, 4),
            "threshold": settings.violence_threshold,
            "frames_sampled": len(frames),
        }

    def analyze_frame(
        self,
        bus_id: str,
        route_id: str,
        timeslot: str,
        frame_bgr: np.ndarray,
        reserved_seats_total: int = 4,
        special_category_present: bool = False,
    ) -> FrameAnalysis:
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid frame dimensions.")

        resized = cv2.resize(frame_bgr, (640, 360))
        people_count = self._people_count(resized)

        # Reserved-seat proxy ROI; replace with calibrated seat polygons in production.
        roi = resized[int(0.2 * 360) : int(0.55 * 360), int(0.55 * 640) : int(0.95 * 640)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        occupied_proxy = float(np.mean(gray > 120))
        reserved_seats_occupied = min(reserved_seats_total, int(round(occupied_proxy * reserved_seats_total)))

        with self._lock:
            previous = self._prev_frame_by_bus.get(bus_id)
            aggression_score = self._estimate_aggression(resized, previous)
            self._prev_frame_by_bus[bus_id] = resized

        payload = EventIngestRequest(
            bus_id=bus_id,
            route_id=route_id,
            timeslot=timeslot,
            reserved_seats_total=reserved_seats_total,
            reserved_seats_occupied=reserved_seats_occupied,
            special_category_present=special_category_present,
            aggression_score=aggression_score,
            model_confidence=0.8 if self._model is None else 0.9,
        )
        return FrameAnalysis(payload=payload, people_count=people_count)

    def _classify_gender(self, crop_bgr: np.ndarray) -> str:
        """Classify gender of a person crop using face detection on the head region.
        Returns 'Male', 'Female', or 'Unknown'."""
        try:
            from deepface import DeepFace

            h_crop, w_crop = crop_bgr.shape[:2]
            if h_crop < 30 or w_crop < 20:
                return "Unknown"

            # Extract the upper 40% of the person bounding box (head/face region)
            head_region = crop_bgr[0:int(h_crop * 0.4), :]

            # Attempt 1: Use face detection on head region (most accurate)
            try:
                result = DeepFace.analyze(
                    head_region,
                    actions=["gender"],
                    enforce_detection=True,
                    silent=True,
                    detector_backend="opencv",
                )
                if isinstance(result, list):
                    result = result[0]
            except (ValueError, Exception):
                # Face not detected in head crop — fallback to head crop without detection
                try:
                    result = DeepFace.analyze(
                        head_region,
                        actions=["gender"],
                        enforce_detection=False,
                        silent=True,
                        detector_backend="skip",
                    )
                    if isinstance(result, list):
                        result = result[0]
                except Exception:
                    return "Unknown"

            gender_scores = result.get("gender", {})
            dominant = result.get("dominant_gender", "Unknown")
            logger.debug("  Gender result: dominant=%s scores=%s", dominant, gender_scores)

            # Apply confidence threshold — only trust strong predictions
            man_score = gender_scores.get("Man", 0)
            woman_score = gender_scores.get("Woman", 0)
            confidence = max(man_score, woman_score)

            if confidence < 65:  # Below 65% confidence → unreliable
                return "Unknown"

            if dominant == "Man":
                return "Male"
            elif dominant == "Woman":
                return "Female"
            return "Unknown"
        except Exception as e:
            logger.warning("  Gender classification failed: %s", e)
            return "Unknown"

    @staticmethod
    def _box_in_roi(bx1, by1, bx2, by2, rx1, ry1, rx2, ry2) -> bool:
        """Check if the center of a bounding box falls within the ROI."""
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def analyze_frame_annotated(
        self,
        frame_bgr: np.ndarray,
        violence_score: float = 0.0,
    ) -> np.ndarray:
        """Run YOLO + gender classification and return annotated frame.
        
        Args:
            frame_bgr: Input frame
            violence_score: Optional violence confidence (0.0-1.0). If > threshold, shows warning.
        """
        logger.info("analyze_frame_annotated: frame=%dx%d", frame_bgr.shape[1], frame_bgr.shape[0])
        annotated = frame_bgr.copy()
        resized = cv2.resize(annotated, (640, 360))
        h, w = 360, 640

        # Configurable reserved-seat ROI from settings
        roi_x1 = int(settings.roi_x1 * w)
        roi_y1 = int(settings.roi_y1 * h)
        roi_x2 = int(settings.roi_x2 * w)
        roi_y2 = int(settings.roi_y2 * h)
        logger.debug("  ROI config: (%.2f,%.2f)-(%.2f,%.2f) -> px (%d,%d)-(%d,%d)",
                     settings.roi_x1, settings.roi_y1, settings.roi_x2, settings.roi_y2,
                     roi_x1, roi_y1, roi_x2, roi_y2)

        # Draw ROI rectangle (cyan dashed effect via thick border)
        overlay = resized.copy()
        cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, resized, 0.9, 0, resized)
        cv2.rectangle(resized, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
        cv2.putText(resized, "RESERVED SEAT (Female)", (roi_x1 + 4, roi_y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        violations = 0
        people_count = 0

        if self._model is not None:
            try:
                # Use tracking for temporally consistent detections
                result = self._detect_persons_tracked(resized)
                if result is not None and result.boxes is not None:
                    people_count = len(result.boxes)
                    # Get track IDs if available
                    track_ids = result.boxes.id
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        tid = int(track_ids[i]) if track_ids is not None else None

                        # Crop person for gender classification
                        crop = resized[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        gender = "Unknown"
                        if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                            gender = self._classify_gender(crop)

                        in_roi = self._box_in_roi(x1, y1, x2, y2, roi_x1, roi_y1, roi_x2, roi_y2)

                        # Track ID prefix
                        id_tag = f"#{tid} " if tid is not None else ""

                        # Color coding
                        if gender == "Male" and in_roi:
                            color = (0, 0, 255)       # RED — violation
                            label = f"{id_tag}MALE {conf:.0%} [VIOLATION]"
                            violations += 1
                        elif gender == "Male":
                            color = (255, 128, 0)     # BLUE-ish — male outside ROI
                            label = f"{id_tag}Male {conf:.0%}"
                        elif gender == "Female":
                            color = (255, 0, 255)     # PINK
                            label = f"{id_tag}Female {conf:.0%}"
                        else:
                            if in_roi:
                                color = (0, 165, 255) # ORANGE — unknown in ROI
                                label = f"{id_tag}Occupant {conf:.0%} [CHECK]"
                            else:
                                color = (128, 128, 128)  # GRAY
                                label = f"{id_tag}Person {conf:.0%}"

                        # Draw bounding box
                        thickness = 3 if (gender == "Male" and in_roi) else 2
                        cv2.rectangle(resized, (x1, y1), (x2, y2), color, thickness)

                        # Draw label background + text
                        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(resized, (x1, y1 - th_t - 8), (x1 + tw + 6, y1), color, -1)
                        cv2.putText(resized, label, (x1 + 3, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            except Exception as e:
                cv2.putText(resized, f"YOLO ERROR: {str(e)[:40]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(resized, "YOLO MODEL NOT LOADED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Status bar at bottom
        cv2.rectangle(resized, (0, h - 30), (w, h), (0, 0, 0), -1)
        status = f"People: {people_count}  |  Violations: {violations}"
        cv2.putText(resized, status, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Violation banner
        if violations > 0:
            cv2.rectangle(resized, (0, 0), (w, 32), (0, 0, 200), -1)
            cv2.putText(resized, f"!! RESERVED SEAT VIOLATION DETECTED ({violations}) !!",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif violence_score > 0.75:
            cv2.rectangle(resized, (0, 0), (w, 32), (0, 0, 200), -1)
            cv2.putText(resized, f"!! VIOLENCE DETECTED (Score: {violence_score:.2f}) !!",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return resized

    def analyze(self, payload: EventIngestRequest) -> InferenceResult:
        seat_misuse = (
            payload.reserved_seats_occupied > 0
            and not payload.special_category_present
            and payload.reserved_seats_total > 0
        )
        violence = payload.aggression_score >= 0.75

        if violence and payload.aggression_score >= 0.9:
            severity = "critical"
        elif violence or (seat_misuse and payload.aggression_score >= 0.5):
            severity = "high"
        elif seat_misuse:
            severity = "medium"
        else:
            severity = "low"

        summary_parts = []
        if seat_misuse:
            summary_parts.append("Reserved seat misuse detected")
        if violence:
            summary_parts.append(f"Aggressive behavior score={payload.aggression_score:.2f}")
        if not summary_parts:
            summary_parts.append("No violation detected")

        return InferenceResult(
            seat_misuse=seat_misuse,
            violence=violence,
            severity=severity,
            summary="; ".join(summary_parts),
            people_count=0,
        )

    # ── Unified video analysis ────────────────────────────────────────────
    def analyze_video_unified(
        self, video_path: str, n_frames: int = 16
    ) -> dict:
        """Run ALL models on a video file and return combined results.

        Combines: YOLO people detection, gender/seat violations, violence classifier.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total_frames <= 0:
            cap.release()
            return {"error": "Cannot read video"}

        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        sampled_frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                sampled_frames.append(frame)
        cap.release()

        if not sampled_frames:
            return {"error": "No frames could be read"}

        # ── Violence detection (on all sampled frames) ──
        violence_score = self.classify_violence_frames(sampled_frames)
        violence_detected = violence_score >= settings.violence_threshold

        # ── YOLO + gender analysis (on representative frames) ──
        # Analyze a subset of frames for people/seat violations
        analysis_frames = sampled_frames[::max(1, len(sampled_frames) // 4)]  # ~4 frames
        total_people = 0
        max_violations = 0
        gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
        max_people_in_frame = 0

        h, w_target = 360, 640
        roi_x1 = int(settings.roi_x1 * w_target)
        roi_y1 = int(settings.roi_y1 * w_target)
        roi_x2 = int(settings.roi_x2 * w_target)
        roi_y2 = int(settings.roi_y2 * h)

        for frame in analysis_frames:
            resized = cv2.resize(frame, (w_target, h))
            if self._model is None:
                continue
            try:
                results = self._model.predict(
                    source=resized, conf=settings.yolo_conf_threshold,
                    classes=[0], verbose=False,
                )
                if not results or results[0].boxes is None:
                    continue
                frame_people = len(results[0].boxes)
                total_people += frame_people
                max_people_in_frame = max(max_people_in_frame, frame_people)

                frame_genders = {"Male": 0, "Female": 0, "Unknown": 0}
                frame_violations = 0

                for box in results[0].boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    crop = resized[max(0, by1):min(h, by2), max(0, bx1):min(w_target, bx2)]
                    gender = "Unknown"
                    if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                        gender = self._classify_gender(crop)
                    frame_genders[gender] = frame_genders.get(gender, 0) + 1
                    in_roi = self._box_in_roi(bx1, by1, bx2, by2, roi_x1, roi_y1, roi_x2, roi_y2)
                    if gender == "Male" and in_roi:
                        frame_violations += 1
                        
                for g in frame_genders:
                    gender_counts[g] = max(gender_counts.get(g, 0), frame_genders[g])
                max_violations = max(max_violations, frame_violations)
            except Exception as e:
                logger.warning("YOLO error in unified analysis: %s", e)

        n_analysis = max(len(analysis_frames), 1)
        avg_people = total_people / n_analysis

        # Sanity: total genders should never exceed max people in any frame
        total_gender = sum(gender_counts.values())
        if total_gender > max_people_in_frame and max_people_in_frame > 0:
            scale = max_people_in_frame / total_gender
            gender_counts = {k: max(0, round(v * scale)) for k, v in gender_counts.items()}

        # ── Severity calculation ──
        seat_misuse = max_violations > 0
        if violence_detected and violence_score >= 0.9:
            severity = "critical"
        elif violence_detected or (seat_misuse and violence_score >= 0.5):
            severity = "high"
        elif seat_misuse:
            severity = "medium"
        else:
            severity = "low"

        # ── Summary ──
        summary_parts = []
        if seat_misuse:
            summary_parts.append(f"Reserved seat violation ({max_violations} instances)")
        if violence_detected:
            summary_parts.append(f"Violence detected (confidence={violence_score:.2f})")
        if not summary_parts:
            summary_parts.append("No violations detected")

        return {
            "violence": violence_detected,
            "violence_score": round(violence_score, 4),
            "violence_threshold": settings.violence_threshold,
            "seat_misuse": seat_misuse,
            "seat_violations": max_violations,
            "people_count_avg": round(avg_people, 1),
            "people_count_max": max_people_in_frame,
            "gender_breakdown": gender_counts,
            "severity": severity,
            "summary": "; ".join(summary_parts),
            "frames_analyzed": len(analysis_frames),
            "frames_sampled": len(sampled_frames),
            "video_duration_sec": round(total_frames / fps, 1),
        }

