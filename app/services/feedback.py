import logging
import shutil
from pathlib import Path

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.domain import Feedback, FeedbackSubmitRequest, ViolationEvent

logger = logging.getLogger("transport.feedback")

DATASET_ROOT = Path("dataset/feedback")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class FeedbackService:
    def __init__(self, retrain_service=None):
        self.evidence_dir = Path("evidence")
        self._retrain_service = retrain_service
        
        # Datasets directories for model retraining
        self.violence_feedback_dir = DATASET_ROOT / "Violence"
        self.noviolence_feedback_dir = DATASET_ROOT / "NoViolence"

        _ensure_dir(self.violence_feedback_dir)
        _ensure_dir(self.noviolence_feedback_dir)

    def process_feedback(self, payload: FeedbackSubmitRequest, username: str, session: Session) -> Feedback:
        """
        Record feedback and collect hard examples for retraining if the prediction was incorrect.
        """
        # 1. Record the feedback in DB
        feedback_record = Feedback(
            event_reference_id=payload.event_reference_id,
            feedback_type=payload.feedback_type,
            is_correct=payload.is_correct,
            comments=payload.comments,
            created_by=username
        )
        session.add(feedback_record)
        session.commit()
        session.refresh(feedback_record)

        # 2. Hard Example Mining: if the AI made a mistake on violence classification, copy the evidence
        if not payload.is_correct and "violence" in payload.feedback_type.lower():
            try:
                self._collect_hard_example(payload, session)
            except Exception as e:
                logger.error(f"Failed to collect hard example for retraining: {e}")
                # We do not fail the request if just the file copy fails

            # 3. Trigger background retraining
            if self._retrain_service is not None:
                self._retrain_service.retrain_async()
        
        return feedback_record

    def _collect_hard_example(self, payload: FeedbackSubmitRequest, session: Session) -> None:
        """
        Finds the evidence file and copies it to the appropriate dataset directory based on the user correction.
        """
        # Event reference ID might end with a file extension from a manual scan, or it might be raw
        ref_id = payload.event_reference_id
        if ref_id.isdigit():
            # Alert/event ID from live ingest: find the evidence saved for that event.
            event = session.get(ViolationEvent, int(ref_id))
            if not event or not event.evidence_hash:
                logger.warning("No evidence hash found for event id '%s'", ref_id)
                return
            key = event.evidence_hash[:12]
            matches = [f for f in self.evidence_dir.glob("*.bin") if f.stem.endswith("_" + key)]
            found_file = matches[0] if matches else None
        else:
            # Manual scan file name; exact match only (format is checked by the request model).
            found_file = self.evidence_dir / ref_id

        if not found_file or not found_file.is_file():
            logger.warning("No evidence found to copy for reference '%s'", ref_id)
            return

        # Determine correct destination
        # If user says violence_false_positive is `incorrect` -> Actually NO Violence
        # If user says violence_false_negative is `incorrect` -> Actually Violence
        # We'll simplify:
        # If feedback suggests the model falsely triggered violence (False Positive): it means it's NoViolence.
        # If feedback suggests the model missed violence (False Negative): it means it's Violence.
        
        dest_dir = self.noviolence_feedback_dir
        if "negative" in payload.feedback_type.lower() or "missed" in payload.feedback_type.lower() or payload.feedback_type.lower() == "violence_true":
            dest_dir = self.violence_feedback_dir

        dest_file = dest_dir / f"feedback_{found_file.name}"
        logger.info(f"Copying hard example {found_file} -> {dest_file}")
        shutil.copy2(found_file, dest_file)
