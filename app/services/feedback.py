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
        search_key = ref_id
        
        if ref_id.isdigit():
            # It's an alert/event ID from the live ingest
            event = session.get(ViolationEvent, int(ref_id))
            if event and event.evidence_hash:
                search_key = event.evidence_hash[:12]
            else:
                logger.warning(f"No evidence hash found for event id '{ref_id}'")
                return
        
        # Searching best match in evidence directory
        found_file = None
        for file in self.evidence_dir.glob("*"):
            if search_key in file.name:
                found_file = file
                break
        
        if not found_file or not found_file.exists():
            logger.warning(f"No evidence found to copy for search key '{search_key}'")
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
