"""
Online incremental retraining service for the violence classifier.

After the user submits feedback marking a prediction as incorrect, this service
fine-tunes the model on accumulated hard examples and hot-reloads the weights
into the running inference service.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logger = logging.getLogger("transport.retrain")

FEEDBACK_DIR = Path("dataset/feedback")
MODEL_PATH = Path("models/violence_classifier.pth")

FRAMES_PER_CLIP = 16
IMG_SIZE = 112
FINETUNE_EPOCHS = 3
FINETUNE_LR = 5e-5
FINETUNE_BATCH = 4


class _FeedbackDataset(Dataset):
    """Loads video clips from the feedback directories."""

    def __init__(self, transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform

        violence_dir = FEEDBACK_DIR / "Violence"
        noviolence_dir = FEEDBACK_DIR / "NoViolence"

        if violence_dir.exists():
            for p in violence_dir.glob("*"):
                if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                    self.samples.append((p, 1))

        if noviolence_dir.exists():
            for p in noviolence_dir.glob("*"):
                if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                    self.samples.append((p, 0))

        logger.info("Feedback dataset: %d clips (V=%d, NV=%d)",
                     len(self.samples),
                     sum(1 for _, l in self.samples if l == 1),
                     sum(1 for _, l in self.samples if l == 0))

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * FRAMES_PER_CLIP

        indices = np.linspace(0, total - 1, FRAMES_PER_CLIP, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        cap.release()

        while len(frames) < FRAMES_PER_CLIP:
            frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._sample_frames(path)
        if self.transform:
            frames = [self.transform(f) for f in frames]
        else:
            frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
        clip = torch.stack(frames)
        return clip, torch.tensor(label, dtype=torch.float32)


class RetrainService:
    """Manages background fine-tuning of the violence classifier from feedback."""

    def __init__(self, inference_service):
        self._inference_service = inference_service
        self._lock = threading.Lock()
        self._is_training = False
        self._last_retrain: str | None = None
        self._last_result: str | None = None

    @property
    def status(self) -> dict:
        return {
            "is_training": self._is_training,
            "last_retrain": self._last_retrain,
            "last_result": self._last_result,
        }

    def retrain_async(self):
        """Kick off retraining in a background thread (non-blocking)."""
        if self._is_training:
            logger.info("Retrain already in progress, skipping")
            return
        thread = threading.Thread(target=self._retrain, daemon=True)
        thread.start()

    def _retrain(self):
        """Fine-tune the violence model on feedback hard examples."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._is_training = True
            logger.info("Starting incremental retrain from feedback...")

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            ds = _FeedbackDataset(transform=transform)
            if len(ds) == 0:
                logger.info("No feedback examples found, skipping retrain")
                self._last_result = "skipped: no examples"
                return

            loader = DataLoader(ds, batch_size=FINETUNE_BATCH, shuffle=True, num_workers=0)

            # Load current model weights as starting point
            if not MODEL_PATH.exists():
                logger.warning("No base model at %s, cannot retrain", MODEL_PATH)
                self._last_result = "skipped: no base model"
                return

            # Import the model classes from inference and models
            from app.services.inference import _ViolenceClassifier
            from app.services.models.advanced_violence import TemporalPyramidNetwork, Int2D3DCNN

            device = self._inference_service._violence_device or torch.device("cpu")
            if device == "onnx":
                # Fallback to a real torch device if inferencing was on CPU ONNX
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

            model_type = ckpt.get("model_type", "legacy")
            if model_type == "int2d3d":
                model = Int2D3DCNN()
            elif model_type == "tpn":
                model = TemporalPyramidNetwork()
            else:
                model = _ViolenceClassifier()

            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).train()

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)

            # Fine-tune
            for epoch in range(1, FINETUNE_EPOCHS + 1):
                total_loss = 0.0
                all_preds = []
                all_labels = []
                for clips, labels in loader:
                    clips, labels = clips.to(device), labels.to(device)
                    optimizer.zero_grad()
                    logits = model(clips)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * clips.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # Calculate metrics
                acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(len(all_labels), 1)
                if _HAS_SKLEARN:
                    precision = precision_score(all_labels, all_preds, zero_division=0)
                    recall = recall_score(all_labels, all_preds, zero_division=0)
                    f1 = f1_score(all_labels, all_preds, zero_division=0)
                    logger.info("  Finetune epoch %d/%d — loss=%.4f  acc=%.3f  P=%.3f R=%.3f F1=%.3f",
                                epoch, FINETUNE_EPOCHS, total_loss / max(len(all_labels), 1), acc, precision, recall, f1)
                else:
                    logger.info("  Finetune epoch %d/%d — loss=%.4f  acc=%.3f",
                                epoch, FINETUNE_EPOCHS, total_loss / max(len(all_labels), 1), acc)

            # Save updated PyTorch model
            ckpt["model_state_dict"] = model.state_dict()
            ckpt["finetuned"] = True
            ckpt["finetune_examples"] = len(ds)
            torch.save(ckpt, MODEL_PATH)
            logger.info("Saved fine-tuned PyTorch model → %s", MODEL_PATH)

            # Export to ONNX for accelerated inference
            onnx_path = MODEL_PATH.with_suffix(".onnx")
            try:
                model.eval()
                # Create a sample input tensor (batch_size=1, time=16, channels=3, h=112, w=112)
                dummy_input = torch.randn(1, FRAMES_PER_CLIP, 3, IMG_SIZE, IMG_SIZE, device=device)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                logger.info("Successfully exported ONNX model → %s", onnx_path)
            except Exception as onnx_e:
                logger.warning("Failed to export ONNX model: %s", onnx_e)

            # Hot-reload into inference service
            self._inference_service._load_violence_model()
            logger.info("Hot-reloaded violence model into inference service")

            self._last_retrain = datetime.now().isoformat()
            self._last_result = f"success: {len(ds)} examples, {FINETUNE_EPOCHS} epochs"
            logger.info("Retrain complete: %s", self._last_result)

        except Exception as e:
            logger.exception("Retrain failed: %s", e)
            self._last_result = f"error: {e}"
        finally:
            self._is_training = False
            self._lock.release()
