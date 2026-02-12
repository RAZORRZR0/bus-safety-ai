"""
Train a binary Violence / Non-Violence classifier on the bus-violence dataset.

Usage:  python train_violence.py
Output: models/violence_classifier.pth  (best checkpoint)

Architecture:
  MobileNetV2 (ImageNet pretrained) → global average pool → FC(1280,1) → sigmoid
  Each video is represented by N uniformly-sampled frames; frame-level features
  are averaged before the classifier head (temporal mean-pooling).
"""

import os
import random
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Optional: AMD GPU via DirectML
try:
    import torch_directml
    DML_AVAILABLE = True
except ImportError:
    DML_AVAILABLE = False

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("bus-violence")
VIOLENCE_DIR = DATA_DIR / "Violence"
NOVIOLENCE_DIR = DATA_DIR / "NoViolence"
TRAIN_TXT = DATA_DIR / "train.txt"
TEST_TXT = DATA_DIR / "test.txt"
MODEL_OUT = Path("models/violence_classifier.pth")

FRAMES_PER_CLIP = 16       # number of frames to sample per video
IMG_SIZE = 112             # resize frames to 112x112
BATCH_SIZE = 8
NUM_EPOCHS = 12
LR = 1e-4
SEED = 42


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ViolenceDataset(Dataset):
    """Loads mp4 clips, samples FRAMES_PER_CLIP frames, returns (tensor, label)."""

    def __init__(self, file_list: list[str], transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform

        for fname in file_list:
            fname = fname.strip()
            if not fname:
                continue
            # Determine label + folder from prefix
            if fname.upper().startswith("VIOLENCE_"):
                path = VIOLENCE_DIR / fname
                label = 1
            elif fname.upper().startswith("NONVIOLENCE_"):
                path = NOVIOLENCE_DIR / fname
                label = 0
            else:
                continue
            if path.exists():
                self.samples.append((path, label))

        logger.info("Dataset: %d clips (Violence: %d, NonViolence: %d)",
                    len(self.samples),
                    sum(1 for _, l in self.samples if l == 1),
                    sum(1 for _, l in self.samples if l == 0))

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path: Path) -> list[np.ndarray]:
        """Uniformly sample FRAMES_PER_CLIP frames from the video."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            # Return black frames as fallback
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

        # Pad if needed
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

        # Stack to (FRAMES_PER_CLIP, C, H, W)
        clip = torch.stack(frames)
        return clip, torch.tensor(label, dtype=torch.float32)


# ─── Model ────────────────────────────────────────────────────────────────────
class ViolenceClassifier(nn.Module):
    """MobileNetV2 backbone + temporal average pooling + binary head."""

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the final classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)          # flatten batch & time
        x = self.features(x)                  # (B*T, 1280, h, w)
        x = self.pool(x).flatten(1)           # (B*T, 1280)
        x = x.view(B, T, -1).mean(dim=1)     # temporal mean → (B, 1280)
        x = self.head(x)                       # (B, 1)
        return x.squeeze(-1)


# ─── Training ─────────────────────────────────────────────────────────────────
def load_split(txt_path: Path) -> list[str]:
    """Read filenames from a split file."""
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = (torch_directml.device() if DML_AVAILABLE
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s%s", device,
                " (AMD DirectML)" if DML_AVAILABLE else "")

    # Transforms (ImageNet normalization for pretrained backbone)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_files = load_split(TRAIN_TXT)
    test_files = load_split(TEST_TXT)
    train_ds = ViolenceDataset(train_files, transform=train_transform)
    test_ds = ViolenceDataset(test_files, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Model
    model = ViolenceClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Create output dir
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * clips.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += clips.size(0)

            if (batch_idx + 1) % 20 == 0:
                logger.info("  Epoch %d  batch %d/%d  loss=%.4f",
                            epoch, batch_idx + 1, len(train_loader), loss.item())

        train_acc = train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)

        # ── Evaluate ──
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for clips, labels in test_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                test_correct += (preds == labels).sum().item()
                test_total += clips.size(0)

        test_acc = test_correct / max(test_total, 1)
        scheduler.step()

        logger.info("Epoch %d/%d — train_loss=%.4f  train_acc=%.3f  test_acc=%.3f  lr=%.2e",
                    epoch, NUM_EPOCHS, avg_train_loss, train_acc, test_acc,
                    optimizer.param_groups[0]['lr'])

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'frames_per_clip': FRAMES_PER_CLIP,
                'img_size': IMG_SIZE,
            }, MODEL_OUT)
            logger.info("  ✓ Saved best model (test_acc=%.3f) → %s", test_acc, MODEL_OUT)

    logger.info("Training complete. Best test accuracy: %.3f", best_acc)


if __name__ == "__main__":
    main()
