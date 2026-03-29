"""
Kaggle Violence Detection Training Script
==========================================
Trains 3 violence detection architectures on multiple datasets:
  1. Legacy MobileNetV2 (temporal mean-pooling)
  2. Int2D-3D-CNN (spatial + temporal fusion)
  3. Temporal Pyramid Network (multi-scale temporal pooling)

Usage:
  - Upload to a Kaggle notebook
  - Add datasets: busviolence, Real Life Violence, RWF2000, SCVD
  - Enable GPU accelerator
  - Run all cells

Output: 3 model checkpoints saved + downloadable
"""

import os
import sys
import time
import random
import logging
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

# ── tqdm: works in both notebook and terminal ──────────────────────────────────
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("violence_trainer")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FRAMES_PER_CLIP = 16
IMG_SIZE = 224              # MobileNetV2 native resolution
BATCH_SIZE = 16             # T4 16GB can handle this
NUM_EPOCHS = 12             # Sufficient with large dataset + cosine decay
LR = 3e-4                   # Slightly higher LR with cosine decay
SEED = 42
VAL_SPLIT = 0.15            # 15% validation split
NUM_WORKERS = 2             # Kaggle has 2-4 CPUs
OUTPUT_DIR = Path("/kaggle/working/models")

# ── Dataset Discovery ─────────────────────────────────────────────────────────
# Folder names that indicate VIOLENCE (case-insensitive matching)
VIOLENCE_FOLDER_NAMES = {"violence", "fight", "weaponized"}
# Folder names that indicate NON-VIOLENCE
NONVIOLENCE_FOLDER_NAMES = {"nonviolence", "noviolence", "nonfight", "normal"}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

KAGGLE_INPUT = Path("/kaggle/input")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def discover_videos() -> tuple[list[Path], list[Path]]:
    """Auto-discover violence and non-violence videos by scanning /kaggle/input/."""
    violence_videos = []
    nonviolence_videos = []

    if not KAGGLE_INPUT.exists():
        logger.error("❌ /kaggle/input/ not found — are you running on Kaggle?")
        return violence_videos, nonviolence_videos

    # First, print the directory tree so we can see what's there
    logger.info("  📁 Scanning /kaggle/input/ tree...")
    for dataset_dir in sorted(KAGGLE_INPUT.iterdir()):
        if dataset_dir.is_dir():
            logger.info("    📦 %s/", dataset_dir.name)

    # Now recursively find folders matching violence/nonviolence names
    for folder in KAGGLE_INPUT.rglob("*"):
        if not folder.is_dir():
            continue

        folder_lower = folder.name.lower()

        if folder_lower in VIOLENCE_FOLDER_NAMES:
            found = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
            if found:
                violence_videos.extend(found)
                logger.info("  ✓ %d violence videos  ← %s", len(found), folder.relative_to(KAGGLE_INPUT))

        elif folder_lower in NONVIOLENCE_FOLDER_NAMES:
            found = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
            if found:
                nonviolence_videos.extend(found)
                logger.info("  ✓ %d non-violence videos ← %s", len(found), folder.relative_to(KAGGLE_INPUT))

    return violence_videos, nonviolence_videos


class MultiDatasetViolence(Dataset):
    """Unified dataset that loads videos from multiple sources."""

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path: Path) -> list[np.ndarray]:
        """Uniformly sample FRAMES_PER_CLIP frames from a video."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ViolenceClassifier(nn.Module):
    """Legacy MobileNetV2 backbone + temporal average pooling + binary head."""

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = x.view(B, T, -1).mean(dim=1)
        x = self.head(x)
        return x.squeeze(-1)


class Int2D3DCNN(nn.Module):
    """Spatial (2D MobileNetV3) + Temporal (3D CNN) fusion architecture."""

    def __init__(self, num_classes=1):
        super().__init__()
        spatial_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.spatial_features = spatial_backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_branch = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(576 + 64, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 2D branch
        x_2d = x.view(B * T, C, H, W)
        feat_2d = self.spatial_features(x_2d)
        feat_2d = self.spatial_pool(feat_2d).view(B, T, -1).mean(dim=1)
        # 3D branch
        x_3d = x.transpose(1, 2)
        feat_3d = self.temporal_branch(x_3d).view(B, -1)
        # Fuse
        fused = torch.cat((feat_2d, feat_3d), dim=1)
        logits = self.head(fused)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


class TemporalPyramidNetwork(nn.Module):
    """TPN with multi-scale temporal pooling over ResNet18 features."""

    def __init__(self, num_classes=1):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_dim = 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.temporal_scales = [1, 2, 4]
        total_features = self.feature_dim * sum(self.temporal_scales)  # 512 * 7 = 3584
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_features, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(B * T, -1)
        features = features.view(B, T, -1).transpose(1, 2)  # (B, Feat, T)
        pooled = []
        for scale in self.temporal_scales:
            pool = nn.functional.adaptive_avg_pool1d(features, scale)
            pooled.append(pool.reshape(B, -1))
        concat = torch.cat(pooled, dim=1)
        logits = self.head(concat)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
) -> dict:
    """Train a single model with full logging and progress bars."""

    logger.info("=" * 60)
    logger.info("  TRAINING: %s", model_name)
    logger.info("  Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    logger.info("  Device: %s  |  Epochs: %d  |  LR: %.1e", device, num_epochs, lr)
    logger.info("=" * 60)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    save_path = OUTPUT_DIR / f"{model_name}.pth"

    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"[{model_name}] Epochs", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"  Train E{epoch}", leave=False,
                         unit="batch", bar_format="{l_bar}{bar:20}{r_bar}")

        for clips, labels in train_bar:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * clips.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += clips.size(0)

            # Update progress bar with live metrics
            running_acc = train_correct / max(train_total, 1)
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.1%}")

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        val_tp, val_fp, val_tn, val_fn = 0, 0, 0, 0

        val_bar = tqdm(val_loader, desc=f"  Val   E{epoch}", leave=False,
                       unit="batch", bar_format="{l_bar}{bar:20}{r_bar}")

        with torch.no_grad():
            for clips, labels in val_bar:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += clips.size(0)

                # Confusion matrix
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        val_acc = val_correct / max(val_total, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Update epoch progress bar
        epoch_bar.set_postfix(
            loss=f"{avg_train_loss:.4f}",
            t_acc=f"{train_acc:.1%}",
            v_acc=f"{val_acc:.1%}",
            f1=f"{f1:.3f}",
            best=f"{best_acc:.1%}",
        )

        # Detailed epoch log
        logger.info(
            "  E%02d │ loss=%.4f │ train_acc=%.1f%% │ val_acc=%.1f%% │ "
            "P=%.3f R=%.3f F1=%.3f │ lr=%.1e%s",
            epoch, avg_train_loss, train_acc * 100, val_acc * 100,
            precision, recall, f1, current_lr,
            " ★ BEST" if val_acc > best_acc else "",
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_acc": val_acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "frames_per_clip": FRAMES_PER_CLIP,
                "img_size": IMG_SIZE,
                "architecture": model_name,
            }, save_path)

    logger.info("  ✓ %s — Best val_acc=%.1f%% at epoch %d → %s",
                model_name, best_acc * 100, best_epoch, save_path)
    logger.info("")
    return {"model_name": model_name, "best_acc": best_acc, "best_epoch": best_epoch, "history": history}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("🖥️  GPU: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        logger.info("⚠️  No GPU detected — training will be slow!")

    # ── Discover datasets ─────────────────────────────────────────────────────
    logger.info("📂 Scanning for datasets...")
    violence_vids, nonviolence_vids = discover_videos()

    logger.info("─" * 50)
    logger.info("📊 Total discovered:")
    logger.info("   Violence:     %d videos", len(violence_vids))
    logger.info("   Non-Violence: %d videos", len(nonviolence_vids))
    logger.info("   Total:        %d videos", len(violence_vids) + len(nonviolence_vids))
    logger.info("─" * 50)

    if not violence_vids or not nonviolence_vids:
        logger.error("❌ Need both violence and non-violence videos!")
        return

    # ── Build unified sample list ─────────────────────────────────────────────
    all_samples = [(p, 1) for p in violence_vids] + [(p, 0) for p in nonviolence_vids]
    random.shuffle(all_samples)

    # ── Train/Val split ───────────────────────────────────────────────────────
    val_size = int(len(all_samples) * VAL_SPLIT)
    train_size = len(all_samples) - val_size

    logger.info("📋 Split: %d train + %d val (%.0f%%/%.0f%%)",
                train_size, val_size, (1 - VAL_SPLIT) * 100, VAL_SPLIT * 100)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets with the split
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]

    train_ds = MultiDatasetViolence(train_samples, transform=train_transform)
    val_ds = MultiDatasetViolence(val_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    logger.info("🔧 Config: IMG_SIZE=%d  FRAMES=%d  BATCH=%d  EPOCHS=%d  LR=%.1e",
                IMG_SIZE, FRAMES_PER_CLIP, BATCH_SIZE, NUM_EPOCHS, LR)
    logger.info("")

    # ── Create output directory ───────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAIN TPN (best architecture)
    # ══════════════════════════════════════════════════════════════════════════
    results = []

    logger.info("🚀 Training: Temporal Pyramid Network (ResNet18 backbone)")
    model = TemporalPyramidNetwork()
    r = train_model(model, "temporal_pyramid_network", train_loader, val_loader, device)
    results.append(r)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("═" * 60)
    logger.info("  TRAINING COMPLETE — %.1f minutes total", elapsed / 60)
    logger.info("═" * 60)
    logger.info("")
    logger.info("  %-30s │ Best Acc │ Epoch", "Model")
    logger.info("  " + "─" * 50)
    for r in results:
        logger.info("  %-30s │  %5.1f%%  │  %d",
                     r["model_name"], r["best_acc"] * 100, r["best_epoch"])
    logger.info("")
    logger.info("  📦 Models saved to: %s", OUTPUT_DIR)
    logger.info("     Download them and place in your project's models/ folder.")
    logger.info("")

    # List output files
    for f in sorted(OUTPUT_DIR.glob("*.pth")):
        size_mb = f.stat().st_size / 1e6
        logger.info("     📄 %s (%.1f MB)", f.name, size_mb)


if __name__ == "__main__":
    main()
