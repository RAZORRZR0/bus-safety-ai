"""Evaluate the trained violence classifiers on the Bus Violence test split.

Writes research_plots/metrics.json, which generate_plots.py reads.
Usage: python scripts/evaluate.py [models/*.pth ...]
"""
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_violence as tv  # noqa: E402
from app.services.inference import _ViolenceClassifier  # noqa: E402
from app.services.models.advanced_violence import Int2D3DCNN, TemporalPyramidNetwork  # noqa: E402

ARCHS = {"int2d3d": Int2D3DCNN, "tpn": TemporalPyramidNetwork, "legacy": _ViolenceClassifier}
DEFAULT = ["models/legacy_mobilenetv2.pth", "models/int2d-3d-cnn.pth", "models/temporal_pyramid_network.pth"]
OUT = Path("research_plots/metrics.json")


@torch.no_grad()
def evaluate(ckpt_path: str, threshold: float = 0.5) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ARCHS[ckpt.get("model_type", "legacy")]()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Match the clip shape the checkpoint was trained with.
    tv.IMG_SIZE = ckpt.get("img_size", tv.IMG_SIZE)
    tv.FRAMES_PER_CLIP = ckpt.get("frames_per_clip", tv.FRAMES_PER_CLIP)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = tv.ViolenceDataset(tv.load_split(tv.TEST_TXT), transform=tf)
    if len(ds) == 0:
        sys.exit(f"No test clips found under {tv.DATA_DIR}/ — download the Bus Violence dataset first.")

    y_true, y_pred = [], []
    for clips, labels in DataLoader(ds, batch_size=8):
        probs = torch.sigmoid(model(clips))
        y_pred += (probs >= threshold).int().tolist()
        y_true += labels.int().tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "checkpoint": Path(ckpt_path).name,
        "model_type": ckpt.get("model_type", "legacy"),
        "test_clips": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }


def main():
    results = []
    for path in sys.argv[1:] or DEFAULT:
        r = evaluate(path)
        print(f'{r["checkpoint"]:34s} acc={r["accuracy"]:.3f} p={r["precision"]:.3f} '
              f'r={r["recall"]:.3f} f1={r["f1"]:.3f} {r["confusion"]}')
        results.append(r)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
