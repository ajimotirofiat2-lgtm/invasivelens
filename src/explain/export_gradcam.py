"""
Export Grad-CAM heatmap overlays for trained checkpoints.

Usage:
    python -m src.explain.export_gradcam --model resnet50 --manifest manifests/combined_manifest.csv
    python -m src.explain.export_gradcam --model baseline --checkpoint checkpoints/baseline_fold0.pt --n-samples 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import CHECKPOINT_DIR, IMAGE_SIZE, RESULTS_DIR, SEED
from src.data.augmentation import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform
from src.data.paths import resolve_manifest_path
from src.explain.gradcam import GradCAM, overlay_heatmap
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model


def load_model(model_name: str, num_classes: int, checkpoint: Path, device: str):
    if model_name == "baseline":
        model = BaselineCNN(num_classes=num_classes, input_size=IMAGE_SIZE)
    else:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def export_samples(
    model_name: str,
    manifest_path: Path,
    checkpoint: Path | None = None,
    out_dir: Path | None = None,
    n_samples: int = 3,
    device: str = "cpu",
    seed: int = SEED,
):
    df = pd.read_csv(manifest_path)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    checkpoint = checkpoint or (CHECKPOINT_DIR / f"{model_name}_fold0.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_model(model_name, len(classes), checkpoint, device)
    cam = GradCAM(model, model_name)
    transform = get_eval_transform(IMAGE_SIZE)

    out_dir = out_dir or (RESULTS_DIR / f"gradcam_{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)

    for i, idx in enumerate(sample_idx):
        row = df.iloc[idx]
        image_path = resolve_manifest_path(row["filepath"], manifest_path)
        pil_img = Image.open(image_path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0).to(device)

        heatmap, pred_idx = cam(tensor, class_idx=None)
        pred_label = classes[pred_idx]
        true_label = row["label"]

        rgb = _denormalize(tensor.squeeze(0))
        overlay = overlay_heatmap(rgb, heatmap)

        stem = image_path.stem
        Image.fromarray(rgb).save(out_dir / f"{i:02d}_{stem}_input.jpg")
        Image.fromarray(overlay).save(out_dir / f"{i:02d}_{stem}_gradcam.jpg")
        print(f"  [{i + 1}/{len(sample_idx)}] true={true_label} pred={pred_label} -> {out_dir.name}/")

    print(f"Grad-CAM overlays saved to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["baseline", "resnet50", "efficientnet_v2_s"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    export_samples(
        model_name=args.model,
        manifest_path=Path(args.manifest),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
