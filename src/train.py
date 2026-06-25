"""
Trains one architecture (baseline, resnet50, or efficientnet_v2_s) across
all geographic CV folds from src/data/splits.py, saving:
  - per-fold metrics + their average (results/<model>_metrics.json)
  - a checkpoint per fold (checkpoints/<model>_fold<i>.pt)
  - fold 0's raw predictions + labels (results/<model>_fold0_preds.npz) —
    this is the designated McNemar's comparison set (see splits.py); run
    src/compare_models.py afterwards to compare two architectures' fold-0
    predictions on the exact same held-out rows.

Usage:
    python -m src.train --manifest manifests/combined_manifest.csv --model resnet50 --epochs 15
    python -m src.train --manifest manifests/combined_manifest.csv --model baseline --epochs 30 --no-pretrained
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINT_DIR, IMAGE_SIZE, LEARNING_RATE, N_FOLDS, NUM_WORKERS, RESULTS_DIR, SEED
from src.data.augmentation import get_eval_transform, get_train_transform
from src.data.dataset import ManifestDataset
from src.data.splits import make_splits
from src.evaluate import average_metrics_over_folds, compute_metrics
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model


def build_model_for_run(model_name: str, num_classes: int, pretrained: bool):
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes, input_size=IMAGE_SIZE)
    return build_model(model_name, num_classes=num_classes, pretrained=pretrained)


def train_one_fold(model, train_loader, device, epochs: int, lr: float = 1e-4):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * images.size(0)
        print(f"  epoch {epoch + 1}/{epochs}  loss={running_loss / len(train_loader.dataset):.4f}")
    return model


@torch.no_grad()
def predict(model, loader, device):
    """Returns predictions, labels, confidences, and logits for uncertainty quantification."""
    model.to(device).eval()
    all_preds, all_labels, all_confidences, all_logits = [], [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        confidences = probs.max(dim=1)[0].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_confidences.append(confidences)
        all_logits.append(logits.cpu().numpy())
    
    if not all_preds:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), 
                np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    
    return (np.concatenate(all_preds), np.concatenate(all_labels),
            np.concatenate(all_confidences), np.concatenate(all_logits))


def _choose_n_folds(df: pd.DataFrame, group_col: str, max_folds: int) -> int:
    """Pick the largest fold count that yields no empty validation splits."""
    n_groups = df[group_col].nunique()
    upper = min(max_folds, len(df), n_groups)
    for n_folds in range(upper, 1, -1):
        result = make_splits(df, group_col=group_col, n_folds=n_folds, seed=SEED)
        if all(size > 0 for size in result["fold_sizes"]):
            return n_folds
    raise ValueError(
        f"Dataset too small for grouped CV: {len(df)} rows, {n_groups} groups."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True,
                         choices=["baseline", "resnet50", "efficientnet_v2_s"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for Adam optimizer (default from config)")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    df = pd.read_csv(args.manifest)
    if "group" not in df.columns:
        raise ValueError(
            "Manifest needs a 'group' column — run src/data/merge_manifests.py first."
        )

    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} classes: {classes}")

    n_folds = _choose_n_folds(df, group_col="group", max_folds=N_FOLDS)
    if n_folds != N_FOLDS:
        print(f"Reduced folds {N_FOLDS} -> {n_folds} to avoid empty validation splits.")
    split_result = make_splits(
        df, label_col="label", group_col="group", n_folds=n_folds, seed=SEED,
    )
    print(f"Fold sizes: {split_result['fold_sizes']}")

    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(split_result["cv_folds"]):
        print(f"\n=== Fold {fold_i} ===")
        train_df, val_df = df.loc[train_idx], df.loc[val_idx]

        train_ds = ManifestDataset(train_df, transform=get_train_transform(IMAGE_SIZE), class_to_idx=class_to_idx)
        val_ds = ManifestDataset(val_df, transform=get_eval_transform(IMAGE_SIZE), class_to_idx=class_to_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

        if len(val_df) == 0:
            print(f"  fold {fold_i}: empty validation set — skipped")
            continue

        model = build_model_for_run(args.model, num_classes=len(classes), pretrained=args.pretrained)
        model = train_one_fold(model, train_loader, args.device, epochs=args.epochs, lr=args.learning_rate)

        preds, labels, confidences, logits = predict(model, val_loader, args.device)
        metrics = compute_metrics(labels, preds, class_names=classes)
        print(f"  fold {fold_i}: accuracy={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}  "
              f"mean_confidence={confidences.mean():.3f}")
        fold_metrics.append(metrics)

        ckpt_path = CHECKPOINT_DIR / f"{args.model}_fold{fold_i}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if fold_i == split_result["mcnemar_fold_index"]:
            np.savez(
                RESULTS_DIR / f"{args.model}_fold0_preds.npz",
                preds=preds, labels=labels, val_idx=val_idx,
                confidences=confidences, logits=logits,
            )

    if not fold_metrics:
        raise RuntimeError("No folds produced validation metrics — dataset may be too small.")

    summary = average_metrics_over_folds(fold_metrics)
    summary["per_fold"] = fold_metrics
    summary["classes"] = classes
    out_path = RESULTS_DIR / f"{args.model}_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAveraged over {summary['n_folds']} folds: "
          f"accuracy={summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f}  "
          f"macro_f1={summary['macro_f1_mean']:.3f}±{summary['macro_f1_std']:.3f}")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
