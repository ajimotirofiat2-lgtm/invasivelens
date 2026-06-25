"""
Loads two architectures' fold-0 predictions (saved by src/train.py to
results/<model>_fold0_preds.npz) and runs McNemar's test between them —
this is only valid because both were evaluated on the exact same held-out
rows (fold 0, see src/data/splits.py's mcnemar_split).

Usage:
    python -m src.compare_models --model-a resnet50 --model-b efficientnet_v2_s
"""
import argparse
from pathlib import Path

import numpy as np

from config import RESULTS_DIR
from src.evaluate import mcnemar_test


def load_fold0(model_name: str):
    path = RESULTS_DIR / f"{model_name}_fold0_preds.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/train.py --model {model_name} first.")
    data = np.load(path)
    preds = data["preds"]
    labels = data["labels"]
    idx = data["val_idx"]
    confidences = data.get("confidences", np.ones_like(preds, dtype=np.float32) * np.nan)
    return preds, labels, idx, confidences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    args = parser.parse_args()

    preds_a, labels_a, idx_a, conf_a = load_fold0(args.model_a)
    preds_b, labels_b, idx_b, conf_b = load_fold0(args.model_b)

    if not np.array_equal(idx_a, idx_b):
        raise ValueError(
            "The two models were evaluated on different fold-0 row indices — "
            "they must be trained on the SAME manifest/seed/n_folds for "
            "McNemar's test to be valid. Re-run src/train.py for both with "
            "identical --manifest and config.SEED/N_FOLDS."
        )

    result = mcnemar_test(labels_a, preds_a, preds_b)
    print(f"\nComparing {args.model_a} vs {args.model_b} on {len(idx_a)} held-out rows (fold 0):")
    print(f"  contingency table: {result['contingency_table']}")
    print(f"  statistic={result['statistic']:.4f}  p_value={result['p_value']:.4f}")
    print(f"  {result['interpretation']}")
    
    # Confidence analysis
    print(f"\nConfidence Analysis:")
    print(f"  {args.model_a}: mean={np.nanmean(conf_a):.3f}  std={np.nanstd(conf_a):.3f}")
    print(f"  {args.model_b}: mean={np.nanmean(conf_b):.3f}  std={np.nanstd(conf_b):.3f}")
    
    # Identify cases where one model is more confident
    conf_diff = conf_b - conf_a
    more_confident_b = (conf_diff > 0.1).sum()
    more_confident_a = (conf_diff < -0.1).sum()
    print(f"\n  Cases where {args.model_b} is >10% more confident: {more_confident_b}")
    print(f"  Cases where {args.model_a} is >10% more confident: {more_confident_a}")


if __name__ == "__main__":
    main()
