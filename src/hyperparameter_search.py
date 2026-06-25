"""
Hyperparameter tuning for InvasiveLens models.

Provides grid search and random search for optimal hyperparameter selection.

Usage:
    python -m src.hyperparameter_search --manifest combined_manifest.csv --model resnet50
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import CHECKPOINT_DIR, HYPERPARAMETER_GRID, RESULTS_DIR, SEED
from src.data.augmentation import get_eval_transform, get_train_transform
from src.data.dataset import ManifestDataset
from src.data.splits import make_splits
from src.evaluate import compute_metrics
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model
from src.train import train_one_fold
from torch.utils.data import DataLoader


class HyperparameterSearch:
    """Base class for hyperparameter search strategies."""

    def __init__(
        self,
        manifest_path: str | Path,
        model_name: str,
        base_params: Dict[str, Any],
        search_space: Dict[str, List[Any]],
        n_folds: int = 3,  # Use fewer folds for faster tuning
        seed: int = SEED,
    ):
        """
        Args:
            manifest_path: Path to training manifest
            model_name: "baseline", "resnet50", or "efficientnet_v2_s"
            base_params: Default parameter values
            search_space: Dict mapping param names to lists of values to search
            n_folds: Number of CV folds for evaluation
            seed: Random seed for reproducibility
        """
        self.manifest_path = Path(manifest_path)
        self.model_name = model_name
        self.base_params = base_params.copy()
        self.search_space = search_space
        self.n_folds = n_folds
        self.seed = seed

        self.df = pd.read_csv(self.manifest_path)
        if "group" not in self.df.columns:
            raise ValueError("Manifest must have 'group' column for spatial CV")

        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.results = []

    def _evaluate_params(self, params: Dict[str, Any]) -> dict:
        """Evaluate a parameter combination via cross-validation."""
        print(f"\nEvaluating: {params}")

        split_result = make_splits(
            self.df,
            label_col="label",
            group_col="group",
            n_folds=self.n_folds,
            seed=self.seed,
        )

        fold_metrics = []
        for fold_i, (train_idx, val_idx) in enumerate(split_result["cv_folds"]):
            train_df, val_df = self.df.loc[train_idx], self.df.loc[val_idx]

            # Build dataloaders
            from config import IMAGE_SIZE, NUM_WORKERS
            train_ds = ManifestDataset(
                train_df, transform=get_train_transform(IMAGE_SIZE), class_to_idx=self.class_to_idx
            )
            val_ds = ManifestDataset(
                val_df, transform=get_eval_transform(IMAGE_SIZE), class_to_idx=self.class_to_idx
            )

            train_loader = DataLoader(
                train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=NUM_WORKERS
            )
            val_loader = DataLoader(
                val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=NUM_WORKERS
            )

            # Build and train model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model_name == "baseline":
                model = BaselineCNN(num_classes=len(self.classes))
            else:
                model = build_model(self.model_name, num_classes=len(self.classes), pretrained=False)

            model = train_one_fold(
                model,
                train_loader,
                device,
                epochs=params.get("epochs", 5),  # Tune fewer epochs for speed
                lr=params["learning_rate"],
            )

            # Evaluate
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    logits = model(images.to(device))
                    all_preds.append(logits.argmax(dim=1).cpu().numpy())
                    all_labels.append(labels.numpy())

            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            metrics = compute_metrics(labels, preds, class_names=self.classes)
            fold_metrics.append(metrics)

            print(f"  Fold {fold_i}: accuracy={metrics['accuracy']:.3f}  f1={metrics['macro_f1']:.3f}")

        # Average metrics
        avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
        avg_f1 = np.mean([m["macro_f1"] for m in fold_metrics])

        return {
            "params": params,
            "accuracy": avg_acc,
            "macro_f1": avg_f1,
            "per_fold_metrics": fold_metrics,
        }

    def search(self, n_trials: Optional[int] = None) -> List[dict]:
        """
        Run hyperparameter search.

        Args:
            n_trials: Number of trials (for random search); None for grid search

        Returns:
            List of results sorted by primary metric (macro_f1)
        """
        raise NotImplementedError

    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save search results to JSON."""
        output_path = output_path or (RESULTS_DIR / f"{self.model_name}_hyperparam_search.json")
        output_path.write_text(json.dumps(self.results, indent=2, default=str))
        print(f"\nSaved {len(self.results)} results to {output_path}")
        return output_path


class GridSearch(HyperparameterSearch):
    """Exhaustive grid search over hyperparameter space."""

    def search(self, n_trials: Optional[int] = None) -> List[dict]:
        """
        Run grid search over all combinations.

        n_trials is ignored for grid search.
        """
        # Generate all combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        combinations = list(itertools.product(*param_values))

        print(f"Grid search: {len(combinations)} combinations")

        for params_tuple in combinations:
            params = self.base_params.copy()
            params.update({name: value for name, value in zip(param_names, params_tuple)})

            result = self._evaluate_params(params)
            self.results.append(result)

        # Sort by macro_f1
        self.results.sort(key=lambda x: x["macro_f1"], reverse=True)
        return self.results


class RandomSearch(HyperparameterSearch):
    """Random search over hyperparameter space."""

    def search(self, n_trials: int = 10) -> List[dict]:
        """
        Run random search with n_trials random samples.
        """
        rng = np.random.RandomState(self.seed)
        param_names = list(self.search_space.keys())

        print(f"Random search: {n_trials} trials")

        for trial in range(n_trials):
            params = self.base_params.copy()
            for name in param_names:
                values = self.search_space[name]
                params[name] = values[rng.randint(0, len(values))]

            result = self._evaluate_params(params)
            self.results.append(result)

        # Sort by macro_f1
        self.results.sort(key=lambda x: x["macro_f1"], reverse=True)
        return self.results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True,
                        choices=["baseline", "resnet50", "efficientnet_v2_s"])
    parser.add_argument("--strategy", default="grid",
                        choices=["grid", "random"])
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of random trials (ignored for grid search)")
    parser.add_argument("--n-folds", type=int, default=3,
                        help="Number of CV folds for evaluation (use fewer for speed)")
    args = parser.parse_args()

    base_params = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 5,
    }

    if args.strategy == "grid":
        search = GridSearch(
            args.manifest,
            args.model,
            base_params,
            HYPERPARAMETER_GRID,
            n_folds=args.n_folds,
        )
        search.search()
    else:  # random
        search = RandomSearch(
            args.manifest,
            args.model,
            base_params,
            HYPERPARAMETER_GRID,
            n_folds=args.n_folds,
        )
        search.search(n_trials=args.n_trials)

    # Print top results
    print("\n" + "=" * 60)
    print("Top 5 results:")
    for i, result in enumerate(search.results[:5]):
        print(f"{i + 1}. Accuracy: {result['accuracy']:.3f}  F1: {result['macro_f1']:.3f}")
        print(f"   Params: {result['params']}")

    # Save results
    search.save_results()


if __name__ == "__main__":
    main()
