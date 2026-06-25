"""
Inference module for InvasiveLens — load trained models and make predictions
with confidence scores, uncertainty quantification, and abstention thresholds.

This is the interface for deployment (REST API, mobile apps, batch inference).

Usage:
    from src.inference import InvasiveLensPredictor
    predictor = InvasiveLensPredictor(model_name="resnet50", fold=0, threshold=0.7)
    pred, confidence, abstain = predictor.predict_from_file("path/to/image.jpg")
"""
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import CHECKPOINT_DIR, IMAGE_SIZE, RESULTS_DIR, SEED
from src.data.augmentation import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model


class Prediction(NamedTuple):
    """Result of a single prediction."""
    predicted_class: str
    confidence: float
    abstain: bool  # True if confidence < threshold
    top_3_classes: list[Tuple[str, float]]  # [(class_name, confidence), ...]


class InvasiveLensPredictor:
    """Load a trained model checkpoint and make predictions with calibration."""

    def __init__(
        self,
        model_name: str = "resnet50",
        fold: int = 0,
        threshold: float = 0.5,
        device: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Args:
            model_name: "baseline", "resnet50", or "efficientnet_v2_s"
            fold: which fold's checkpoint to load (0-4)
            threshold: confidence threshold below which to abstain
            device: "cuda" or "cpu"; auto-detects if None
            checkpoint_path: override default checkpoint path
        """
        self.model_name = model_name
        self.fold = fold
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_eval_transform(IMAGE_SIZE)

        # Load class list from metrics
        metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}. "
                f"Train the model first with src/train.py"
            )
        
        import json
        metrics_data = json.loads(metrics_path.read_text())
        self.classes = metrics_data.get("classes", [])
        if not self.classes:
            raise ValueError("Could not load class list from metrics file")

        self.model = self._load_checkpoint(checkpoint_path)
        self.model.to(self.device).eval()

    def _load_checkpoint(self, checkpoint_path: Optional[Path]) -> torch.nn.Module:
        """Load model state dict from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / f"{self.model_name}_fold{self.fold}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Train the model with src/train.py first."
            )

        num_classes = len(self.classes)
        if self.model_name == "baseline":
            model = BaselineCNN(num_classes=num_classes, input_size=IMAGE_SIZE)
        else:
            model = build_model(self.model_name, num_classes=num_classes, pretrained=False)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        return model

    @torch.no_grad()
    def predict_from_file(self, image_path: str | Path) -> Prediction:
        """
        Predict on a single image file.

        Args:
            image_path: path to an image (JPG, PNG, etc.)

        Returns:
            Prediction with class, confidence, abstention flag, and top 3
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.predict_from_tensor(tensor)

    @torch.no_grad()
    def predict_from_tensor(self, tensor: torch.Tensor) -> Prediction:
        """
        Predict on a single image tensor (1, 3, H, W).

        Args:
            tensor: normalized image tensor

        Returns:
            Prediction object
        """
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"Expected shape (1, 3, H, W), got {tensor.shape}")

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        abstain = confidence < self.threshold

        # Top 3
        top_3_idx = np.argsort(probs)[::-1][:3]
        top_3 = [(self.classes[i], float(probs[i])) for i in top_3_idx]

        return Prediction(
            predicted_class=self.classes[pred_idx],
            confidence=confidence,
            abstain=abstain,
            top_3_classes=top_3,
        )

    @torch.no_grad()
    def predict_batch(self, image_paths: list[str | Path]) -> list[Prediction]:
        """Predict on multiple images. Returns list of Prediction objects."""
        return [self.predict_from_file(path) for path in image_paths]

    def get_calibration_info(self) -> dict:
        """
        Return calibration info from fold 0 predictions.
        Used to assess model overconfidence and set threshold appropriately.
        """
        fold0_path = RESULTS_DIR / f"{self.model_name}_fold0_preds.npz"
        if not fold0_path.exists():
            return {"warning": "Fold 0 predictions not found"}

        data = np.load(fold0_path)
        preds = data["preds"]
        labels = data["labels"]
        confidences = data.get("confidences", np.ones_like(preds, dtype=np.float32) * np.nan)

        if np.isnan(confidences).all():
            return {"warning": "No confidence data in fold 0 predictions"}

        correct = preds == labels
        correct_conf = confidences[correct]
        incorrect_conf = confidences[~correct]

        return {
            "mean_confidence_correct": float(np.mean(correct_conf)) if len(correct_conf) > 0 else None,
            "mean_confidence_incorrect": float(np.mean(incorrect_conf)) if len(incorrect_conf) > 0 else None,
            "n_correct": int(np.sum(correct)),
            "n_incorrect": int(np.sum(~correct)),
            "accuracy": float(np.sum(correct) / len(correct)),
            "recommended_threshold": float(
                np.percentile(correct_conf, 25) if len(correct_conf) > 0 else 0.5
            ),
        }


def batch_predict_csv(
    csv_path: str | Path,
    model_name: str = "resnet50",
    fold: int = 0,
    threshold: float = 0.5,
    image_col: str = "filepath",
) -> pd.DataFrame:
    """
    Run inference on a CSV of image paths and return results.

    Args:
        csv_path: Path to CSV with image paths
        model_name, fold, threshold: predictor config
        image_col: column name containing image paths

    Returns:
        DataFrame with predictions, confidence, abstention for each row
    """
    predictor = InvasiveLensPredictor(model_name=model_name, fold=fold, threshold=threshold)
    df = pd.read_csv(csv_path)

    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in {csv_path}")

    results = []
    for idx, row in df.iterrows():
        image_path = row[image_col]
        try:
            pred = predictor.predict_from_file(image_path)
            results.append({
                "index": idx,
                "filepath": image_path,
                "predicted_class": pred.predicted_class,
                "confidence": pred.confidence,
                "abstain": pred.abstain,
                "top_3": str(pred.top_3_classes),
            })
        except Exception as e:
            results.append({
                "index": idx,
                "filepath": image_path,
                "error": str(e),
                "predicted_class": None,
                "confidence": None,
                "abstain": True,
                "top_3": None,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Single image to predict on")
    parser.add_argument("--csv", help="CSV file with image paths for batch prediction")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-calibration", action="store_true")
    args = parser.parse_args()

    predictor = InvasiveLensPredictor(
        model_name=args.model,
        fold=args.fold,
        threshold=args.threshold,
    )

    if args.show_calibration:
        calib = predictor.get_calibration_info()
        print("Calibration Info:")
        for k, v in calib.items():
            print(f"  {k}: {v}")

    if args.image:
        pred = predictor.predict_from_file(args.image)
        print(f"\nPrediction for {args.image}:")
        print(f"  Class: {pred.predicted_class}")
        print(f"  Confidence: {pred.confidence:.3f}")
        print(f"  Abstain: {pred.abstain}")
        print(f"  Top 3: {pred.top_3_classes}")

    if args.csv:
        print(f"Running batch predictions on {args.csv}...")
        results_df = batch_predict_csv(
            args.csv,
            model_name=args.model,
            fold=args.fold,
            threshold=args.threshold,
        )
        out_csv = Path(args.csv).stem + "_predictions.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")
        print(results_df.head())
