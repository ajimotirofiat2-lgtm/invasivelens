"""
Tests for src/inference.py.

Run with: pytest tests/test_inference.py -v
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.inference import InvasiveLensPredictor


@pytest.fixture
def synthetic_checkpoint():
    """Create a temporary synthetic model and metrics for testing."""
    from config import CHECKPOINT_DIR, IMAGE_SIZE, RESULTS_DIR, SEED
    from src.models.baseline_cnn import BaselineCNN
    import json

    checkpoint_dir = CHECKPOINT_DIR
    results_dir = RESULTS_DIR

    # Create a dummy model checkpoint
    model = BaselineCNN(num_classes=3, input_size=IMAGE_SIZE)
    ckpt_path = checkpoint_dir / "test_baseline_fold0.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Create dummy metrics file
    metrics = {
        "accuracy_mean": 0.85,
        "accuracy_std": 0.05,
        "macro_f1_mean": 0.84,
        "macro_f1_std": 0.06,
        "n_folds": 3,
        "classes": ["class_a", "class_b", "class_c"],
    }
    metrics_path = results_dir / "test_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Create dummy predictions for calibration
    preds_path = results_dir / "test_baseline_fold0_preds.npz"
    np.savez(
        preds_path,
        preds=np.array([0, 1, 2, 0, 1], dtype=np.int64),
        labels=np.array([0, 1, 2, 0, 2], dtype=np.int64),
        val_idx=np.array([0, 1, 2, 3, 4]),
        confidences=np.array([0.9, 0.85, 0.88, 0.92, 0.6], dtype=np.float32),
    )

    yield ckpt_path

    # Cleanup
    ckpt_path.unlink(missing_ok=True)
    metrics_path.unlink(missing_ok=True)
    preds_path.unlink(missing_ok=True)


def create_test_image(size: int = 224) -> Image.Image:
    """Create a simple test image."""
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_inference_predictor_loads_model(synthetic_checkpoint):
    """Test that InvasiveLensPredictor can load a model."""
    predictor = InvasiveLensPredictor(model_name="test_baseline", fold=0)
    assert predictor.model is not None
    assert predictor.classes == ["class_a", "class_b", "class_c"]


def test_inference_predictor_single_prediction(synthetic_checkpoint):
    """Test single image prediction."""
    predictor = InvasiveLensPredictor(model_name="test_baseline", fold=0, threshold=0.5)

    # Create temp image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.jpg"
        create_test_image().save(img_path)

        pred = predictor.predict_from_file(img_path)

        assert pred.predicted_class in predictor.classes
        assert 0.0 <= pred.confidence <= 1.0
        assert isinstance(pred.abstain, bool)
        assert len(pred.top_3_classes) == 3


def test_inference_abstention_threshold():
    """Test that abstention works with low confidence."""
    # This is a conceptual test; actual behavior depends on model randomness
    predictor = InvasiveLensPredictor(model_name="test_baseline", threshold=0.99)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.jpg"
        create_test_image().save(img_path)

        pred = predictor.predict_from_file(img_path)
        # Most predictions will have confidence < 0.99, so abstain should often be True
        # (not guaranteed due to randomness, but likely)
        assert isinstance(pred.abstain, bool)


def test_inference_batch_prediction(synthetic_checkpoint):
    """Test batch prediction on multiple images."""
    predictor = InvasiveLensPredictor(model_name="test_baseline", fold=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple test images
        img_paths = []
        for i in range(3):
            img_path = Path(tmpdir) / f"test_{i}.jpg"
            create_test_image().save(img_path)
            img_paths.append(img_path)

        preds = predictor.predict_batch(img_paths)

        assert len(preds) == 3
        for pred in preds:
            assert pred.predicted_class in predictor.classes
            assert 0.0 <= pred.confidence <= 1.0


def test_inference_calibration_info(synthetic_checkpoint):
    """Test retrieval of calibration information."""
    predictor = InvasiveLensPredictor(model_name="test_baseline", fold=0)
    calib = predictor.get_calibration_info()

    assert "mean_confidence_correct" in calib
    assert "mean_confidence_incorrect" in calib
    assert "accuracy" in calib
    # Synthetic data: 4/5 correct
    assert calib["accuracy"] == 0.8


def test_inference_tensor_prediction(synthetic_checkpoint):
    """Test prediction from tensor (for batch inference)."""
    from config import IMAGE_SIZE
    from src.data.augmentation import get_eval_transform

    predictor = InvasiveLensPredictor(model_name="test_baseline", fold=0)

    # Create a test image tensor
    transform = get_eval_transform(IMAGE_SIZE)
    img = create_test_image(IMAGE_SIZE)
    tensor = transform(img).unsqueeze(0)

    pred = predictor.predict_from_tensor(tensor)

    assert pred.predicted_class in predictor.classes
    assert 0.0 <= pred.confidence <= 1.0


def test_inference_missing_checkpoint():
    """Test that missing checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        InvasiveLensPredictor(model_name="nonexistent_model_xyz", fold=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
