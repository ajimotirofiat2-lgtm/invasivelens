"""
Tests for src/models/. Uses pretrained=False everywhere because this
sandbox's egress allowlist blocks download.pytorch.org (confirmed: HTTP
403) — that restriction is specific to this development container, not a
problem with the code. Run with pretrained=True on your own machine/Colab.

Run with:  pytest tests/test_models.py -v
"""
import torch

from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model

NUM_CLASSES = 6
BATCH = 4
IMG = 224


def test_baseline_cnn_output_shape():
    model = BaselineCNN(num_classes=NUM_CLASSES)
    x = torch.randn(BATCH, 3, IMG, IMG)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)
    print("BaselineCNN output shape:", out.shape)


def test_efficientnet_v2_s_head_replaced():
    model = build_model("efficientnet_v2_s", num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(BATCH, 3, IMG, IMG)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)
    print("EfficientNetV2-S output shape:", out.shape)


def test_resnet50_head_replaced():
    model = build_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(BATCH, 3, IMG, IMG)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)
    print("ResNet50 output shape:", out.shape)


def test_unknown_model_name_raises():
    try:
        build_model("not_a_real_model", num_classes=NUM_CLASSES)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("Unknown model name correctly raises ValueError.")


if __name__ == "__main__":
    test_baseline_cnn_output_shape()
    test_efficientnet_v2_s_head_replaced()
    test_resnet50_head_replaced()
    test_unknown_model_name_raises()
    print("\nAll model tests passed.")
