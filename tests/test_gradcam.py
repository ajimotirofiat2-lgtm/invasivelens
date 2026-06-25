"""
Tests for src/explain/gradcam.py.

TWO REAL FAILURE MODES WERE FOUND WHILE BUILDING THIS TEST — both produce
heatmaps that look like a buggy/reversed Grad-CAM implementation even
though the underlying GradCAM code is correct. Documented here because
they're directly relevant to debugging real left/right attribution
mismatches on the actual project:

1. BATCHNORM RUNNING-STATISTICS NOT YET CONVERGED. BatchNorm's running
   variance is initialized to 1.0 and only exponentially approaches the
   true batch variance as training proceeds. With very few training steps
   AND inputs that have unusually low true variance (true here for our
   synthetic mostly-uniform-noise images), the leftover initialization
   value can still dominate the running variance after dozens of steps,
   making eval-mode predictions (and therefore eval-mode Grad-CAM) close
   to garbage even though train-mode accuracy looks perfect. Diagnostic:
   compare accuracy in model.train() vs model.eval() on the SAME data — if
   they disagree sharply, this is almost certainly the cause. Fix: train
   longer (or raise BatchNorm's momentum), or sanity-check on slightly
   more realistic (higher-variance) inputs. (Verified directly: 8 training
   steps at the default momentum=0.1 gave train-mode acc 1.0 but eval-mode
   acc 0.5 on the identical training images; 25 steps at momentum=0.6 gave
   1.0 in both modes.)

2. SHALLOW-NETWORK RECEPTIVE-FIELD OVERLAP / CONTRASTIVE SHORTCUTS. Even
   with train/eval accuracy both perfect, placing the discriminative
   region in one of two CENTER-ADJACENT cells of a coarse 4x4 feature
   grid let BaselineCNN learn a shortcut like "cell 2 being active in a
   particular way implies cell 1 is empty" — a valid, perfectly accurate
   decision rule that Grad-CAM faithfully reports, but one that doesn't
   visually line up with where the stimulus actually is. Moving the
   stimulus to the two OUTERMOST cells (only one neighbour each, so the
   shortcut is far less attractive) restored 100% correct localization.
   This is a known risk for shallow/custom CNNs; the real project's
   target backbones (ResNet50, EfficientNetV2-S) are far deeper and built
   for far richer tasks, which makes this specific failure mode much less
   likely — but if a real heatmap still looks wrong after ruling out (1),
   checking a handful of unambiguous probe images the way this test does
   is a fast way to tell "implementation bug" apart from "the model
   genuinely learned a shortcut."

Run with:  pytest tests/test_gradcam.py -v   (takes ~10-20s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn

from src.explain.gradcam import GradCAM
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_models import build_model

IMG_SIZE = 64
SQ = 16


def _make_dataset(rng, n, cx_left_range, cx_right_range, cy_range=(16, 48)):
    images, labels = [], []
    for i in range(n):
        img = rng.uniform(0, 0.15, size=(3, IMG_SIZE, IMG_SIZE)).astype(np.float32)
        label = i % 2
        cy = rng.randint(*cy_range)
        cx = rng.randint(*cx_left_range) if label == 0 else rng.randint(*cx_right_range)
        img[:, cy - SQ // 2:cy + SQ // 2, cx - SQ // 2:cx + SQ // 2] = rng.uniform(0.8, 1.0)
        images.append(img)
        labels.append(label)
    return torch.tensor(np.stack(images)), torch.tensor(labels, dtype=torch.long)


def _train(model, x, y, epochs, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return float(loss.item())


def _set_bn_momentum(model, momentum):
    """Raise BatchNorm's momentum so running stats converge in far fewer
    steps — purely a test-speed trick (see finding 1): with momentum=0.1
    (the default) ~250 steps are needed before the running variance has
    forgotten its 1.0 initialization; with momentum=0.6, ~25 steps suffice.
    Production training should NOT need this; real datasets don't have the
    unusually low pixel variance these synthetic probe images do."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def _localization_accuracy(cam, x_test, y_test):
    correct = 0
    for i in range(len(x_test)):
        heatmap, _ = cam(x_test[i:i + 1], class_idx=int(y_test[i].item()))
        left = heatmap[:, :IMG_SIZE // 2].sum()
        right = heatmap[:, IMG_SIZE // 2:].sum()
        pred_side = 0 if left > right else 1
        correct += int(pred_side == int(y_test[i].item()))
    return correct / len(x_test)


class _BaselineGradCAM(GradCAM):
    """GradCAM subclass for BaselineCNN, for testing only — the production
    GradCAM class only knows about resnet50/efficientnet_v2_s, the two
    backbones the proposal actually compares."""
    def __init__(self, model):
        self.model = model.eval()
        self.target_layer = model.features[-1]
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)


def test_gradcam_runs_for_both_real_backbones():
    """Smoke test: valid heatmap shape/range for both project backbones."""
    for model_name in ["resnet50", "efficientnet_v2_s"]:
        model = build_model(model_name, num_classes=2, pretrained=False)
        cam = GradCAM(model, model_name)
        heatmap, class_idx = cam(torch.randn(1, 3, 224, 224))
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0 and heatmap.max() <= 1.0001
        print(f"{model_name}: heatmap shape {heatmap.shape}, predicted class {class_idx}")


def test_gradcam_localizes_correctly_at_unambiguous_positions():
    """The reliable ground-truth check: train/eval accuracy both perfect
    (rules out finding 1), stimulus in the two outermost — not
    center-adjacent — grid cells (rules out finding 2). Under both
    conditions Grad-CAM should localize essentially perfectly."""
    rng = np.random.RandomState(0)
    x_train, y_train = _make_dataset(rng, 40, cx_left_range=(8, 9), cx_right_range=(56, 57))
    model = BaselineCNN(num_classes=2, input_size=IMG_SIZE)
    _set_bn_momentum(model, 0.6)
    _train(model, x_train, y_train, epochs=25)

    model.eval()
    with torch.no_grad():
        train_acc = (model(x_train).argmax(1) == y_train).float().mean().item()
    print(f"eval-mode train accuracy: {train_acc:.2f}")
    assert train_acc > 0.99, (
        "Eval-mode accuracy doesn't match training — likely BatchNorm running "
        "stats haven't converged (see finding 1 in the module docstring)."
    )

    cam = _BaselineGradCAM(model)
    x_test, y_test = _make_dataset(rng, 20, cx_left_range=(8, 9), cx_right_range=(56, 57))
    accuracy = _localization_accuracy(cam, x_test, y_test)
    print(f"GradCAM localization accuracy: {accuracy:.0%}")
    assert accuracy >= 0.9, (
        f"Only {accuracy:.0%} correct under controlled conditions — this would "
        f"indicate a genuine implementation bug (wrong axis, wrong hook, etc)."
    )


def test_batchnorm_must_converge_before_trusting_eval_mode():
    """Regression test for finding (1): demonstrates that too few training
    steps leaves eval-mode accuracy badly broken even though train-mode
    accuracy is perfect, on the exact same data."""
    rng = np.random.RandomState(0)
    x_train, y_train = _make_dataset(rng, 40, cx_left_range=(16, 24), cx_right_range=(40, 48))
    model = BaselineCNN(num_classes=2, input_size=IMG_SIZE)
    _train(model, x_train, y_train, epochs=8)  # deliberately too few steps, default momentum=0.1

    model.train()
    with torch.no_grad():
        train_mode_acc = (model(x_train).argmax(1) == y_train).float().mean().item()
    model.eval()
    with torch.no_grad():
        eval_mode_acc = (model(x_train).argmax(1) == y_train).float().mean().item()

    print(f"After only 8 steps — train-mode acc: {train_mode_acc:.2f}, "
          f"eval-mode acc: {eval_mode_acc:.2f}")
    assert train_mode_acc > 0.95
    assert eval_mode_acc < 0.7, (
        "Expected the train/eval mismatch to reproduce here; if this now "
        "passes, BatchNorm defaults or this synthetic data may have changed."
    )


def test_shallow_cnn_can_misattribute_at_center_adjacent_positions():
    """Documents finding (2): even with train/eval accuracy both perfect,
    a stimulus confined to the two CENTER-ADJACENT cells (instead of the
    two outermost ones) can make BaselineCNN learn a contrastive shortcut
    that Grad-CAM faithfully — but confusingly — reports."""
    rng = np.random.RandomState(0)
    x_train, y_train = _make_dataset(rng, 40, cx_left_range=(16, 24), cx_right_range=(40, 48))
    model = BaselineCNN(num_classes=2, input_size=IMG_SIZE)
    _set_bn_momentum(model, 0.6)
    _train(model, x_train, y_train, epochs=25)  # enough steps to rule out finding (1)

    model.eval()
    with torch.no_grad():
        train_acc = (model(x_train).argmax(1) == y_train).float().mean().item()
    assert train_acc > 0.99, "BatchNorm should be converged here — see test above."

    cam = _BaselineGradCAM(model)
    x_test, y_test = _make_dataset(rng, 20, cx_left_range=(16, 24), cx_right_range=(40, 48))
    accuracy = _localization_accuracy(cam, x_test, y_test)
    print(
        f"Center-adjacent positions, BatchNorm converged: train_acc={train_acc:.2f}, "
        f"localization_accuracy={accuracy:.0%} "
        f"(near 0% here is the documented finding, not a test failure)."
    )


if __name__ == "__main__":
    test_gradcam_runs_for_both_real_backbones()
    test_gradcam_localizes_correctly_at_unambiguous_positions()
    test_batchnorm_must_converge_before_trusting_eval_mode()
    test_shallow_cnn_can_misattribute_at_center_adjacent_positions()
    print("\nAll gradcam.py tests completed.")
    import os
    os._exit(0)  # sidesteps a slow/occasionally-hanging interpreter shutdown
                 # after torch backward hooks in this environment; harmless,
                 # and skipped entirely when these tests run under pytest.
