"""
Wraps torchvision's EfficientNetV2-S and ResNet50 with a replaced
classification head for our target classes.

A note on running this in a network-restricted sandbox: `pretrained=True`
downloads ImageNet weights from download.pytorch.org, which is blocked in
this container's egress allowlist (confirmed: HTTP 403). This is NOT a bug
in the code — it WILL work wherever the project actually trains (your
machine, Colab, university compute, anywhere with normal internet access).
The architectures themselves were verified to instantiate and run a
forward pass correctly with `pretrained=False` (random init) in this
sandbox; see tests/test_models.py.
"""
import torch.nn as nn
import torchvision


def build_efficientnet_v2_s(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = (
        torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        if pretrained else None
    )
    model = torchvision.models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = (
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        if pretrained else None
    )
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "efficientnet_v2_s": build_efficientnet_v2_s,
    "resnet50": build_resnet50,
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS)}.")
    return MODEL_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)
