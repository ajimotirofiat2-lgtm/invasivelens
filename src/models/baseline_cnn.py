"""
A deliberately simple CNN trained from scratch — this exists purely to
establish a performance floor that the transfer-learning models (see
transfer_models.py) need to beat in order to justify their extra
complexity and training cost.
"""
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
