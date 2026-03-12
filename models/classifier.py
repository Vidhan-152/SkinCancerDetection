"""
Refined MobileNetV2-based skin lesion classifier.
Binary classification: Benign (0) vs Malignant (1)

Improvements:
- Gradual unfreezing strategy
- Better classifier head with dropout
- Label smoothing support
"""

import torch
import torch.nn as nn
from torchvision import models


class SkinClassifier(nn.Module):
    """
    MobileNetV2 fine-tuned for binary skin cancer classification.
    Output: single logit (apply sigmoid → probability of malignant)
    """
    def __init__(self, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v2(weights=weights)

        # Freeze all backbone layers initially
        for param in base.features.parameters():
            param.requires_grad = False

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Richer classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def unfreeze_top_n(self, n=3):
        """Unfreeze last n feature blocks for fine-tuning."""
        blocks = list(self.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
