"""
EfficientNet model definitions.
"""

import torch.nn as nn
from torchvision import models
from .registry import register_model


@register_model("efficientnet-b0")
def efficientnet_b0(num_classes=100, pretrained=True):
    """Create EfficientNet-B0."""
    model = None
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
    else:
        try:
            model = models.efficientnet_b0(weights=None)
        except Exception:
            model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
