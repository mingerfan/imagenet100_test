"""
ResNet-32 (CIFAR-style) 模型定义
"""

from .registry import register_model
from .resnet20 import BasicBlock, ResNetCifar


@register_model("resnet32")
def resnet32(num_classes: int = 100, pretrained: bool = False):
    """
    CIFAR-style ResNet-32

    Args:
        num_classes: 类别数量，默认为100
        pretrained: 预训练权重（不提供，参数仅占位）
    """
    _ = pretrained
    return ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes)

