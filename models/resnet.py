"""
ResNet模型定义
"""

import torch
import torch.nn as nn
from torchvision import models
from .registry import register_model


@register_model('resnet18')
def resnet18(num_classes=100, pretrained=True):
    """
    创建ResNet18模型
    
    Args:
        num_classes: 类别数量，默认为100
        pretrained: 是否使用预训练权重，默认为True
    
    Returns:
        model: ResNet18模型
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # 修改最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


@register_model('resnet34')
def resnet34(num_classes=100, pretrained=True):
    """
    创建ResNet34模型
    
    Args:
        num_classes: 类别数量，默认为100
        pretrained: 是否使用预训练权重，默认为True
    
    Returns:
        model: ResNet34模型
    """
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # 修改最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


@register_model('resnet50')
def resnet50(num_classes=100, pretrained=True):
    """
    创建ResNet50模型
    
    Args:
        num_classes: 类别数量，默认为100
        pretrained: 是否使用预训练权重，默认为True
    
    Returns:
        model: ResNet50模型
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # 修改最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: 模型
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params