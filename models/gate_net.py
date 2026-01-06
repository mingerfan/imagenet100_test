"""
自定义 ResNet-18 变体，用于实验不同 block 类型和激活函数
"""

import torch
import torch.nn as nn
from .registry import register_model
from .gate_net_cmp.block_compose import SpecialResNet
from .gate_net_cmp.block_def import (
    Relu,
    LearnableSwish,
    LearnableRelu,
    StablePoly7,
)


class ResNet18Gate(nn.Module):
    """
    ResNet-18 基础类，支持自定义 layer1 的第一个 block
    
    Args:
        first_block_type: 替换 layer1 第一个 block 的类型
        activation: 使用的激活函数类
        num_classes: 分类数量，默认100
        first_block_factor: bottleneck 的扩展因子，默认 0.25
    """
    
    def __init__(self, 
                 first_block_type: str,
                 activation,
                 num_classes: int = 100,
                 first_block_factor: float = 0.25):
        super().__init__()
        
        self.activation = activation
        
        # 初始层: conv7x7 + bn + relu + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建特殊配置：替换 layer1 的第一个 block
        self.special_resnet = self._build_special_resnet(first_block_type, first_block_factor)
        
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _build_special_resnet(self, first_block_type, factor):
        """
        构建 SpecialResNet 配置，替换 layer1 的第一个 block
        
        ResNet-18 标准配置:
        - layer1: 2个 BasicBlock (64→64)
        - layer2: 2个 BasicBlock (64→128, stride=2)
        - layer3: 2个 BasicBlock (128→256, stride=2)
        - layer4: 2个 BasicBlock (256→512, stride=2)
        """
        
        config = [
            # Layer 1: 第一个 block 使用指定类型，第二个保持 basic
            {"block_type": first_block_type, "out_channels": 64, "num_blocks": 1, 
             "activation": self.activation, "factor": factor},
            {"block_type": "basic", "out_channels": 64, "num_blocks": 1, 
             "activation": self.activation},
            # Layer 2
            {"block_type": "basic", "out_channels": 128, "stride": 2, "num_blocks": 2, 
             "activation": Relu},
            # Layer 3
            {"block_type": "basic", "out_channels": 256, "stride": 2, "num_blocks": 2, 
             "activation": Relu},
            # Layer 4
            {"block_type": "basic", "out_channels": 512, "stride": 2, "num_blocks": 2, 
             "activation": Relu},
        ]
        
        return SpecialResNet(config=config, in_channels=64)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 主干网络
        x = self.special_resnet(x)
        
        # 输出层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============ 生成 16 个变体 ============
# Block types: basic, basic_self_gated, bottleneck, bottleneck_self_gated
# Activations: Relu, LearnableSwish, LearnableRelu, StablePoly7

# ===== Basic Block Variants =====
@register_model('resnet-basic-relu-layer1block1')
def resnet_basic_relu(num_classes=100, pretrained=False):
    """
    创建 Basic Block + ReLU 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic', Relu, num_classes, first_block_factor=0.25)


@register_model('resnet-basic-learnableswish-layer1block1')
def resnet_basic_learnableswish(num_classes=100, pretrained=False):
    """
    创建 Basic Block + LearnableSwish 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic', LearnableSwish, num_classes, first_block_factor=0.25)


@register_model('resnet-basic-learnablerelu-layer1block1')
def resnet_basic_learnablerelu(num_classes=100, pretrained=False):
    """
    创建 Basic Block + LearnableRelu 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic', LearnableRelu, num_classes, first_block_factor=0.25)


@register_model('resnet-basic-stablepoly7-layer1block1')
def resnet_basic_stablepoly7(num_classes=100, pretrained=False):
    """
    创建 Basic Block + StablePoly7 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic', StablePoly7, num_classes, first_block_factor=0.25)


# ===== Basic Self-Gated Block Variants =====
@register_model('resnet-basic_self_gated-relu-layer1block1')
def resnet_basic_self_gated_relu(num_classes=100, pretrained=False):
    """
    创建 Basic Self-Gated Block + ReLU 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic_self_gated', Relu, num_classes, first_block_factor=0.25)


@register_model('resnet-basic_self_gated-learnableswish-layer1block1')
def resnet_basic_self_gated_learnableswish(num_classes=100, pretrained=False):
    """
    创建 Basic Self-Gated Block + LearnableSwish 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic_self_gated', LearnableSwish, num_classes, first_block_factor=0.25)


@register_model('resnet-basic_self_gated-learnablerelu-layer1block1')
def resnet_basic_self_gated_learnablerelu(num_classes=100, pretrained=False):
    """
    创建 Basic Self-Gated Block + LearnableRelu 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic_self_gated', LearnableRelu, num_classes, first_block_factor=0.25)


@register_model('resnet-basic_self_gated-stablepoly7-layer1block1')
def resnet_basic_self_gated_stablepoly7(num_classes=100, pretrained=False):
    """
    创建 Basic Self-Gated Block + StablePoly7 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic_self_gated', StablePoly7, num_classes, first_block_factor=0.25)


# ===== Bottleneck Block Variants =====
@register_model('resnet-bottleneck-relu-layer1block1')
def resnet_bottleneck_relu(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Block + ReLU 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck', Relu, num_classes, first_block_factor=0.25)


@register_model('resnet-bottleneck-learnableswish-layer1block1')
def resnet_bottleneck_learnableswish(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Block + LearnableSwish 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck', LearnableSwish, num_classes, first_block_factor=0.25)


@register_model('resnet-bottleneck-learnablerelu-layer1block1')
def resnet_bottleneck_learnablerelu(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Block + LearnableRelu 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck', LearnableRelu, num_classes, first_block_factor=0.25)


@register_model('resnet-bottleneck-stablepoly7-layer1block1')
def resnet_bottleneck_stablepoly7(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Block + StablePoly7 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck', StablePoly7, num_classes, first_block_factor=0.25)


# ===== Bottleneck Self-Gated Block Variants =====
@register_model('resnet-bottleneck_self_gated-relu-layer1block1')
def resnet_bottleneck_self_gated_relu(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Self-Gated Block + ReLU 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck_self_gated', Relu, num_classes, first_block_factor=0.25)


@register_model('resnet-bottleneck_self_gated-learnableswish-layer1block1')
def resnet_bottleneck_self_gated_learnableswish(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Self-Gated Block + LearnableSwish 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck_self_gated', LearnableSwish, num_classes, first_block_factor=0.25)


@register_model('resnet-bottleneck_self_gated-learnablerelu-layer1block1')
def resnet_bottleneck_self_gated_learnablerelu(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Self-Gated Block + LearnableRelu 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck_self_gated', LearnableRelu, num_classes, first_block_factor=0.25)


@register_model('resnet-bottleneck_self_gated-stablepoly7-layer1block1')
def resnet_bottleneck_self_gated_stablepoly7(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Self-Gated Block + StablePoly7 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck_self_gated', StablePoly7, num_classes, first_block_factor=0.25)
