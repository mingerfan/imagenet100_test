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
    StablePoly4,
    Swish,
    BN_EPS,
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
    
    def __init__(
        self,
        first_block_type: str,
        activation,
        num_classes: int = 100,
        first_block_factor: float = 0.25,
        stem_activation=None,
        stem_pool: str = "max",
        full_activation: bool = False,
        use_pre_fc_bn: bool = False,
    ):
        super().__init__()
        
        self.activation = activation
        self.stem_activation = stem_activation or Relu
        self.stem_pool = stem_pool
        self.full_activation = full_activation
        self.use_pre_fc_bn = use_pre_fc_bn
        
        # 初始层: conv7x7 + bn + relu + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stem_act = self.stem_activation()
        if self.stem_pool == "max":
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.stem_pool == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif self.stem_pool == "none":
            self.pool = nn.Identity()
        else:
            raise ValueError(f"Unknown stem_pool: {self.stem_pool}")
        
        # 构建特殊配置：替换 layer1 的第一个 block
        self.special_resnet = self._build_special_resnet(first_block_type, first_block_factor)
        
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_fc_bn = nn.BatchNorm1d(512, eps=BN_EPS) if self.use_pre_fc_bn else None
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
        
        later_activation = self.activation if self.full_activation else Relu
        config = [
            # Layer 1: 第一个 block 使用指定类型，第二个保持 basic
            {"block_type": first_block_type, "out_channels": 64, "num_blocks": 1, 
             "activation": self.activation, "factor": factor},
            {"block_type": "basic", "out_channels": 64, "num_blocks": 1, 
             "activation": self.activation},
            # Layer 2
            {"block_type": "basic", "out_channels": 128, "stride": 2, "num_blocks": 2, 
             "activation": later_activation},
            # Layer 3
            {"block_type": "basic", "out_channels": 256, "stride": 2, "num_blocks": 2, 
             "activation": later_activation},
            # Layer 4
            {"block_type": "basic", "out_channels": 512, "stride": 2, "num_blocks": 2, 
             "activation": later_activation},
        ]
        
        return SpecialResNet(config=config, in_channels=64)
    
    def _initialize_weights(self):
        """初始化网络权重
        
        针对 StablePoly4/SiLU 激活函数优化的初始化策略：
        - 使用 fan_in 模式 + leaky_relu 假设，产生更保守的初始权重
        - 降低 BN 的初始 gamma，防止特征值在网络深处膨胀
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.stem_act(x)
        x = self.pool(x)
        
        # 主干网络
        x = self.special_resnet(x)
        
        # 输出层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.pre_fc_bn is not None:
            x = self.pre_fc_bn(x)
        x = self.fc(x)
        
        return x


# ============ 生成 16 个变体 ============
# Block types: basic, basic_self_gated, bottleneck, bottleneck_self_gated
# Activations: Relu, LearnableSwish, LearnableRelu, StablePoly4

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

@register_model('resnet-basic-swish-layer1block1')
def resnet_basic_swish(num_classes=100, pretrained=False):
    return ResNet18Gate(
        'basic',
        Swish,
        num_classes,
        first_block_factor=0.25,
        stem_activation=Swish,
        stem_pool="avg",
        full_activation=True,
        use_pre_fc_bn=True,
    )


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


@register_model('resnet-basic-stablepoly4-layer1block1')
def resnet_basic_stablepoly4(num_classes=100, pretrained=False):
    """
    创建 Basic Block + StablePoly4 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic', StablePoly4, num_classes, first_block_factor=0.25)


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

@register_model('resnet-basic_self_gated-swish-layer1block1')
def resnet_basic_self_gated_swish(num_classes=100, pretrained=False):
    return ResNet18Gate('basic_self_gated', Swish, num_classes, first_block_factor=0.25)


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


@register_model('resnet-basic_self_gated-stablepoly4-layer1block1')
def resnet_basic_self_gated_stablepoly4(num_classes=100, pretrained=False):
    """
    创建 Basic Self-Gated Block + StablePoly4 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('basic_self_gated', StablePoly4, num_classes, first_block_factor=0.25)


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


@register_model('resnet-bottleneck-stablepoly4-layer1block1')
def resnet_bottleneck_stablepoly4(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Block + StablePoly4 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck', StablePoly4, num_classes, first_block_factor=0.25)


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


@register_model('resnet-bottleneck_self_gated-stablepoly4-layer1block1')
def resnet_bottleneck_self_gated_stablepoly4(num_classes=100, pretrained=False):
    """
    创建 Bottleneck Self-Gated Block + StablePoly4 的 ResNet 变体
    
    Args:
        num_classes: 类别数量
        pretrained: 预训练权重（自定义模型不支持此参数，保留用于兼容性）
    
    Returns:
        model: ResNet模型
    """
    return ResNet18Gate('bottleneck_self_gated', StablePoly4, num_classes, first_block_factor=0.25)
