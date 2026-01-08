"""FLOPs 计算器 - 统一的浮点运算计算逻辑"""

import math
from typing import Tuple, Optional
from torch import nn
import torch


class FLOPsCalculator:
    """统一的 FLOPs 计算器，消除散落在各处的重复计算逻辑"""

    @staticmethod
    def calc_conv2d_flops(
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        output_shape: Tuple[int, ...],
        groups: int = 1,
    ) -> int:
        """计算 Conv2d 的 FLOPs

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            output_shape: 输出形状 (batch, channels, height, width)
            groups: 分组卷积的组数

        Returns:
            FLOPs 总数
        """
        if len(output_shape) < 4:
            return 0

        batch, _, out_h, out_w = output_shape
        kernel_h, kernel_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)

        # 每个输出点的计算量：kernel_h * kernel_w * (in_channels // groups)
        per_output_flops = kernel_h * kernel_w * (in_channels // groups)
        # 总 FLOPs：输出大小 * 输出通道数 * 单个输出计算量
        total_flops = batch * out_h * out_w * out_channels * per_output_flops

        return total_flops

    @staticmethod
    def calc_linear_flops(
        in_features: int,
        out_features: int,
        batch_size: int = 1,
    ) -> int:
        """计算 Linear 的 FLOPs

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            batch_size: 批大小

        Returns:
            FLOPs 总数
        """
        # 每个输出：in_features 次乘法 + (in_features-1) 次加法 ≈ 2*in_features
        # 总 FLOPs = batch_size * out_features * 2 * in_features
        return batch_size * out_features * in_features * 2

    @staticmethod
    def calc_batchnorm_flops(
        channels: int,
        spatial_size: int,
        batch_size: int = 1,
    ) -> int:
        """计算 BatchNorm 的 FLOPs

        Args:
            channels: 通道数
            spatial_size: 空间大小 (height * width)
            batch_size: 批大小

        Returns:
            FLOPs 总数
        """
        # BatchNorm: 标准化 + 缩放偏移，约 5 个操作每个元素
        num_elements = batch_size * channels * spatial_size
        return num_elements * 5

    @staticmethod
    def calc_activation_flops(
        num_elements: int,
        complexity: int = 1,
    ) -> int:
        """计算激活函数的 FLOPs

        Args:
            num_elements: 元素总数
            complexity: 激活函数的复杂度（1=simple, >1=complex）

        Returns:
            FLOPs 总数
        """
        return num_elements * complexity

    @staticmethod
    def calc_pooling_flops(
        kernel_size: Tuple[int, int],
        output_shape: Tuple[int, ...],
    ) -> int:
        """计算池化操作的 FLOPs

        Args:
            kernel_size: 池化核大小
            output_shape: 输出形状 (batch, channels, height, width)

        Returns:
            FLOPs 总数
        """
        if len(output_shape) < 4:
            return 0

        batch, channels, out_h, out_w = output_shape
        kernel_h, kernel_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)

        # 每个输出点的计算量：kernel_h * kernel_w
        per_output_flops = kernel_h * kernel_w
        # 总 FLOPs：输出大小 * 通道数 * 单个输出计算量
        total_flops = batch * channels * out_h * out_w * per_output_flops

        return total_flops
