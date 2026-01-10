import torch
from torch import nn
import os
import sys
from typing import List, Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加到搜索路径
sys.path.insert(0, current_dir)

from block_def import (
    BasicBlock,
    BasicSelfGatedBlock,
    BottleneckBlock,
    BottleneckSelfGatedBlock,
    Relu,
)


class SpecialResNet(nn.Module):
    def __init__(self, config: List[Dict[str, Any]], in_channels: int = 64):
        """
        SpecialResNet with configurable block structure.
        
        Args:
            config: List of dicts defining the network structure. Each dict should contain:
                - block_type: Type of block ("basic", "bottleneck", "basic_self_gated", "bottleneck_self_gated")
                - out_channels: Number of output channels
                - stride: Stride for the block (default: 1)
                - factor: Expansion factor for bottleneck blocks (default: 4.0)
                - num_blocks: Number of blocks of this type (default: 1)
            in_channels: Number of input channels (typically from initial conv+pool)
        """
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        self._build_layers()

    def get_config(self) -> List[Dict[str, Any]]:
        """Return the network configuration."""
        return self.config

    def _get_block_class(self, block_type: str):
        """Get the block class based on block_type string."""
        block_map = {
            "basic": BasicBlock,
            "bottleneck": BottleneckBlock,
            "basic_self_gated": BasicSelfGatedBlock,
            "bottleneck_self_gated": BottleneckSelfGatedBlock,
        }
        if block_type not in block_map:
            raise ValueError(f"Unknown block_type: {block_type}. Available types: {list(block_map.keys())}")
        return block_map[block_type]

    def _build_layers(self):
        """Build layers based on config."""
        current_channels = self.in_channels
        
        for stage_config in self.config:
            block_type = stage_config.get("block_type", "basic")
            out_channels = stage_config.get("out_channels", current_channels)
            stride = stage_config.get("stride", 1)
            factor = stage_config.get("factor", 4.0)
            num_blocks = stage_config.get("num_blocks", 1)
            activation = stage_config.get("activation", Relu)
            full_gated = stage_config.get("full_gated", False)
            
            block_class = self._get_block_class(block_type)
            
            # Build blocks for this stage
            for i in range(num_blocks):
                # Only use stride for the first block in each stage
                current_stride = stride if i == 0 else 1
                
                if block_type == "basic_self_gated":
                    block = block_class(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=current_stride,
                        activation=activation,
                        full_gated=full_gated
                    )
                elif block_type in ["bottleneck", "bottleneck_self_gated"]:
                    block = block_class(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=current_stride,
                        factor=factor,
                        activation=activation
                    )
                else:  # basic
                    block = block_class(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=current_stride,
                        activation=activation
                    )
                
                self.layers.append(block)
                current_channels = out_channels

    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
