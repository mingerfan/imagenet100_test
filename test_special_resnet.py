import torch
from models.gate_net_cmp.block_compose import SpecialResNet
from models.gate_net_cmp.block_def import Relu

def test_basic_config():
    """Test with basic blocks."""
    config = [
        {"block_type": "basic", "out_channels": 64, "num_blocks": 2},
        {"block_type": "basic", "out_channels": 128, "stride": 2, "num_blocks": 2},
    ]
    
    model = SpecialResNet(config=config, in_channels=64)
    x = torch.randn(2, 64, 56, 56)
    output = model(x)
    print(f"Basic test - Input shape: {x.shape}, Output shape: {output.shape}")
    print("✓ Basic config test passed")
    

def test_bottleneck_config():
    """Test with bottleneck blocks."""
    config = [
        {"block_type": "bottleneck", "out_channels": 64, "num_blocks": 3, "factor": 4.0},
        {"block_type": "bottleneck", "out_channels": 128, "stride": 2, "num_blocks": 4, "factor": 4.0},
    ]
    
    model = SpecialResNet(config=config, in_channels=64)
    x = torch.randn(2, 64, 56, 56)
    output = model(x)
    print(f"\nBottleneck test - Input shape: {x.shape}, Output shape: {output.shape}")
    print("✓ Bottleneck config test passed")


def test_self_gated_config():
    """Test with self-gated blocks."""
    config = [
        {"block_type": "basic_self_gated", "out_channels": 64, "num_blocks": 2, "full_gated": False},
        {"block_type": "basic_self_gated", "out_channels": 128, "stride": 2, "num_blocks": 2, "full_gated": True},
    ]
    
    model = SpecialResNet(config=config, in_channels=64)
    x = torch.randn(2, 64, 56, 56)
    output = model(x)
    print(f"\nSelf-gated test - Input shape: {x.shape}, Output shape: {output.shape}")
    print("✓ Self-gated config test passed")


def test_mixed_config():
    """Test with mixed block types."""
    config = [
        {"block_type": "basic", "out_channels": 64, "num_blocks": 2},
        {"block_type": "bottleneck_self_gated", "out_channels": 128, "stride": 2, "num_blocks": 2, "factor": 2.0},
        {"block_type": "basic_self_gated", "out_channels": 256, "stride": 2, "num_blocks": 2, "full_gated": True},
    ]
    
    model = SpecialResNet(config=config, in_channels=64)
    x = torch.randn(2, 64, 56, 56)
    output = model(x)
    print(f"\nMixed test - Input shape: {x.shape}, Output shape: {output.shape}")
    print("✓ Mixed config test passed")


def test_get_config():
    """Test get_config method."""
    config = [
        {"block_type": "basic", "out_channels": 64, "num_blocks": 2},
    ]
    
    model = SpecialResNet(config=config, in_channels=64)
    returned_config = model.get_config()
    print(f"\nget_config test - Original: {config}")
    print(f"get_config test - Returned: {returned_config}")
    assert returned_config == config
    print("✓ get_config test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SpecialResNet Implementation")
    print("=" * 60)
    
    test_basic_config()
    test_bottleneck_config()
    test_self_gated_config()
    test_mixed_config()
    test_get_config()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
