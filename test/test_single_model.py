import torch
from models.registry import get_model

print("Testing single model creation...")
model_name = 'resnet-basic-relu-layer1block1'

try:
    print(f"Creating model: {model_name}")
    model = get_model(model_name, num_classes=100)
    print(f"✓ Model created successfully")
    
    print(f"\nModel structure:")
    print(model)
    
    print(f"\nTesting forward pass...")
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"✓ Forward pass successful: {test_input.shape} -> {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()