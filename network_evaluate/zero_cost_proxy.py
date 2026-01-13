import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import time
from typing import List, Tuple, Optional

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    else:
        raise NotImplementedError
    return model


def prepare_poly4_for_evaluation(model):
    """Set all StablePoly4 activations to post-warmup mode for evaluation

    This ensures that polynomial activations are evaluated in their full polynomial
    form rather than in the ReLU warmup phase.

    Args:
        model: PyTorch model potentially containing StablePoly4 activations
    """
    # Import here to avoid circular dependencies
    try:
        from models.gate_net_cmp.block_def import StablePoly4
        for module in model.modules():
            if isinstance(module, StablePoly4):
                module.set_epoch(9999)  # Set to max epoch for full polynomial behavior
    except ImportError:
        # StablePoly4 not available, skip
        pass


class ModelWrapper(nn.Module):
    """Wraps models to provide layer feature extraction interface

    This wrapper adds the ability to extract intermediate layer features
    required by the zero-cost proxy evaluation without modifying the
    original model architecture.

    Automatically detects model structure and hooks into appropriate layers:
    - For GeneratedNetwork models: hooks into blocks
    - For ResNet-style models: hooks into layer1, layer2, layer3, layer4
    - For other models: attempts to detect sequential modules

    Args:
        model: The model to wrap
        extract_from_layers: List of layer names to extract features from
                           (auto-detected if None)
    """

    def __init__(self, model: nn.Module, extract_from_layers: Optional[List[str]] = None):
        super().__init__()
        self.model = model
        self.no_reslink = getattr(model, 'no_reslink', False)

        # Auto-detect layers if not provided
        if extract_from_layers is None:
            extract_from_layers = self._auto_detect_layers()

        self.extract_from_layers = extract_from_layers
        self.layer_features = []
        self.hooks = []

        # Register forward hooks to capture layer outputs
        self._register_hooks()

    def _auto_detect_layers(self) -> List[str]:
        """Auto-detect which layers to extract features from

        Returns:
            List of layer names to hook into
        """
        # Check if model has 'blocks' attribute (GeneratedNetwork style)
        if hasattr(self.model, 'blocks') and isinstance(self.model.blocks, nn.ModuleList):
            num_blocks = len(self.model.blocks)
            if num_blocks == 0:
                return []

            # Extract features from evenly spaced blocks (4 points)
            # Similar to extracting from layer1, layer2, layer3, layer4 in ResNet
            if num_blocks >= 4:
                # Take features after blocks at 25%, 50%, 75%, 100%
                indices = [
                    num_blocks // 4 - 1,
                    num_blocks // 2 - 1,
                    3 * num_blocks // 4 - 1,
                    num_blocks - 1
                ]
                # Ensure all indices are valid and unique
                indices = sorted(list(set([max(0, idx) for idx in indices])))
                return [f'blocks.{idx}' for idx in indices]
            else:
                # For very small networks, use all blocks
                return [f'blocks.{i}' for i in range(num_blocks)]

        # Check for ResNet-style layers
        has_layer1 = any(name == 'layer1' for name, _ in self.model.named_modules())
        if has_layer1:
            return ['layer1', 'layer2', 'layer3', 'layer4']

        # Default: try to find any Sequential modules
        sequential_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Sequential) and name and '.' not in name:
                sequential_names.append(name)

        if sequential_names:
            return sequential_names[:4]  # Use first 4

        # Fallback: empty list (will extract no intermediate features)
        print("Warning: Could not auto-detect layers for feature extraction")
        return []

    def _register_hooks(self):
        """Register forward hooks on specified layers"""
        for name, module in self.model.named_modules():
            if name in self.extract_from_layers:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

        # Debug: warn if no hooks registered
        if len(self.hooks) == 0 and len(self.extract_from_layers) > 0:
            print(f"Warning: No hooks registered for layers: {self.extract_from_layers}")
            available_layers = [name for name, _ in self.model.named_modules() if name]
            print(f"Available layers (first 20): {available_layers[:20]}")

    def _make_hook(self, layer_name):
        """Create a hook function for a specific layer"""
        def hook_fn(module, input, output):
            # Store output with gradient tracking
            self.layer_features.append(output)
        return hook_fn

    def _clear_features(self):
        """Clear stored features"""
        self.layer_features = []

    def forward(self, x):
        """Forward pass through the model"""
        self._clear_features()
        output = self.model(x)
        return output

    def extract_layer_features_and_logit(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract intermediate layer features and final logits

        Args:
            x: Input tensor

        Returns:
            (layer_features, logits): Tuple of feature list and final output
        """
        self._clear_features()
        output = self.model(x)

        # layer_features now contains outputs from hooked layers
        features = self.layer_features.copy()

        return features, output

    def extract_layer_features_nores(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract layer features for models without residual connections

        Args:
            x: Input tensor

        Returns:
            layer_features: List of intermediate layer outputs
        """
        features, _ = self.extract_layer_features_and_logit(x)
        return features

    def get_FLOPs(self, resolution: int) -> float:
        """Get FLOPs count (delegates to wrapped model if available)"""
        if hasattr(self.model, 'get_FLOPs'):
            return self.model.get_FLOPs(resolution)
        else:
            # Fallback: return 0 if not available
            return 0.0

    def __del__(self):
        """Cleanup hooks when wrapper is deleted"""
        for hook in self.hooks:
            hook.remove()


def compute_fhe_latency(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> dict:
    """Compute FHE latency using fhe_statistics module

    Args:
        model: PyTorch model to evaluate
        input_shape: Input tensor shape (batch_size, channels, height, width)

    Returns:
        dict with keys:
            - 'fhe_latency': Total FHE latency (operation + bootstrap)
            - 'fhe_boot_count': Number of bootstrap operations
            - 'fhe_max_depth': Maximum circuit depth
            - 'fhe_operation_latency': Operation latency (excluding bootstrap)
            - 'fhe_boot_latency': Bootstrap latency only
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from fhe_statistics import FheInfo

        # FheInfo requires model to be on CPU for tracing
        # Check if model is on CUDA and move to CPU temporarily
        is_cuda = next(model.parameters()).is_cuda
        if is_cuda:
            model = model.cpu()

        # Create FHE statistics analyzer
        fhe_info = FheInfo(model, input_shape=input_shape, optimize_boot=True)
        fhe_info.run_statistics()

        # Move model back to CUDA if it was there originally
        if is_cuda:
            model = model.cuda()

        # Total latency = operation latency + bootstrap latency
        total_latency = fhe_info.total_latency + fhe_info.total_boot_latency

        return {
            'fhe_latency': float(total_latency),
            'fhe_boot_count': int(fhe_info.total_boot_count),
            'fhe_max_depth': int(fhe_info.get_max_depth()),
            'fhe_operation_latency': float(fhe_info.total_latency),
            'fhe_boot_latency': float(fhe_info.total_boot_latency)
        }
    except Exception as e:
        import traceback
        print(f"Warning: FHE latency calculation failed: {e}")
        traceback.print_exc()
        # Return placeholder values if calculation fails
        return {
            'fhe_latency': float('inf'),
            'fhe_boot_count': 0,
            'fhe_max_depth': 0,
            'fhe_operation_latency': 0.0,
            'fhe_boot_latency': 0.0
        }



def compute_nas_score(model, gpu, trainloader, resolution, batch_size, fp16=False, init=True, use_wrapper=True):
    """Compute NAS score using AZ-NAS zero-cost proxies and FHE latency

    Args:
        model: PyTorch model to evaluate
        gpu: GPU device ID
        trainloader: DataLoader for training data (None to use random input)
        resolution: Input image resolution
        batch_size: Batch size for evaluation
        fp16: Use FP16 precision
        init: Whether to initialize model weights
        use_wrapper: Whether to wrap model for feature extraction (True for most cases)

    Returns:
        dict with keys:
            - 'expressivity': Expressivity score
            - 'progressivity': Progressivity score
            - 'trainability': Trainability score
            - 'fhe_latency': Total FHE latency
            - 'fhe_boot_count': Number of bootstrap operations
            - 'fhe_max_depth': Maximum circuit depth
            - 'fhe_operation_latency': Operation latency (excluding bootstrap)
            - 'fhe_boot_latency': Bootstrap latency only
    """
    # Wrap model if not already wrapped (for feature extraction)
    if use_wrapper and not isinstance(model, ModelWrapper):
        model = ModelWrapper(model)

    # Prepare polynomial activations for evaluation (set to post-warmup mode)
    prepare_poly4_for_evaluation(model)

    model.train()
    model.cuda()
    info = {}

    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    if init:
        init_model(model, 'kaiming_norm_fanin')

    if trainloader == None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
    else:
        input_ = next(iter(trainloader))[0]
    
    if model.no_reslink:
        layer_features = model.extract_layer_features_nores(input_)
    else:
        layer_features, output = model.extract_layer_features_and_logit(input_)

    ################ expressivity & progressivity scores ################
    expressivity_scores = []
    for i in range(len(layer_features)):
        feat = layer_features[i].detach().clone()
        b,c,h,w = feat.size()
        feat = feat.permute(0,2,3,1).contiguous().view(b*h*w,c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.transpose(1,0),feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma) # faster version for computing eignevalues, can be adopted since sigma is symmetric
        prob_s = s / s.sum()
        score = (-prob_s)*torch.log(prob_s+1e-8)
        score = score.sum().item()
        expressivity_scores.append(score)
    expressivity_scores = np.array(expressivity_scores)
    progressivity = np.min(expressivity_scores[1:] - expressivity_scores[:-1])
    expressivity = np.sum(expressivity_scores)
    #####################################################################

    ################ trainability score ##############
    scores = []
    for i in reversed(range(1, len(layer_features))):
        f_out = layer_features[i]
        f_in = layer_features[i-1]

        # Note: f_out and f_in are intermediate features (non-leaf tensors)
        # We don't need to check or zero their gradients as they are computed on-the-fly

        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size()==g_in.size() and torch.all(g_in == g_out):
            scores.append(-np.inf)
        else:
            if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                bo,co,ho,wo = g_out.size()
                bi,ci,hi,wi = g_in.size()
                stride = int(hi/ho)
                pixel_unshuffle = nn.PixelUnshuffle(stride)
                g_in = pixel_unshuffle(g_in)
            bo,co,ho,wo = g_out.size()
            bi,ci,hi,wi = g_in.size()
            ### straight-forward way
            # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
            # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
            # mat = torch.bmm(g_in,g_out).mean(dim=0)
            ### efficient way # print(torch.allclose(mat, mat2, atol=1e-6))
            g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,co)
            g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci)
            mat = torch.mm(g_in.transpose(1,0),g_out) / (bo*ho*wo)
            ### make faster on cpu
            if mat.size(0) < mat.size(1):
                mat = mat.transpose(0,1)
            ###
            s = torch.linalg.svdvals(mat)
            scores.append(-s.max().item() - 1/(s.max().item()+1e-6)+2)
    trainability = np.mean(scores)
    #################################################

    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['progressivity'] = float(progressivity) if not np.isnan(progressivity) else -np.inf
    info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf

    # Compute FHE latency (replaces FLOPs complexity)
    # Get the underlying model if wrapped
    eval_model = model.model if isinstance(model, ModelWrapper) else model
    fhe_metrics = compute_fhe_latency(eval_model, (batch_size, 3, resolution, resolution))
    info.update(fhe_metrics)

    return info

