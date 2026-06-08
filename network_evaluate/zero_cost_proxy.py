import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import random
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


def synflow_init(m):
    """
    SynFlow initialization: Initialize weights with mean=1.0

    This initialization is used for the SynFlow zero-cost proxy.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=1.0, std=0.05)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def init_model_synflow(model):
    """Initialize model for SynFlow evaluation"""
    model.apply(synflow_init)
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


def replace_poly4_with_swish_for_zcp(model):
    """Temporarily replace all StablePoly4 activations with Swish for ZCP evaluation

    StablePoly4 requires training to achieve good performance. During zero-cost proxy
    evaluation (before training), the polynomial parameters are randomly initialized
    and may perform poorly. We use Swish as a substitute during ZCP evaluation.

    Args:
        model: PyTorch model potentially containing StablePoly4 activations

    Returns:
        replaced_modules: List of (parent, attr_name, original_module) tuples for restoration
    """
    replaced_modules = []

    try:
        from models.gate_net_cmp.block_def import StablePoly4, Swish

        # Find and replace all StablePoly4 instances
        for name, module in model.named_modules():
            # Check each attribute of the module
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, StablePoly4):
                        # Replace with Swish
                        swish = Swish()
                        setattr(module, attr_name, swish)
                        replaced_modules.append((module, attr_name, attr))
                except (AttributeError, TypeError):
                    continue

    except ImportError:
        # StablePoly4 not available, skip
        pass

    return replaced_modules


def restore_poly4_activations(replaced_modules):
    """Restore original StablePoly4 activations after ZCP evaluation

    Args:
        replaced_modules: List of (parent, attr_name, original_module) tuples from replace_poly4_with_swish_for_zcp
    """
    for parent, attr_name, original_module in replaced_modules:
        setattr(parent, attr_name, original_module)


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
        original_device = next(model.parameters()).device
        is_cuda = original_device.type == 'cuda'
        if is_cuda:
            model = model.cpu()

        # Create FHE statistics analyzer
        fhe_info = FheInfo(model, input_shape=input_shape, optimize_boot=True)
        fhe_info.run_statistics()

        # Move model back to original device if it was there originally
        if is_cuda:
            model = model.to(original_device)

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



def compute_nas_score(model, gpu, trainloader, resolution, batch_size, fp16=False,
                       repeat=8, mixup_gamma=1e-2, include_synflow=False,
                       fhe_batch_size=1):
    """Compute NAS score using ZenNAS zero-cost proxy and FHE latency

    Uses ZEN score as the primary evaluation metric, which has been shown to have
    strong correlation with actual network performance.

    Reference: "Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition"
               (Lin et al., ICCV 2021)

    Args:
        model: PyTorch model to evaluate
        gpu: GPU device ID
        trainloader: DataLoader (unused, kept for API compatibility)
        resolution: Input image resolution
        batch_size: Batch size for evaluation
        fp16: Use FP16 precision
        repeat: Number of repetitions for ZEN score averaging
        mixup_gamma: Mixup coefficient for ZEN score
        include_synflow: Whether to compute SynFlow for sanity checking
        fhe_batch_size: Batch dimension used only for FHE shape propagation.
                        FHE latency accounting depends on C/H/W ciphertext
                        packing, not batch size, so keep this small.

    Returns:
        dict with keys:
            - 'zen_score': ZEN score (primary metric, higher is better)
            - 'std_zen_score': Standard deviation of ZEN scores
            - 'params': Number of parameters
            - 'flops': FLOPs count
            - 'fhe_latency': Total FHE latency
            - 'fhe_boot_count': Number of bootstrap operations
            - 'fhe_max_depth': Maximum circuit depth
            - 'fhe_operation_latency': Operation latency (excluding bootstrap)
            - 'fhe_boot_latency': Bootstrap latency only
            - 'synflow_score': SynFlow score (if enabled)
            - 'synflow_grad_norm': SynFlow gradient norm (if enabled)
            - 'synflow_params': SynFlow parameter count (if enabled)
            - 'synflow_issue': Potential SynFlow issue flag (if enabled)
            - 'synflow_ok': True if SynFlow looks valid (if enabled)
    """
    # Get the underlying model if wrapped
    eval_model = model.model if isinstance(model, ModelWrapper) else model

    # Replace Poly4 with Swish for ZCP evaluation (untrained Poly4 performs poorly)
    replaced_modules = replace_poly4_with_swish_for_zcp(eval_model)

    info = {}

    try:
        # ============ 1. Compute ZEN Score (Primary Metric) ============
        zen_results = compute_zen_score(
            eval_model, gpu, resolution, batch_size,
            repeat=repeat, mixup_gamma=mixup_gamma, fp16=fp16
        )
        info['zen_score'] = zen_results['zen_score']
        info['std_zen_score'] = zen_results['std_zen_score']

        # ============ 2. Compute Model Statistics ============
        # Parameter count
        n_params = sum(p.numel() for p in eval_model.parameters())
        info['params'] = int(n_params)

        # FLOPs calculation
        flops = compute_flops(eval_model, resolution)
        info['flops'] = float(flops)

        # ============ 3. Compute FHE Latency (for constraints) ============
        fhe_metrics = compute_fhe_latency(eval_model, (fhe_batch_size, 3, resolution, resolution))
        info.update(fhe_metrics)

        # ============ 4. Optional SynFlow Check ============
        if include_synflow:
            try:
                synflow_results = compute_synflow(
                    eval_model, gpu, resolution, batch_size, fp16=fp16
                )
            except Exception as e:
                synflow_results = {
                    'synflow_score': float('nan'),
                    'synflow_grad_norm': float('nan'),
                    'synflow_params': 0,
                    'synflow_issue': f'error: {e}',
                    'synflow_ok': False
                }
            info.update(synflow_results)

    finally:
        # Always restore original Poly4 activations
        restore_poly4_activations(replaced_modules)

    return info


def compute_flops(model: nn.Module, resolution: int) -> float:
    """Compute FLOPs for a model using torch profiler or manual calculation

    Args:
        model: PyTorch model
        resolution: Input image resolution

    Returns:
        FLOPs count (float)
    """
    # First try model's own method
    if hasattr(model, 'get_FLOPs'):
        try:
            return float(model.get_FLOPs(resolution))
        except:
            pass

    # Try using thop library
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, resolution, resolution).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(flops)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: thop FLOPs calculation failed: {e}")

    # Try using fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, resolution, resolution).to(device)
        flops = FlopCountAnalysis(model, dummy_input).total()
        return float(flops)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: fvcore FLOPs calculation failed: {e}")

    # Manual estimation as last resort
    return _estimate_flops_manual(model, resolution)


def _estimate_flops_manual(model: nn.Module, resolution: int) -> float:
    """Manually estimate FLOPs by analyzing model structure

    This is a rough estimation when profiling libraries are not available.
    """
    total_flops = 0
    input_size = resolution

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * Cin * Cout * K^2 * Hout * Wout
            out_size = input_size // module.stride[0] if module.stride[0] > 1 else input_size
            flops = 2 * module.in_channels * module.out_channels * \
                    module.kernel_size[0] * module.kernel_size[1] * \
                    out_size * out_size / module.groups
            total_flops += flops
            if module.stride[0] > 1:
                input_size = out_size

        elif isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features
            total_flops += 2 * module.in_features * module.out_features

        elif isinstance(module, nn.BatchNorm2d):
            # FLOPs ≈ 4 * num_features * H * W (mean, var, normalize, scale)
            total_flops += 4 * module.num_features * input_size * input_size

    return float(total_flops)


def compute_synflow(model, gpu, resolution, batch_size, fp16=False):
    """
    Compute SynFlow zero-cost proxy score

    SynFlow (Synthetic Flow) measures complexity and potential of a network
    by computing the sum of absolute gradients of all parameters.

    Reference: "Zero-Cost Proxies for Lightweight NAS" (Abdelfattah et al., 2021)

    Args:
        model: PyTorch model to evaluate
        gpu: GPU device ID
        resolution: Input image resolution
        batch_size: Batch size for evaluation
        fp16: Use FP16 precision

    Returns:
        dict with keys:
            - 'synflow_score': SynFlow score (higher is better)
            - 'synflow_grad_norm': L2 norm of gradients
            - 'synflow_params': Number of parameters
            - 'synflow_issue': Issue string if gradients look invalid
            - 'synflow_ok': True if score looks valid
    """
    model = model.model if isinstance(model, ModelWrapper) else model

    # Replace Poly4 with Swish for ZCP evaluation
    replaced_modules = replace_poly4_with_swish_for_zcp(model)

    try:
        model.train()

        if gpu is not None:
            device = torch.device('cuda:{}'.format(gpu))
        else:
            device = torch.device('cpu')

        model.to(device)

        if fp16:
            dtype = torch.half
        else:
            dtype = torch.float32

        # SynFlow initialization
        init_model_synflow(model)

        # Use all-ones input for SynFlow
        input_ = torch.ones(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)

        # Forward pass
        output = model(input_)

        # Compute loss (output should be as large as possible, so we minimize negative output)
        # For classification, we want to maximize output, so we minimize -output
        loss = -output.mean()

        # Backward pass
        loss.backward()

        # Compute SynFlow score: sum of absolute gradients for all parameters
        synflow_score = 0.0
        total_params = 0
        grad_norm_sq = 0.0

        for param in model.parameters():
            if param.grad is not None:
                synflow_score += param.grad.abs().sum().item()
                grad_norm_sq += (param.grad ** 2).sum().item()
                total_params += param.numel()

        # Compute gradient norm
        grad_norm = np.sqrt(grad_norm_sq)

        # Clean up gradients
        model.zero_grad()

        results = {
            'synflow_score': float(synflow_score),
            'synflow_grad_norm': float(grad_norm),
            'synflow_params': int(total_params)
        }
        synflow_issue = detect_synflow_issue(results)
        results['synflow_issue'] = synflow_issue
        results['synflow_ok'] = synflow_issue is None
        return results

    finally:
        # Always restore original Poly4 activations
        restore_poly4_activations(replaced_modules)


def detect_synflow_issue(results: dict) -> Optional[str]:
    """Check SynFlow results for potential architecture issues."""
    synflow_score = results.get('synflow_score', float('nan'))
    grad_norm = results.get('synflow_grad_norm', float('nan'))
    params = results.get('synflow_params', 0)

    if not np.isfinite(synflow_score) or not np.isfinite(grad_norm):
        return "non_finite"
    if params <= 0:
        return "no_params"
    if synflow_score <= 0 or grad_norm <= 0:
        return "zero_grad"
    return None


def compute_meco(model, gpu, resolution, batch_size, measure='meco_opt', fp16=False):
    """
    Compute MECO (Minimum Eigenvalue of COrrelation) zero-cost proxy score

    MECO measures the expressiveness of a network by computing the minimum
    eigenvalue of the correlation matrix of features at each layer. A higher
    MECO score indicates better feature diversity and less redundancy.

    Args:
        model: PyTorch model to evaluate
        gpu: GPU device ID
        resolution: Input image resolution
        batch_size: Batch size for evaluation
        measure: 'meco' for full correlation or 'meco_opt' for optimized sampling
        fp16: Use FP16 precision

    Returns:
        dict with keys:
            - 'meco_score': MECO score (higher is better)
            - 'num_layers': Number of layers evaluated
    """
    model = model.model if isinstance(model, ModelWrapper) else model

    # Replace Poly4 with Swish for ZCP evaluation
    replaced_modules = replace_poly4_with_swish_for_zcp(model)

    result_list = []
    hooks = []

    def forward_hook(module, data_input, data_output):
        # Handle tuple outputs (some layers return tuples)
        if isinstance(data_output, tuple):
            fea = data_output[0].clone().detach()
        else:
            fea = data_output.clone().detach()

        # Only process if we have enough dimensions
        if fea.dim() < 2:
            return

        n = fea.shape[0]
        if n < 2:
            return

        fea = fea.reshape(n, -1)

        if measure == 'meco':
            corr = torch.corrcoef(fea)
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            result = torch.min(torch.real(values))
        elif measure == 'meco_opt':
            # Optimized: sample 8 random indices for efficiency
            sample_size = min(8, n)
            if n > sample_size:
                idxs = random.sample(range(n), sample_size)
                fea = fea[idxs, :]
            corr = torch.corrcoef(fea)
            corr[torch.isnan(corr)] = 0
            corr[torch.isinf(corr)] = 0
            values = torch.linalg.eig(corr)[0]
            result = torch.min(torch.real(values)) * n / sample_size
        else:
            raise ValueError(f"Unknown measure: {measure}")

        result_list.append(result)

    try:
        model.eval()

        if gpu is not None:
            device = torch.device('cuda:{}'.format(gpu))
        else:
            device = torch.device('cpu')

        model.to(device)

        if fp16:
            dtype = torch.half
        else:
            dtype = torch.float32

        # Initialize model with kaiming normal
        init_model(model, method='kaiming_norm_fanin')

        # Register forward hooks on all modules
        for name, module in model.named_modules():
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

        # Generate random input
        x = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype)

        # Forward pass
        with torch.no_grad():
            model(x)

        # Aggregate results
        results = torch.tensor(result_list)
        results = results[torch.logical_not(torch.isnan(results))]
        results = results[torch.logical_not(torch.isinf(results))]
        meco_score = torch.sum(results).item()

        return {
            'meco_score': float(meco_score),
            'num_layers': int(len(results))
        }

    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        result_list.clear()

        # Restore original Poly4 activations
        restore_poly4_activations(replaced_modules)


def network_weight_gaussian_init(net: nn.Module):
    """
    Initialize network weights with Gaussian distribution

    Args:
        net: PyTorch model to initialize

    Returns:
        Initialized model
    """
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net


class ZENScoreWrapper(nn.Module):
    """
    Wrapper to extract features before the global average pooling layer
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.features = None
        self.hook = None

        # Register hook to capture features before GAP
        self._register_hook()

    def _register_hook(self):
        """Find and hook into the layer before global average pooling"""
        # Try to find common patterns for GAP/average pooling layers
        for name, module in list(self.model.named_modules())[-30:]:  # Check last 30 modules
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                # Found GAP layer, register hook on it to capture input
                self._hook_before_module(module)
                return  # Hook registered, exit

        # If no GAP found, try to find a layer before classifier
        # Look for the last Conv2d or Linear layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                # Register forward hook on this layer
                def capture_features(module, input, output):
                    self.features = output
                self.hook = module.register_forward_hook(capture_features)
                return

    def _hook_before_module(self, target_module):
        """Register hook on the module before the target"""
        def find_prev_hook(module, input):
            # Store the input to the GAP layer
            if len(input) > 0:
                self.features = input[0]

        # Register forward pre-hook on GAP layer
        self.hook = target_module.register_forward_pre_hook(find_prev_hook)

    def forward_pre_GAP(self, x):
        """Forward pass to get features before GAP"""
        if self.features is not None:
            self.features = None
        self.model(x)
        return self.features

    def __del__(self):
        if self.hook is not None:
            self.hook.remove()


def compute_zen_score(model, gpu, resolution, batch_size, repeat=32, mixup_gamma=1e-2, fp16=False):
    """
    Compute ZEN (Zero-Shot Evaluation of Networks) score

    ZEN measures the network's expressiveness by evaluating how the network responds
    to mixup perturbations.

    Reference: "Zero-Shot Evaluation of Networks" (You et al., 2021)

    Args:
        model: PyTorch model to evaluate
        gpu: GPU device ID
        resolution: Input image resolution
        batch_size: Batch size for evaluation
        repeat: Number of repetitions for averaging
        mixup_gamma: Mixup coefficient
        fp16: Use FP16 precision

    Returns:
        dict with keys:
            - 'zen_score': Average ZEN score
            - 'std_zen_score': Standard deviation of ZEN scores
            - 'zen_precision': 95% confidence interval precision
    """
    model = model.model if isinstance(model, ModelWrapper) else model

    # Wrap model to get features before GAP
    zen_model = ZENScoreWrapper(model)
    zen_model.train()

    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    zen_model.to(device)

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    zen_score_list = []

    with torch.no_grad():
        for repeat_count in range(repeat):
            # Initialize weights with Gaussian distribution
            network_weight_gaussian_init(zen_model)

            # Generate two random inputs
            input1 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)

            # Create mixup input
            mixup_input = input1 + mixup_gamma * input2

            # Get features before GAP
            output = zen_model.forward_pre_GAP(input1)
            mixup_output = zen_model.forward_pre_GAP(mixup_input)

            # Compute ZEN score
            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # Compute BN scaling factor
            log_bn_scaling_factor = 0.0
            for m in zen_model.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)

            # Final ZEN score
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            zen_score_list.append(float(nas_score))

    # Compute statistics
    std_zen_score = np.std(zen_score_list)
    avg_precision = 1.96 * std_zen_score / np.sqrt(len(zen_score_list))
    avg_zen_score = np.mean(zen_score_list)

    return {
        'zen_score': float(avg_zen_score),
        'std_zen_score': float(std_zen_score),
        'zen_precision': float(avg_precision)
    }
