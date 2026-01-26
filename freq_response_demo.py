import argparse
import math
import os
import sys
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_list(arg: str, cast_type=float) -> List:
    if arg is None or arg == "":
        return []
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    return [cast_type(p) for p in parts]


def set_epoch_for_model(model: torch.nn.Module, epoch: int) -> None:
    for module in model.modules():
        if hasattr(module, "set_epoch") and callable(module.set_epoch):
            module.set_epoch(epoch)


def prepare_poly4_eval(model: torch.nn.Module) -> None:
    try:
        from network_evaluate.zero_cost_proxy import prepare_poly4_for_evaluation
    except Exception:
        return
    prepare_poly4_for_evaluation(model)


def _load_json_dict(path: str) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("config"), dict):
        return data["config"]
    if isinstance(data, dict):
        return data
    raise ValueError("JSON root must be an object or contain a 'config' object")


def _expand_block_choices(block_choices: List[int], num_blocks: int) -> List[int]:
    if len(block_choices) == num_blocks:
        return block_choices
    try:
        from network_gen.network_generator import HierarchicalBlockSelector

        selector = HierarchicalBlockSelector()
        expanded = selector.expand_choices(block_choices, num_blocks)
        return expanded
    except Exception:
        raise ValueError(
            f"block_choices length {len(block_choices)} does not match num_blocks {num_blocks}"
        )


def _build_config_from_dict(config_dict: dict, input_size: int) -> "NetworkConfig":
    from network_gen.network_config import NetworkConfig, BlockConfig
    from network_gen.search_space import StrideEncoder, ChannelCalculator

    if "blocks" in config_dict:
        return NetworkConfig.from_dict(config_dict)

    required = ["stem_code", "second_ds_code", "stride_code", "ct_policies", "block_choices"]
    missing = [k for k in required if k not in config_dict]
    if missing:
        raise KeyError(f"Missing keys in json config: {missing}")

    stem_code = config_dict["stem_code"]
    second_ds_code = config_dict["second_ds_code"]
    stride_code = config_dict["stride_code"]
    ct_policies = list(config_dict["ct_policies"])
    block_choices = list(config_dict["block_choices"])
    initial_ct_count = config_dict.get("initial_ct_count", 1)

    ct_slots = config_dict.get("ct_slots", 32768)
    config_input_size = int(config_dict.get("input_size", input_size))

    stride_encoder = StrideEncoder()
    num_blocks, stride_positions = stride_encoder.decode(stride_code)
    strides = stride_encoder.get_strides_list(num_blocks, stride_positions)

    if len(ct_policies) < 3:
        ct_policies = ct_policies + ["keep"] * (3 - len(ct_policies))

    block_ids = _expand_block_choices(block_choices, num_blocks)

    channel_calculator = ChannelCalculator(ct_slots, config_input_size)
    channels, _, _ = channel_calculator.compute_channels_sequence(
        strides=strides,
        ct_policies=ct_policies,
        initial_ct_count=initial_ct_count,
    )

    stem_out_channels = channel_calculator.get_initial_channels(initial_ct_count)
    blocks = []
    for i in range(num_blocks):
        in_channels = stem_out_channels if i == 0 else blocks[-1].out_channels
        out_channels = channels[i]
        blocks.append(
            BlockConfig(
                block_id=block_ids[i],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[i],
            )
        )

    return NetworkConfig(
        stem_code=stem_code,
        second_ds_code=second_ds_code,
        stride_code=stride_code,
        ct_policies=ct_policies,
        block_choices=block_ids,
        blocks=blocks,
        initial_ct_count=initial_ct_count,
        stem_out_channels=stem_out_channels,
        num_classes=config_dict.get("num_classes", 100),
        name=config_dict.get("name"),
        description=config_dict.get("description"),
        created_at=config_dict.get("created_at"),
    )


def load_model(
    model_name: Optional[str],
    json_config: Optional[str],
    num_classes: int,
    pretrained: bool,
    checkpoint: Optional[str],
    input_size: int,
) -> torch.nn.Module:
    if json_config:
        from network_gen.network_generator import create_network

        config_dict = _load_json_dict(json_config)
        config = _build_config_from_dict(config_dict, input_size=input_size)
        if num_classes is not None:
            config.num_classes = num_classes
        model = create_network(config)
    else:
        if not model_name:
            raise ValueError("Either --model-name or --json-config must be provided.")
        from models import get_model

        model = get_model(model_name, num_classes=num_classes, pretrained=pretrained)

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"WARN load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


class _LimitedLoader:
    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        try:
            return min(len(self.loader), self.max_batches)
        except Exception:
            return self.max_batches


class SineGratingDataset(Dataset):
    def __init__(
        self,
        freqs: List[float],
        angles: List[float],
        samples_per_class: int,
        size: int,
        amplitude: float,
        normalize: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        noise_std: float,
        noise_type: str,
        hf_cutoff: float,
        label_mode: str,
        seed: int,
    ):
        self.freqs = freqs
        self.angles = angles
        self.samples_per_class = samples_per_class
        self.size = size
        self.amplitude = amplitude
        self.normalize = normalize
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.hf_cutoff = hf_cutoff
        self.label_mode = label_mode
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        if label_mode not in ("freq", "angle", "freq_angle"):
            raise ValueError("label_mode must be freq, angle, or freq_angle")

        self.class_pairs = []
        for f in self.freqs:
            for a in self.angles:
                self.class_pairs.append((f, a))

    def __len__(self):
        return len(self.class_pairs) * self.samples_per_class

    def _label_for_pair(self, f: float, a: float) -> int:
        if self.label_mode == "freq":
            return self.freqs.index(f)
        if self.label_mode == "angle":
            return self.angles.index(a)
        # freq_angle
        return self.class_pairs.index((f, a))

    def __getitem__(self, idx):
        class_idx = idx // self.samples_per_class
        f, a = self.class_pairs[class_idx]
        x = make_sine_batch(
            batch_size=1,
            size=self.size,
            freq=f,
            angle_deg=a,
            amplitude=self.amplitude,
            device=torch.device("cpu"),
            phase_random=True,
        )[0]

        if self.noise_std > 0:
            x, _ = add_noise(x.unsqueeze(0), self.noise_std, self.noise_type, self.hf_cutoff)
            x = x[0]

        if self.normalize is not None:
            mean, std = self.normalize
            x = normalize_batch(x.unsqueeze(0), mean, std)[0]

        y = self._label_for_pair(f, a)
        return x, y


def build_sine_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, int]:
    if args.sine_label_mode == "freq":
        num_classes = len(args.sine_freqs)
    elif args.sine_label_mode == "angle":
        num_classes = len(args.sine_angles)
    else:
        num_classes = len(args.sine_freqs) * len(args.sine_angles)

    normalize = None
    if args.normalize == "imagenet":
        normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif args.normalize == "cifar10":
        normalize = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    dataset = SineGratingDataset(
        freqs=args.sine_freqs,
        angles=args.sine_angles,
        samples_per_class=args.sine_samples_per_class,
        size=args.input_size,
        amplitude=args.sine_amplitude,
        normalize=normalize,
        noise_std=args.sine_train_noise_std,
        noise_type=args.sine_train_noise_type,
        hf_cutoff=args.sine_train_hf_cutoff,
        label_mode=args.sine_label_mode,
        seed=args.seed,
    )

    val_samples = max(1, int(args.sine_samples_per_class * args.sine_val_ratio))
    val_dataset = SineGratingDataset(
        freqs=args.sine_freqs,
        angles=args.sine_angles,
        samples_per_class=val_samples,
        size=args.input_size,
        amplitude=args.sine_amplitude,
        normalize=normalize,
        noise_std=args.sine_val_noise_std,
        noise_type=args.sine_val_noise_type,
        hf_cutoff=args.sine_val_hf_cutoff,
        label_mode=args.sine_label_mode,
        seed=args.seed + 123,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size or args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=not args.cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.train_num_workers,
        pin_memory=not args.cpu,
    )
    return train_loader, val_loader, num_classes


def maybe_train(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.train_epochs <= 0:
        return

    from trainers.base_trainer import Trainer
    from trainers.multi_gpu_manager import create_smart_optimizer

    if args.train_on_sine:
        train_loader, val_loader, sine_num_classes = build_sine_dataloaders(args)
        if model.classifier is not None:
            pass
        if args.num_classes != sine_num_classes:
            print(
                f"WARN num_classes={args.num_classes} != sine classes {sine_num_classes}, "
                "ensure model output matches."
            )
    else:
        from data import create_dataloaders

        if args.dataset in ("imagenet100", "imagenet1k", "imagenet"):
            if not args.train_dir or not args.val_dir:
                raise ValueError("ImageFolder 数据集需要提供 --train-dir 和 --val-dir")

        train_bs = args.train_batch_size or args.batch_size
        train_loader, val_loader, _, _ = create_dataloaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=train_bs,
            num_workers=args.train_num_workers,
            pin_memory=not args.cpu,
            use_memory_fs=args.use_memory_fs,
            dataset=args.dataset,
            download=args.download,
            input_size=args.input_size,
            seed=args.seed,
        )

    if args.train_max_batches and args.train_max_batches > 0:
        train_loader = _LimitedLoader(train_loader, args.train_max_batches)
        val_loader = _LimitedLoader(val_loader, args.train_max_batches)

    optimizer = create_smart_optimizer(model, lr=args.train_lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        result_dir=args.train_out,
        epochs=args.train_epochs,
        scheduler=None,
        use_amp=not args.train_no_amp,
        save_freq=0,
        save_checkpoints=False,
        grad_clip_max_norm=args.grad_clip_max_norm,
        poly4_warmup_ratio=args.poly4_warmup_ratio,
    )

    trainer.train()


def build_grid(size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(0.0, 1.0, steps=size, device=device)
    xs = torch.linspace(0.0, 1.0, steps=size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return yy, xx


def make_sine_batch(
    batch_size: int,
    size: int,
    freq: float,
    angle_deg: float,
    amplitude: float,
    device: torch.device,
    phase_random: bool = True,
) -> torch.Tensor:
    yy, xx = build_grid(size, device)
    theta = math.radians(angle_deg)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    base = 2.0 * math.pi * freq * proj
    if phase_random:
        phase = torch.rand(batch_size, 1, 1, 1, device=device) * 2.0 * math.pi
    else:
        phase = torch.zeros(batch_size, 1, 1, 1, device=device)
    wave = amplitude * torch.sin(base.unsqueeze(0).unsqueeze(0) + phase)
    return wave.repeat(1, 3, 1, 1)


def normalize_batch(x: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    mean_t = torch.tensor(list(mean), device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(list(std), device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def high_pass_noise(
    shape: Tuple[int, int, int, int],
    cutoff_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    b, c, h, w = shape
    noise = torch.randn(shape, device=device)
    fft = torch.fft.rfft2(noise, norm="ortho")
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx ** 2 + yy ** 2)
    mask = (radius >= cutoff_ratio).to(fft.dtype)
    fft = fft * mask
    hp = torch.fft.irfft2(fft, s=(h, w), norm="ortho")
    return hp


def add_noise(
    x: torch.Tensor,
    noise_std: float,
    noise_type: str,
    hf_cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_std <= 0:
        return x, torch.zeros_like(x)
    if noise_type == "hf":
        noise = high_pass_noise(x.shape, hf_cutoff, x.device)
        noise = noise / (noise.std() + 1e-8)
        noise = noise * noise_std
    else:
        noise = torch.randn_like(x) * noise_std
    return x + noise, noise


def model_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[-1]
    return y


def compute_metrics(
    model: torch.nn.Module,
    x: torch.Tensor,
    noise_std: float,
    noise_type: str,
    hf_cutoff: float,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        y = model_forward(model, x)
        y_flat = y.view(y.shape[0], -1)
        out_rms = torch.sqrt((y_flat ** 2).mean(dim=1))
        out_rms_mean = out_rms.mean().item()

        x_rms = torch.sqrt((x ** 2).mean(dim=(1, 2, 3)))
        in_rms_mean = x_rms.mean().item()

        if noise_std > 0:
            x_noisy, noise = add_noise(x, noise_std, noise_type, hf_cutoff)
            y_noisy = model_forward(model, x_noisy)
            delta = y_noisy - y
            delta_flat = delta.view(delta.shape[0], -1)
            noise_flat = noise.view(noise.shape[0], -1)
            delta_norm = torch.linalg.vector_norm(delta_flat, dim=1)
            noise_norm = torch.linalg.vector_norm(noise_flat, dim=1)
            sens = (delta_norm / (noise_norm + 1e-8)).mean().item()
        else:
            sens = 0.0

    return out_rms_mean, in_rms_mean, sens


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency response demo (single file)")
    parser.add_argument("--model-name", type=str, default=None, help="Registered model name")
    parser.add_argument("--json-config", type=str, default=None, help="NAS JSON config path")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--freqs", type=str, default="1,2,4,8,16")
    parser.add_argument("--angles", type=str, default="0,45,90")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "hf"])
    parser.add_argument("--hf-cutoff", type=float, default=0.25, help="High-pass cutoff ratio")
    parser.add_argument("--normalize", type=str, default="none", choices=["none", "imagenet", "cifar10"])
    parser.add_argument("--epoch", type=int, default=None, help="Set epoch for modules like StablePoly4")
    parser.add_argument("--poly4-eval", action="store_true", help="Force StablePoly4 to full polynomial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--train-epochs", type=int, default=0, help="Train for N epochs before eval")
    parser.add_argument("--train-dir", type=str, default=None, help="Training data directory")
    parser.add_argument("--val-dir", type=str, default=None, help="Validation data directory")
    parser.add_argument("--dataset", type=str, default="imagenet100")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--train-num-workers", type=int, default=8)
    parser.add_argument("--train-lr", type=float, default=0.001)
    parser.add_argument("--train-no-amp", action="store_true")
    parser.add_argument("--train-out", type=str, default="tmp/freq_response_train")
    parser.add_argument("--train-max-batches", type=int, default=0)
    parser.add_argument("--use-memory-fs", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--grad-clip-max-norm", type=float, default=1.0)
    parser.add_argument("--poly4-warmup-ratio", type=float, default=0.5)
    parser.add_argument("--train-on-sine", action="store_true", help="Train on synthetic sine gratings")
    parser.add_argument("--sine-freqs", type=str, default="1,2,4,8,16")
    parser.add_argument("--sine-angles", type=str, default="0,45,90")
    parser.add_argument("--sine-samples-per-class", type=int, default=128)
    parser.add_argument("--sine-val-ratio", type=float, default=0.25)
    parser.add_argument("--sine-amplitude", type=float, default=1.0)
    parser.add_argument("--sine-label-mode", type=str, default="freq_angle", choices=["freq", "angle", "freq_angle"])
    parser.add_argument("--sine-train-noise-std", type=float, default=0.0)
    parser.add_argument("--sine-train-noise-type", type=str, default="gaussian", choices=["gaussian", "hf"])
    parser.add_argument("--sine-train-hf-cutoff", type=float, default=0.25)
    parser.add_argument("--sine-val-noise-std", type=float, default=0.0)
    parser.add_argument("--sine-val-noise-type", type=str, default="gaussian", choices=["gaussian", "hf"])
    parser.add_argument("--sine-val-hf-cutoff", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    args.sine_freqs = parse_list(args.sine_freqs, float)
    args.sine_angles = parse_list(args.sine_angles, float)

    if args.train_on_sine:
        if args.sine_label_mode == "freq":
            sine_classes = len(args.sine_freqs)
        elif args.sine_label_mode == "angle":
            sine_classes = len(args.sine_angles)
        else:
            sine_classes = len(args.sine_freqs) * len(args.sine_angles)
        args.num_classes = sine_classes

    model = load_model(
        model_name=args.model_name,
        json_config=args.json_config,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        checkpoint=args.ckpt,
        input_size=args.input_size,
    ).to(device)

    maybe_train(model, args, device)
    model.eval()

    if args.poly4_eval:
        prepare_poly4_eval(model)
    if args.epoch is not None:
        set_epoch_for_model(model, args.epoch)

    if args.normalize == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif args.normalize == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        mean, std = None, None

    freqs = parse_list(args.freqs, float)
    angles = parse_list(args.angles, float)

    if not freqs:
        raise ValueError("No frequencies provided.")
    if not angles:
        raise ValueError("No angles provided.")

    print("Frequency response demo")
    print(f"  model: {args.model_name or args.json_config}")
    print(f"  device: {device}")
    print(f"  input size: {args.input_size}")
    print(f"  batch size: {args.batch_size} x {args.num_batches} batches")
    print(f"  freqs: {freqs}")
    print(f"  angles: {angles}")
    print(f"  noise std/type: {args.noise_std} / {args.noise_type}")

    rows = []
    header = [
        "freq",
        "angle",
        "out_rms",
        "in_rms",
        "gain_db",
        "noise_sens",
    ]
    print("\n" + "\t".join(header))
    for freq in freqs:
        for angle in angles:
            out_vals = []
            in_vals = []
            sens_vals = []
            for _ in range(args.num_batches):
                x = make_sine_batch(
                    batch_size=args.batch_size,
                    size=args.input_size,
                    freq=freq,
                    angle_deg=angle,
                    amplitude=args.amplitude,
                    device=device,
                    phase_random=True,
                )
                if mean is not None and std is not None:
                    x = normalize_batch(x, mean, std)
                out_rms, in_rms, sens = compute_metrics(
                    model=model,
                    x=x,
                    noise_std=args.noise_std,
                    noise_type=args.noise_type,
                    hf_cutoff=args.hf_cutoff,
                )
                out_vals.append(out_rms)
                in_vals.append(in_rms)
                sens_vals.append(sens)

            out_rms = float(sum(out_vals) / len(out_vals))
            in_rms = float(sum(in_vals) / len(in_vals))
            sens = float(sum(sens_vals) / len(sens_vals))
            gain_db = 20.0 * math.log10(out_rms / (in_rms + 1e-12))
            row = [freq, angle, out_rms, in_rms, gain_db, sens]
            rows.append(row)
            print(
                f"{freq:.3f}\t{angle:.1f}\t{out_rms:.6f}\t{in_rms:.6f}\t{gain_db:.3f}\t{sens:.6f}"
            )

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(
                    f"{row[0]:.6f},{row[1]:.3f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n"
                )
        print(f"\nSaved CSV to: {args.csv}")


if __name__ == "__main__":
    main()
