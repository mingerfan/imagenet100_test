import argparse
import math
import os
import sys
from typing import Iterable, List, Optional, Tuple

import torch


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


def load_model(
    model_name: Optional[str],
    json_config: Optional[str],
    num_classes: int,
    pretrained: bool,
    checkpoint: Optional[str],
) -> torch.nn.Module:
    if json_config:
        from network_gen.network_config import NetworkConfig
        from network_gen.network_generator import create_network

        config = NetworkConfig.load(json_config)
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = load_model(
        model_name=args.model_name,
        json_config=args.json_config,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        checkpoint=args.ckpt,
    ).to(device)
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
