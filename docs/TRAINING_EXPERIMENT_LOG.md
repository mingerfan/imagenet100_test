# Training Experiment Log

This file records proxy experiments used to decide whether a training technique
is stable enough to keep. Every committed training change should have a matching
entry with the command, result path, and conclusion.

## 2026-05-31 SmartPAF Stability Round 1

Environment:

- Project venv: `/home/xuming/Documents/EfficientNet/imagenet100_test/.venv`
- Torch: `2.9.1+cu128`
- GPUs: 4 x V100, GPU0 excluded by default
- ImageNet-100 data: `/home/xuming/Documents/dataset/imagenet_100/train` and `/home/xuming/Documents/dataset/imagenet_100/val`
- Shared cache: `/dev/shm/imagenet100`

Infrastructure changes validated before these runs:

- Non-interactive `tqdm` is disabled, so captured logs only grow by epoch summaries.
- ImageFolder memory cache now follows symlinks for size checks and uses `.copy_complete` to avoid reusing partial copies.
- `train.py` no longer overrides YAML `use_amp`/`val_force_fp32` unless the CLI flag is explicitly passed.
- `train.py` exits non-zero when any model fails.
- Collapse checkpoints are included by the summary tool when determining COLLAPSE status.

Validation commands:

```bash
.venv/bin/python -m py_compile train.py trainers/base_trainer.py trainers/multi_gpu_manager.py data/dataset.py data/memory_fs.py tools/summarize_training_runs.py
.venv/bin/python tools/summarize_training_runs.py results/proxy_imagenet100_96_pa_only_fast results/proxy_imagenet100_96_swish_baseline_fast results/proxy_imagenet100_96_pa_at_fast results/proxy_at_imagenet100_96_long results/proxy_ablation_cifar10
```

### CIFAR-10 proxy ablation

Command source: previous proxy ablation run under `configs/proxy_ablation_cifar10.yaml`.

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ablation-cifar10-pa-only` | 5 | 73.07 | 73.07 | 0.00 | 0 | 0 | 0 | PASS |
| `ablation-cifar10-pa-at-poly-lr-1` | 5 | 67.10 | 67.10 | 5.62 | 0 | 0 | 0 | PASS |
| `ablation-cifar10-pa-at-poly-lr-01` | 5 | 65.51 | 65.51 | 15.56 | 0 | 0 | 1 | COLLAPSE |

Conclusion: PA-only was the best CIFAR-10 proxy result. AT did not improve accuracy
and lower poly LR made collapse behavior worse.

### ImageNet-100 96px baseline and PA reruns

Commands:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_swish_baseline_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 3 --input_size 96 --force > logs/proxy_imagenet100_96_swish_baseline_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_swish_baseline_fast.status' < /dev/null &
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_only_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_only_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_only_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs in CSV | Best | Final CSV | Collapse checkpoint | Status |
| --- | ---: | ---: | ---: | --- | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | no | PASS |
| `imagenet100-96-pa-only-b256-conservative` | 14 | 47.10 | 47.10 | `collapse_epoch_15.pth` | COLLAPSE |

PA-only collapse details from `logs/proxy_imagenet100_96_pa_only_fast.log`:

- Epoch 14 validation accuracy: 47.10%.
- Epoch 15 validation accuracy: 27.20%.
- Collapse guard drop: 19.90 percentage points.
- `nonfinite_train_batches = 0`, `nonfinite_val_batches = 0`.
- `|x_poly| max` at collapse: layer0 3.198, layer1 0.7251.

Conclusion: PA-only tracks the Swish baseline closely through epoch 14 but collapses
when the replacement is near full polynomial activation. The failure is not caused
by NaN/Inf batches.

### ImageNet-100 96px PA + AT rerun

Command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_at_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_at_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_at_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs in CSV | Best | Final CSV | Collapse checkpoint | Status |
| --- | ---: | ---: | ---: | --- | --- |
| `imagenet100-96-pa-at-b256-cycle1` | 17 | 41.04 | 41.04 | `collapse_epoch_18.pth` | COLLAPSE |

PA+AT collapse details from `logs/proxy_imagenet100_96_pa_at_fast.log`:

- Epoch 14 poly phase train loss: 147373.44.
- Epoch 16 poly phase train loss: 20097787743.71.
- Epoch 17 validation accuracy recovered to 41.04%.
- Epoch 18 validation accuracy collapsed to 17.32%.
- Collapse guard drop: 23.72 percentage points.
- `nonfinite_train_batches = 0`, `nonfinite_val_batches = 0`.
- `|x_poly| max` at collapse: layer0 7.262, layer1 2.935.

Conclusion: AT delays the collapse compared with PA-only but does not solve it.
It also lowers best accuracy substantially. The likely next targets are coefficient
initialization and scale control rather than more AT-only tuning.

## Technique Backlog From SMART-PAF

Implemented experimentally:

- CT / Coefficient Tuning: implemented as default-off `smartpaf_ct_init`.
- DS / Dynamic Scale: implemented as default-off `poly4_scale_mode: dynamic`,
  but DS-only failed without CT.
- SS / Static Scale: implemented as default-off `smartpaf_ss_calibrate` with
  `poly4_scale_mode: static`.

Not yet implemented or only partially implemented:

- SWA: implemented as default-off global averaging, but not yet integrated into
  the original SMART-PAF per-training-group acceptance loop.
- BN recalibration: implemented as default-off post-training recalibration.
- Dropout mitigation: not implemented.
- Full per-layer CT -> PA -> AT -> DS/SS scheduler: not implemented. Current
  code is a lightweight global scheduler.

## 2026-05-31 Dynamic Scale Attempt

Implemented as an experimental, default-off StablePoly4 option:

- `poly4_scale_mode: learned | dynamic | static`
- `poly4_dynamic_scale_momentum`
- `poly4_dynamic_scale_eps`

Validation commands:

```bash
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py trainers/base_trainer.py
.venv/bin/python - <<'PY'
import torch
from models.gate_net_cmp.block_def import StablePoly4
m = StablePoly4(scale_mode='dynamic')
x = torch.randn(4, 3, 8, 8) * 5
m.train()
y = m(x)
print(torch.isfinite(y).all().item(), float(m.running_absmax), m.scale_mode)
m.eval()
y2 = m(x)
print(torch.isfinite(y2).all().item(), y2.shape)
PY
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_dynamic_scale_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_dynamic_scale_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_dynamic_scale_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ds-b256` | 10 | 36.44 | 9.46 | 26.98 | 0 | 0 | 1 | COLLAPSE |

Collapse details from `logs/proxy_imagenet100_96_pa_dynamic_scale_fast.log`:

- Epoch 9 validation accuracy: 36.44%.
- Epoch 10 validation accuracy: 9.46%.
- Collapse guard drop: 26.98 percentage points.
- `nonfinite_train_batches = 0`, `nonfinite_val_batches = 0`.
- Dynamic scale constrained the polynomial inputs: layer0 `|x_poly| max = 0.6287`, layer1 `|x_poly| max = 0.3752`.

Conclusion: DS alone prevents large polynomial inputs but does not align the
polynomial branch with the Swish target. The next required technique is CT /
coefficient initialization before retrying DS or AT.

## 2026-05-31 Coefficient Tuning Initialization

Implemented as an experimental, default-off trainer option:

- `smartpaf_ct_init`
- `smartpaf_ct_batches`
- `smartpaf_ct_max_samples`
- `smartpaf_ct_steps`
- `smartpaf_ct_lr`

CT samples StablePoly4 pre-activation tensors before normal training and fits
`a,b,c,d,e,log_in_scale` to the module's warmup activation with MSE.

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_fast.status' < /dev/null &
```

CT fit quality:

| Module | Samples | MSE |
| --- | ---: | ---: |
| `special_resnet.layers.0.act` | 20000 | 0.00205496 |
| `special_resnet.layers.1.act` | 20000 | 0.0010167 |

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | PASS |
| `imagenet100-96-pa-only-b256-conservative` | 14 | 47.10 | 47.10 | COLLAPSE |

Conclusion: CT fixes the late PA-only collapse on the ImageNet-100 96px proxy
and nearly matches the Swish baseline. The next useful AT run should be CT+AT,
not AT on scratch-initialized polynomial coefficients.

## 2026-05-31 CT + Alternate Training Attempt

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_at_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 3 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_at_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_at_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-at-b256-cycle1` | 17 | 38.30 | 16.52 | 21.78 | 1 | 1 | 1 | COLLAPSE |

Failure details:

- CT fit matched the CT-only run:
  `special_resnet.layers.0.act` MSE 0.00205496 and
  `special_resnet.layers.1.act` MSE 0.0010167.
- Epoch 14 poly phase train loss exploded to 133118055141.72, but validation
  accuracy only moved from 35.74% to 35.08%.
- Epoch 16 poly phase skipped one optimizer step because gradient norm was
  non-finite.
- Epoch 17 weights phase triggered collapse guard: validation accuracy dropped
  from 38.30% to 16.52%, and `collapse_epoch_17.pth` was saved.

Conclusion: CT delays AT collapse compared with the no-CT PA+AT run, but this
AT schedule is still not stable and is less accurate than CT-only. The current
AT implementation also wastes early poly phases before any polynomial branch
contributes to the forward pass. AT should stay experimental; the next AT change
should delay alternating training until the first polynomial module is active or
move closer to the original per-layer CT -> PA -> AT schedule.

## 2026-05-31 CT + Dynamic Scale Attempt

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ds_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ds_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ds_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ds-b256` | 16 | 44.58 | 42.98 | 5.52 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | PASS |
| `imagenet100-96-pa-ct-ds-b256` | 16 | 44.58 | 42.98 | PASS |
| `imagenet100-96-pa-ds-b256` | 10 | 36.44 | 9.46 | COLLAPSE |

Observations:

- CT fit matched previous CT runs: layer0 MSE 0.00205496 and layer1 MSE
  0.0010167.
- Dynamic scale kept effective input scale around 0.11-0.17 late in training.
- The run did not produce non-finite batches or skipped optimizer steps.
- Accuracy peaked at epoch 14, then dropped from 44.58% to 39.06% at epoch 15
  and recovered to 42.98% at epoch 16.

Conclusion: CT makes DS stable, unlike DS-only, but DS still reduces final
accuracy relative to CT-only on this proxy. Keep DS experimental. A better next
scale-control step is static calibration/frozen deployment scale, not dynamic
batch absmax throughout training.

## 2026-06-01 Static Scale Calibration

Implemented as an experimental, default-off trainer option:

- `smartpaf_ss_calibrate`
- `smartpaf_ss_batches`
- `smartpaf_ss_max_samples`
- `smartpaf_ss_percentile`
- `smartpaf_ss_margin`

SS calibration samples StablePoly4 pre-activation tensors before CT/training and
writes each module's `static_absmax`. CT then fits coefficients using the same
static scale path that forward uses during training.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.gate_net_cmp.block_def import StablePoly4
from trainers.base_trainer import Trainer

model = nn.Sequential(StablePoly4(scale_mode='static'))
x = torch.linspace(-3, 3, steps=64).view(16, 1, 2, 2)
y = torch.zeros(16, dtype=torch.long)
loader = DataLoader(TensorDataset(x, y), batch_size=4)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=opt,
    device=torch.device('cpu'),
    result_dir='/tmp/smartpaf_ss_smoke',
    epochs=1,
    poly4_scale_mode='static',
    smartpaf_ss_calibrate=True,
    smartpaf_ss_batches=2,
    smartpaf_ct_init=True,
    smartpaf_ct_batches=2,
    smartpaf_ct_steps=2,
)
trainer._run_smartpaf_ss_calibration()
assert float(model[0].static_absmax) > 1.0
assert model[0].scale_mode == 'static'
PY
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_fast.status' < /dev/null &
```

Calibration and CT fit:

| Module | Static absmax | In scale | CT MSE |
| --- | ---: | ---: | ---: |
| `special_resnet.layers.0.act` | 6.28725 | 0.159052 | 0.0695138 |
| `special_resnet.layers.1.act` | 6.16644 | 0.162168 | 0.0404399 |

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | PASS |
| `imagenet100-96-pa-ct-ds-b256` | 16 | 44.58 | 42.98 | PASS |
| `imagenet100-96-pa-ds-b256` | 10 | 36.44 | 9.46 | COLLAPSE |

Conclusion: Static scale is the best scale-control variant so far. It is stable
and significantly better than dynamic scale, but still trails CT-only by 1.52
percentage points on this proxy. Keep SS experimental and prefer CT-only as the
current default unless deployment requires fixed polynomial input scaling.

## 2026-06-01 Delayed Alternate Training Attempts

Implemented a default-compatible AT scheduling option:

- `smartpaf_at_start_epoch`
- `smartpaf_at_start_epoch: auto` resolves to the first PA module start epoch
- Before that start epoch, AT keeps all parameters trainable instead of
  alternating between weights/poly phases while the model is still in warmup

Rationale: the previous CT+AT run alternated from epoch 1 even though no
polynomial branch contributed to the forward path until PA began. The delayed
schedule is closer to the SMART-PAF flow where AT is applied inside each
replacement/fine-tuning group, not during the initial warmup-only phase.

Validation command:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
```

Proxy commands:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_delayed_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_delayed_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_delayed_fast.status' < /dev/null &
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_delayed_poly01_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_delayed_poly01_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_delayed_poly01_fast.status' < /dev/null &
```

Result summary:

| Model | Poly LR mult | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-delayed-b256` | 0.5 | 10 | 31.28 | 20.88 | 10.40 | 0 | 0 | 1 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-delayed-poly01-b256` | 0.1 | 8 | 31.94 | 21.54 | 10.40 | 0 | 0 | 1 | COLLAPSE |

Observations:

- Delayed AT scheduling worked as intended: epochs before PA used phase `all`,
  then switched to `weights -> poly -> ...` starting at epoch 7.
- Both failures happened in `poly` phase, with no non-finite batches and no
  skipped optimizer steps.
- The polynomial input and derivative diagnostics stayed bounded:
  `|x_poly| max <= 0.7667` and `|f'| max <= 0.5053`.
- Reducing `poly_lr_mult` from 0.5 to 0.1 did not remove the first poly-phase
  validation drop; it only moved the failure from epoch 10 to epoch 8.

Comparison:

| Model | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | PASS |
| `imagenet100-96-pa-ct-ss-at-delayed-poly01-b256` | 8 | 31.94 | 21.54 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-delayed-b256` | 10 | 31.28 | 20.88 | COLLAPSE |

Conclusion: Delaying AT avoids wasting early warmup epochs, but AT is still not
stable in this implementation. The collapse is not caused by NaN/Inf or large
polynomial input scale; it is caused by the poly-only phase degrading validation
accuracy before the next weights phase can recover. Keep AT experimental. The
next AT-related change should move closer to the paper scheduler: per-layer
training groups with acceptance/recovery logic, SWA, and BN recalibration,
rather than global epoch-level alternation.

## 2026-06-01 BN Recalibration Attempt

Implemented a default-off post-training BatchNorm recalibration option:

- `bn_recalibrate_after_training`
- `bn_recalibrate_batches`
- `bn_recalibrate_use_best`

When enabled, the trainer optionally loads `best_model.pth`, resets BatchNorm
running statistics, forwards train batches without gradients, then validates
and appends a `smartpaf_phase=bn_recal` row to `train_history.csv`.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainers.base_trainer import Trainer

class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
    def forward(self, x):
        return self.net(x)

path = '/tmp/smartpaf_bn_recal_smoke'
shutil.rmtree(path, ignore_errors=True)
model = TinyBN()
x = torch.randn(16, 3, 8, 8)
y = torch.randint(0, 2, (16,))
loader = DataLoader(TensorDataset(x, y), batch_size=4)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    device=torch.device('cpu'),
    result_dir=path,
    epochs=1,
    save_checkpoints=False,
    bn_recalibrate_after_training=True,
    bn_recalibrate_batches=2,
    bn_recalibrate_use_best=False,
)
result = trainer._run_bn_recalibration()
assert result is not None
assert result['batches'] == 2
assert trainer.history['smartpaf_phase'][-1] == 'bn_recal'
assert len(trainer.history['epoch']) == 1
PY
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_bnrecal_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_bnrecal_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_bnrecal_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-bnrecal-b256` | 17 | 47.54 | 47.32 | 2.34 | 0 | 0 | 0 | PASS |

BN recalibration details:

| Pre-recal epoch 16 | BN-recal row | Delta | Batches | Time |
| ---: | ---: | ---: | ---: | ---: |
| 47.54 | 47.32 | -0.22 | 128 | 17.86s |

Comparison:

| Model | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | PASS |
| `imagenet100-96-pa-ct-ss-bnrecal-b256` | 17 | 47.54 | 47.32 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | PASS |

Conclusion: BN recalibration is implemented and stable, but it did not improve
this CT+SS proxy. The recalibrated row was 0.22 percentage points below the
pre-recal best. Keep BN recalibration default-off for now; it is most likely to
be useful after SWA or other weight-averaging techniques, where BN statistics
are known to become stale.

## 2026-06-01 SWA Attempt

Implemented default-off Stochastic Weight Averaging support:

- `swa_enabled`
- `swa_start_epoch`
- `swa_bn_update`
- `swa_bn_batches`

When enabled, the trainer keeps an `AveragedModel` from `swa_start_epoch`
through the end of training. After normal training, it optionally recalibrates
BatchNorm running statistics on the averaged model, validates it, saves
`swa_model.pth`, and appends a `smartpaf_phase=swa` row to `train_history.csv`.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainers.base_trainer import Trainer

class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
    def forward(self, x):
        return self.net(x)

path = '/tmp/smartpaf_swa_smoke'
shutil.rmtree(path, ignore_errors=True)
model = TinyBN()
x = torch.randn(16, 3, 8, 8)
y = torch.randint(0, 2, (16,))
loader = DataLoader(TensorDataset(x, y), batch_size=4)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    device=torch.device('cpu'),
    result_dir=path,
    epochs=2,
    save_checkpoints=False,
    swa_enabled=True,
    swa_start_epoch=1,
    swa_bn_update=True,
    swa_bn_batches=2,
)
trainer._maybe_update_swa_model(1)
trainer._maybe_update_swa_model(2)
result = trainer._run_swa_evaluation()
assert result is not None
assert result['updates'] == 2
assert trainer.history['smartpaf_phase'][-1] == 'swa'
assert trainer.history['train_valid_batches'][-1] == 2
PY
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_swa_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_swa_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_swa_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-swa-b256` | 17 | 47.66 | 44.68 | 2.98 | 0 | 0 | 0 | PASS |

SWA details:

| SWA start | Updates | BN update batches | Pre-SWA epoch 16 | SWA row | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 5 | 128 | 47.66 | 44.68 | -2.98 |

Comparison:

| Model | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | PASS |
| `imagenet100-96-pa-ct-ss-swa-b256` | 17 | 47.66 | 44.68 | PASS |
| `imagenet100-96-pa-ct-ss-bnrecal-b256` | 17 | 47.54 | 47.32 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | PASS |

Conclusion: SWA is implemented and stable, but this simple late-epoch averaging
does not improve the proxy. The averaged model is 2.98 percentage points below
the normal epoch-16 model even after BN update. Keep SWA default-off. A more
faithful SMART-PAF scheduler may still use SWA as a short recovery/acceptance
candidate inside per-layer training groups, but global late-epoch SWA is not a
current default.

## 2026-06-01 AT Paper-Order Recovery Attempt

Implemented a small AT scheduler option:

- `smartpaf_at_initial_phase: weights | poly`

The default remains `weights`, preserving existing behavior. The new `poly`
mode follows the SMART-PAF paper's AT order more closely by training PAF
coefficients first, then ordinary weights. This is still much simpler than the
paper's full scheduler, which performs per-layer training groups with validation
acceptance, SWA candidates, dropout-on-overfit, and target swapping only after
the current group stops improving.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
from trainers.base_trainer import Trainer
class T(Trainer):
    def __init__(self, initial):
        self.smartpaf_alternate_training = True
        self._smartpaf_poly_param_ids = {1}
        self._smartpaf_at_start_epoch = 7.0
        self.smartpaf_at_cycle_epochs = 1
        self.smartpaf_at_initial_phase = initial
for initial in ('weights', 'poly'):
    t = T(initial)
    print(initial, [t._current_smartpaf_phase(e) for e in range(6, 10)])
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_polyfirst_recover_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_polyfirst_recover_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_polyfirst_recover_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-polyfirst-recover-b256` | 16 | 34.94 | 31.68 | 32.06 | 0 | 0 | 5 | COLLAPSE |

Comparison:

| Model | Epochs | Best | Final | Guard | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-polyfirst-recover-b256` | 16 | 34.94 | 31.68 | 5 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-delayed-poly01-b256` | 8 | 31.94 | 21.54 | 1 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-delayed-b256` | 10 | 31.28 | 20.88 | 1 | COLLAPSE |

Phase details:

| Epoch | Phase | Val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly | 17.56 | 0 |
| 6 | weights | 25.40 | 0 |
| 7 | poly | 14.26 | 1 |
| 8 | weights | 34.94 | 0 |
| 9 | poly | 8.66 | 1 |
| 10 | weights | 34.34 | 0 |
| 11 | poly | 15.52 | 1 |
| 12 | weights | 34.46 | 0 |
| 13 | poly | 8.40 | 1 |
| 14 | weights | 33.10 | 0 |
| 15 | poly | 1.04 | 1 |
| 16 | weights | 31.68 | 0 |

Conclusion: paper-order AT plus `restore_best_reduce_lr` prevents a hard process
failure, but the poly-only phases still repeatedly collapse. The best recovered
accuracy, 34.94%, is slightly better than earlier CT+SS delayed AT attempts but
far below CT-only and CT+SS. Keep AT experimental. The next meaningful AT work
should move to the paper's per-layer training-group acceptance loop instead of
more global epoch alternation.

Remaining paper gaps after this run:

- Per-layer SMART-PAF scheduler: CT before a replacement, PA one operator at a
  time, training groups of `E` epochs, accept only the best validation/SWA model,
  and stop a step when no improvement remains.
- AT target scope: the paper alternates current PAF coefficients and related
  linear layers; the current implementation alternates all StablePoly4 params
  versus all non-poly params.
- Dropout-on-overfit: the paper triggers dropout when training accuracy exceeds
  validation accuracy by more than 10 percentage points.
- SWA inside each training group: global late-epoch SWA is implemented, but the
  paper uses SWA as a candidate after every group.
- PAF forms/degrees: this repo currently uses `StablePoly4`; the paper evaluates
  composed sign PAFs such as `f1^2 o g1^2`, `f2 o g3`, `f2 o g2`, and `f1 o g2`.
- Replacing MaxPooling and other non-polynomial operators: current proxies only
  exercise StablePoly4 activation replacement.
- DS-to-SS conversion workflow: DS and SS exist as modes, but the paper uses DS
  through fine-tuning and then converts the deployable model to SS.

## 2026-06-01 AT Active-Scope Recovery Attempt

Implemented default-off AT target scoping:

- `smartpaf_at_poly_scope: all | active`

The default remains `all`. When set to `active`, poly phases only train
StablePoly4 modules whose progressive PA schedule has reached their
`poly_start_epoch`. This is closer to the paper's current-layer AT target scope
than the earlier global all-poly alternation, though it still does not implement
the full training-group acceptance scheduler.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainers.base_trainer import Trainer

class Poly(nn.Module):
    def __init__(self, start):
        super().__init__()
        self.p = nn.Parameter(torch.ones(()))
        self.poly_start_epoch = float(start)
    def set_poly_schedule(self, start_epoch=None, transition_epochs=None):
        if start_epoch is not None:
            self.poly_start_epoch = float(start_epoch)
    def forward(self, x):
        return x * self.p

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Poly(2)
        self.b = Poly(4)
        self.fc = nn.Linear(1, 2)
    def forward(self, x):
        return self.fc(self.b(self.a(x)))

path = '/tmp/smartpaf_active_scope_smoke'
shutil.rmtree(path, ignore_errors=True)
model = Tiny()
loader = DataLoader(TensorDataset(torch.randn(4, 1), torch.zeros(4, dtype=torch.long)), batch_size=2)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    device=torch.device('cpu'),
    result_dir=path,
    epochs=4,
    save_checkpoints=False,
    smartpaf_alternate_training=True,
    smartpaf_at_start_epoch=2,
    smartpaf_at_initial_phase='poly',
    smartpaf_at_poly_scope='active',
    smartpaf_progressive=False,
)
model.a.poly_start_epoch = 2
model.b.poly_start_epoch = 4
ids2 = trainer._active_smartpaf_poly_param_ids(2)
ids4 = trainer._active_smartpaf_poly_param_ids(4)
assert id(model.a.p) in ids2 and id(model.b.p) not in ids2
assert id(model.a.p) in ids4 and id(model.b.p) in ids4
trainer._apply_smartpaf_training_mode(2)
assert model.a.p.requires_grad and not model.b.p.requires_grad
assert not model.fc.weight.requires_grad
trainer._apply_smartpaf_training_mode(3)
assert not model.a.p.requires_grad and not model.b.p.requires_grad
assert model.fc.weight.requires_grad
print('active scope ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_active_recover_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_active_recover_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_active_recover_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-active-recover-b256` | 16 | 37.38 | 35.64 | 36.12 | 0 | 0 | 4 | COLLAPSE |

Comparison:

| Model | Epochs | Best | Final | Guard | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-active-recover-b256` | 16 | 37.38 | 35.64 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-polyfirst-recover-b256` | 16 | 34.94 | 31.68 | 5 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-delayed-poly01-b256` | 8 | 31.94 | 21.54 | 1 | COLLAPSE |

Phase details:

| Epoch | Phase | Val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly | 17.78 | 0 |
| 6 | weights | 24.26 | 0 |
| 7 | poly | 15.06 | 0 |
| 8 | weights | 25.46 | 0 |
| 9 | poly | 10.52 | 1 |
| 10 | weights | 36.28 | 0 |
| 11 | poly | 17.68 | 1 |
| 12 | weights | 37.38 | 0 |
| 13 | poly | 10.28 | 1 |
| 14 | weights | 37.14 | 0 |
| 15 | poly | 1.02 | 1 |
| 16 | weights | 35.64 | 0 |

Conclusion: active-scope AT is a measurable improvement over global all-poly
AT. It delayed the first guard from epoch 7 to epoch 9, reduced guard hits from
5 to 4, and improved best accuracy from 34.94% to 37.38%. It still repeatedly
collapses on poly phases and remains far below CT-only. Keep it available as an
experimental option, but do not make AT a default. The next AT step should add
accept/reject semantics so a bad poly group is discarded immediately rather than
being measured as an epoch-level collapse.

## 2026-06-01 AT Reject/Revalidate Attempt

Implemented default-off rejected-phase revalidation:

- `smartpaf_revalidate_rejected_phase`

When collapse guard restores `best_model.pth` through
`collapse_guard_action: restore_best_reduce_lr`, this option immediately
revalidates the restored model and records the epoch as `poly_rejected` instead
of recording the collapsed candidate's validation result as the accepted epoch
result. The guard hit is still preserved in `collapse_guard_triggered`, so the
summary still marks this run as COLLAPSE.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainers.base_trainer import Trainer

path = '/tmp/smartpaf_reject_smoke'
shutil.rmtree(path, ignore_errors=True)
model = nn.Linear(2, 2)
loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    device=torch.device('cpu'),
    result_dir=path,
    epochs=1,
    save_checkpoints=True,
    collapse_guard_enabled=True,
    collapse_guard_drop=1.0,
    collapse_guard_action='restore_best_reduce_lr',
    smartpaf_revalidate_rejected_phase=True,
)
trainer.best_acc = 50.0
trainer.history['val_acc'].append(50.0)
trainer.save_checkpoint(1, is_best=True)
with torch.no_grad():
    for p in model.parameters():
        p.add_(10.0)
restored = trainer._run_collapse_guard(2, 10.0)
assert restored is True
assert trainer._last_collapse_guard_restored is True
assert trainer.optimizer.param_groups[0]['lr'] == 0.020000000000000004
print('reject restore ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_active_reject_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_active_reject_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_active_reject_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-active-reject-b256` | 16 | 38.10 | 35.58 | 9.64 | 0 | 0 | 4 | COLLAPSE |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-active-reject-b256` | 16 | 38.10 | 35.58 | 9.64 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-active-recover-b256` | 16 | 37.38 | 35.64 | 36.12 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-polyfirst-recover-b256` | 16 | 34.94 | 31.68 | 32.06 | 5 | COLLAPSE |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly | 18.14 | 0 |
| 6 | weights | 24.82 | 0 |
| 7 | poly | 15.18 | 0 |
| 8 | weights | 25.44 | 0 |
| 9 | poly_rejected | 25.44 | 1 |
| 10 | weights | 36.08 | 0 |
| 11 | poly_rejected | 36.08 | 1 |
| 12 | weights | 38.10 | 0 |
| 13 | poly_rejected | 38.10 | 1 |
| 14 | weights | 37.78 | 0 |
| 15 | poly_rejected | 38.10 | 1 |
| 16 | weights | 35.58 | 0 |

Conclusion: reject/revalidate moves AT behavior closer to the paper's
accept-only-good-candidates scheduler. It does not fix the underlying poly-phase
collapse, but it prevents rejected candidates from polluting epoch metrics:
`max_drop` falls from 36.12 to 9.64 and best accuracy rises slightly from 37.38
to 38.10. AT remains far below CT-only and should stay experimental. The next
useful step is a true per-layer training group that rolls back immediately after
each group and advances only when the restored or SWA candidate improves the
step best.

## 2026-06-01 AT Accept-Only Poly Attempt

Implemented a default-off AT acceptance option:

- `smartpaf_at_reject_nonimproving_poly`
- `smartpaf_at_accept_min_delta`
- `smartpaf_at_reject_lr_factor`

When enabled, an AT `poly` phase must beat the current best validation accuracy
by at least `smartpaf_at_accept_min_delta`. Otherwise the trainer restores
`best_model.pth`, optionally revalidates the restored model, and records the
phase as `poly_rejected`.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
python - <<'PY'
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainers.base_trainer import Trainer

path = '/tmp/smartpaf_accept_smoke'
shutil.rmtree(path, ignore_errors=True)
os.makedirs(path, exist_ok=True)
model = nn.Linear(2, 2)
loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    device=torch.device('cpu'),
    result_dir=path,
    epochs=1,
    save_checkpoints=True,
)
trainer.best_acc = 50.0
trainer.save_checkpoint(1, is_best=True)
with torch.no_grad():
    for p in model.parameters():
        p.add_(10.0)
restored = trainer._restore_best_and_scale_lr(1.0, 'smoke')
assert restored is True
assert trainer.optimizer.param_groups[0]['lr'] == 0.1
print('accept restore ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_accept_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_accept_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_accept_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-accept-b256` | 16 | 34.60 | 31.60 | 3.00 | 0 | 0 | 5 | COLLAPSE |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-active-reject-b256` | 16 | 38.10 | 35.58 | 9.64 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-active-recover-b256` | 16 | 37.38 | 35.64 | 36.12 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-polyfirst-recover-b256` | 16 | 34.94 | 31.68 | 32.06 | 5 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-accept-b256` | 16 | 34.60 | 31.60 | 3.00 | 5 | COLLAPSE |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 22.64 | 0 |
| 6 | weights | 24.60 | 0 |
| 7 | poly_rejected | 24.60 | 1 |
| 8 | weights | 34.60 | 0 |
| 9 | poly_rejected | 34.60 | 1 |
| 10 | weights | 34.56 | 0 |
| 11 | poly_rejected | 34.60 | 1 |
| 12 | weights | 34.12 | 0 |
| 13 | poly_rejected | 34.60 | 1 |
| 14 | weights | 33.32 | 0 |
| 15 | poly_rejected | 34.60 | 1 |
| 16 | weights | 31.60 | 0 |

Conclusion: accept-only poly rejection works mechanically and keeps recorded
validation metrics from showing the raw poly collapses. It is still worse than
the previous active reject/revalidate run because collapse guard runs before the
acceptance check. Large bad poly candidates are restored but first reduce LR by
`collapse_guard_lr_factor=0.2`, so later weights phases train with an overly
small learning rate. The next AT change should evaluate poly acceptance before
collapse guard or add a poly-specific reject path that restores without global
LR punishment.

Remaining paper techniques not yet faithfully applied:

- Per-layer SMART-PAF training groups with a local best/SWA candidate and
  immediate rollback after every group.
- Alternating the current PAF coefficients with only their related upstream
  linear layers, instead of all poly parameters versus all non-poly parameters.
- Dropout-on-overfit, gated by the train/validation accuracy gap.
- SWA inside each training group as an acceptance candidate.
- Paper PAF families and degrees (`f1^2 o g1^2`, `f2 o g3`, `f2 o g2`,
  `f1 o g2`, alpha-7), beyond this repo's current `StablePoly4`.
- Replacing MaxPooling and other non-polynomial operators, not only activation
  functions.
- DS fine-tuning followed by SS conversion as a deployment workflow.

## 2026-06-01 AT Pre-Guard Reject Attempt

Implemented a default-off AT option:

- `smartpaf_at_reject_before_collapse_guard`

When this is enabled together with `smartpaf_at_reject_nonimproving_poly`, a bad
AT `poly` candidate is rejected before collapse guard runs. The intended
behavior is to restore the previous best checkpoint and keep the scheduled LR
intact, instead of treating an exploratory poly-only phase as a global collapse.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_preguard_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_preguard_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_preguard_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-active-reject-b256` | 16 | 38.10 | 35.58 | 9.64 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-active-recover-b256` | 16 | 37.38 | 35.64 | 36.12 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-polyfirst-recover-b256` | 16 | 34.94 | 31.68 | 32.06 | 5 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-accept-b256` | 16 | 34.60 | 31.60 | 3.00 | 5 | COLLAPSE |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 21.76 | 0 |
| 6 | weights | 23.94 | 0 |
| 7 | poly_rejected | 23.94 | 0 |
| 8 | weights | 25.76 | 0 |
| 9 | poly_rejected | 25.76 | 0 |
| 10 | weights | 31.34 | 0 |
| 11 | poly_rejected | 31.34 | 0 |
| 12 | weights | 36.20 | 0 |
| 13 | poly_rejected | 36.20 | 0 |
| 14 | weights | 39.94 | 0 |
| 15 | poly_rejected | 39.94 | 0 |
| 16 | weights | 41.42 | 0 |

Conclusion: pre-guard rejection fixes the LR punishment issue found in the
accept-only run. All rejected poly phases were restored before collapse guard,
so `collapse_guard_triggered` stayed at 0 and the weights phases continued to
improve. This is the best AT variant so far, but it still trails CT-only by
7.52 percentage points and CT+SS by 6.00 points. AT should remain experimental.
The next paper-faithful step is a true per-layer training-group loop with local
best/SWA acceptance, instead of global epoch alternation.

## 2026-06-01 AT Dropout-on-Overfit Attempt

Implemented a default-off overfit-triggered dropout option:

- `smartpaf_at_dropout_on_overfit`
- `smartpaf_at_dropout_gap`
- `smartpaf_at_dropout_p`
- `smartpaf_overfit_dropout_active` history column

The implementation registers a forward pre-hook on the classifier `Linear`
layer and applies dropout to the classifier input only while the model is in
training mode and the overfit trigger is active. This avoids replacing `fc` with
a `Sequential`, so checkpoint key names remain unchanged.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainers.base_trainer import Trainer

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

model = Tiny()
loader = DataLoader(TensorDataset(torch.ones(4, 4), torch.zeros(4, dtype=torch.long)), batch_size=2)
keys_before = set(model.state_dict().keys())
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    device=torch.device('cpu'),
    result_dir='/tmp/smartpaf_dropout_smoke',
    epochs=1,
    save_checkpoints=False,
    smartpaf_alternate_training=True,
    smartpaf_at_dropout_on_overfit=True,
    smartpaf_at_dropout_gap=1.0,
    smartpaf_at_dropout_p=0.5,
)
assert set(model.state_dict().keys()) == keys_before
trainer._update_smartpaf_overfit_dropout(1, train_acc=60.0, val_acc=50.0)
assert trainer._smartpaf_overfit_dropout_active is True
model.train()
torch.manual_seed(0)
y_train = model(torch.ones(16, 4))
trainer._smartpaf_overfit_dropout_active = False
torch.manual_seed(0)
y_plain = model(torch.ones(16, 4))
assert not torch.allclose(y_train, y_plain)
model.eval()
trainer._smartpaf_overfit_dropout_active = True
torch.manual_seed(0)
y_eval = model(torch.ones(16, 4))
torch.manual_seed(1)
y_eval_2 = model(torch.ones(16, 4))
assert torch.allclose(y_eval, y_eval_2)
trainer._remove_smartpaf_overfit_dropout_hooks()
print('overfit dropout smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_dropout_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_dropout_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_dropout_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-dropout-b256` | 16 | 41.14 | 41.14 | 0.00 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-dropout-b256` | 16 | 41.14 | 41.14 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-active-reject-b256` | 16 | 38.10 | 35.58 | 9.64 | 4 | COLLAPSE |
| `imagenet100-96-pa-ct-ss-at-accept-b256` | 16 | 34.60 | 31.60 | 3.00 | 5 | COLLAPSE |

Phase and dropout details:

| Epoch | Phase | Recorded val acc | Dropout active | Guard |
| ---: | --- | ---: | ---: | ---: |
| 5 | poly_rejected | 23.46 | 0 | 0 |
| 6 | weights | 24.52 | 0 | 0 |
| 7 | poly_rejected | 24.52 | 0 | 0 |
| 8 | weights | 25.66 | 0 | 0 |
| 9 | poly_rejected | 25.66 | 1 | 0 |
| 10 | weights | 30.88 | 0 | 0 |
| 11 | poly_rejected | 30.88 | 1 | 0 |
| 12 | weights | 35.20 | 0 | 0 |
| 13 | poly_rejected | 35.20 | 1 | 0 |
| 14 | weights | 39.78 | 0 | 0 |
| 15 | poly_rejected | 39.78 | 0 | 0 |
| 16 | weights | 41.14 | 0 | 0 |

Conclusion: dropout-on-overfit is implemented and stable, and the trigger fired
as intended after epochs 8, 10, and 12. It did not improve this proxy: final
accuracy was 0.28 percentage points below the pre-guard AT baseline. Keep the
option available for paper-faithful per-layer training groups, but do not enable
it in the current global epoch alternation schedule.

## 2026-06-01 AT Active-Related Weight Scope Attempt

Implemented a default-off AT weight-scope option:

- `smartpaf_at_weight_scope: all | active_related`

`active_related` maps each StablePoly4 module to non-poly parameters in the same
parent block. For the current proxy model this means:

- `special_resnet.layers.0.act` -> `special_resnet.layers.0.conv*/bn*`
- `special_resnet.layers.1.act` -> `special_resnet.layers.1.conv*/bn*`

During weights phases, only related weights for currently active StablePoly4
modules are trainable. This moves the implementation closer to the paper's
current-PAF/related-layer AT target scope while keeping the option default-off.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.registry import get_model
from trainers.base_trainer import Trainer

model = get_model('resnet-basic-stablepoly4-layer1block1', num_classes=100, pretrained=False)
loader = DataLoader(TensorDataset(torch.randn(2, 3, 96, 96), torch.zeros(2, dtype=torch.long)), batch_size=1)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    device=torch.device('cpu'),
    result_dir='/tmp/smartpaf_related_scope_smoke',
    epochs=16,
    save_checkpoints=False,
    poly4_warmup_ratio=0.35,
    smartpaf_progressive=True,
    smartpaf_group_epochs='auto',
    smartpaf_transition_epochs=6,
    smartpaf_alternate_training=True,
    smartpaf_at_start_epoch='auto',
    smartpaf_at_initial_phase='poly',
    smartpaf_at_poly_scope='active',
    smartpaf_at_weight_scope='active_related',
)
trainer._apply_smartpaf_training_mode(6)
trainable6 = {name for name, p in model.named_parameters() if p.requires_grad}
assert 'special_resnet.layers.0.conv1.weight' in trainable6
assert 'special_resnet.layers.0.act.a' not in trainable6
assert 'special_resnet.layers.1.conv1.weight' not in trainable6
assert 'fc.weight' not in trainable6
trainer._apply_smartpaf_training_mode(12)
trainable12 = {name for name, p in model.named_parameters() if p.requires_grad}
assert 'special_resnet.layers.0.conv1.weight' in trainable12
assert 'special_resnet.layers.1.conv1.weight' in trainable12
assert 'special_resnet.layers.0.act.a' not in trainable12
assert 'special_resnet.layers.1.act.a' not in trainable12
assert 'fc.weight' not in trainable12
print('related weight scope smoke ok', len(trainable6), len(trainable12))
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_related_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_related_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_related_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-related-b256` | 16 | 27.00 | 25.04 | 1.96 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-dropout-b256` | 16 | 41.14 | 41.14 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-related-b256` | 16 | 27.00 | 25.04 | 1.96 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-accept-b256` | 16 | 34.60 | 31.60 | 3.00 | 5 | COLLAPSE |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 22.52 | 0 |
| 6 | weights | 27.00 | 0 |
| 7 | poly_rejected | 27.00 | 0 |
| 8 | weights | 26.68 | 0 |
| 9 | poly_rejected | 27.00 | 0 |
| 10 | weights | 25.36 | 0 |
| 11 | poly_rejected | 27.00 | 0 |
| 12 | weights | 25.94 | 0 |
| 13 | poly_rejected | 27.00 | 0 |
| 14 | weights | 26.60 | 0 |
| 15 | poly_rejected | 27.00 | 0 |
| 16 | weights | 25.04 | 0 |

Conclusion: active-related weight scope is mechanically correct and stable, but
it is a large negative result under global epoch alternation. The narrower
weights phase improves briefly at epoch 6 and then cannot keep recovering the
network; best accuracy is 14.42 percentage points below the pre-guard AT
baseline. Keep the option default-off. The paper's related-layer scope should
only be retried inside a true per-layer training-group loop where the rest of
the network is already a strong fixed context.

## 2026-06-01 AT 2-Epoch Group Attempt

Applied the existing AT group/cycle scheduler with:

- `smartpaf_at_cycle_epochs: 2`
- pre-guard poly rejection enabled
- active poly scope
- all non-poly weights during weights phases

This tests whether simply lengthening AT phases is a useful proxy for the
paper's training groups before implementing a full local best/SWA acceptance
loop.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ss_at_group2_fast.yaml') as f:
    data = yaml.safe_load(f)
assert data['models'][0]['trainer_kwargs']['smartpaf_at_cycle_epochs'] == 2
print(data['models'][0]['name'])
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-b256` | 16 | 38.76 | 38.76 | 0.34 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-dropout-b256` | 16 | 41.14 | 41.14 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-b256` | 16 | 38.76 | 38.76 | 0.34 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-related-b256` | 16 | 27.00 | 25.04 | 1.96 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 23.20 | 0 |
| 6 | poly_rejected | 23.20 | 0 |
| 7 | weights | 27.02 | 0 |
| 8 | weights | 26.68 | 0 |
| 9 | poly_rejected | 27.02 | 0 |
| 10 | poly_rejected | 27.02 | 0 |
| 11 | weights | 30.62 | 0 |
| 12 | weights | 32.24 | 0 |
| 13 | poly_rejected | 32.24 | 0 |
| 14 | poly_rejected | 32.24 | 0 |
| 15 | weights | 36.56 | 0 |
| 16 | weights | 38.76 | 0 |

Conclusion: 2-epoch AT groups are stable with pre-guard rejection but worse than
single-epoch alternation. The consecutive poly epochs are repeatedly rejected
and consume training budget before weights can recover. This confirms that
longer cycles alone are not the paper's training-group method; the missing
piece is local group best/SWA acceptance and early stopping inside each group.

## 2026-06-01 AT 2-Epoch Group Skip Attempt

Added default-off `smartpaf_at_skip_rejected_poly_group`. When a poly epoch is
rejected inside a multi-epoch AT phase group, the remaining epochs in that same
scheduled poly group are switched to weights. This keeps the existing global AT
proxy but avoids spending a second epoch on a poly-only phase that has already
failed the accept check.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ss_at_group2_skip_fast.yaml') as f:
    data = yaml.safe_load(f)
kw = data['models'][0]['trainer_kwargs']
assert kw['smartpaf_at_cycle_epochs'] == 2
assert kw['smartpaf_at_skip_rejected_poly_group'] is True
print(data['models'][0]['name'])
PY
.venv/bin/python - <<'PY'
from trainers.base_trainer import Trainer
t = Trainer.__new__(Trainer)
t.smartpaf_alternate_training = True
t.smartpaf_at_initial_phase = 'poly'
t.smartpaf_at_cycle_epochs = 2
t._smartpaf_at_start_epoch = 5
t.smartpaf_at_skip_rejected_poly_group = True
t._smartpaf_skipped_poly_phase_idx = None
assert t._is_smartpaf_poly_phase(5)
assert t._is_smartpaf_poly_phase(6)
t._mark_rejected_poly_group(5)
assert not t._is_smartpaf_poly_phase(6)
assert not t._is_smartpaf_poly_phase(7)
assert t._is_smartpaf_poly_phase(9)
print('phase smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_skip_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_skip_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_skip_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-skip-b256` | 16 | 43.52 | 43.52 | 0.58 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-skip-b256` | 16 | 43.52 | 43.52 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-dropout-b256` | 16 | 41.14 | 41.14 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-b256` | 16 | 38.76 | 38.76 | 0.34 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-related-b256` | 16 | 27.00 | 25.04 | 1.96 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 23.48 | 0 |
| 6 | weights | 24.14 | 0 |
| 7 | weights | 28.48 | 0 |
| 8 | weights | 28.24 | 0 |
| 9 | poly_rejected | 28.48 | 0 |
| 10 | weights | 30.08 | 0 |
| 11 | weights | 35.02 | 0 |
| 12 | weights | 37.28 | 0 |
| 13 | poly_rejected | 37.28 | 0 |
| 14 | weights | 41.10 | 0 |
| 15 | weights | 40.52 | 0 |
| 16 | weights | 43.52 | 0 |

Conclusion: skip-after-reject is a meaningful positive step for AT. It improves
the AT proxy from 38.76 with plain 2-epoch groups and 41.42 with single-epoch
pre-guard to 43.52, while keeping nonfinite and guard counts at zero. It still
trails CT+SS by 3.90 points and CT-only by 5.42 points, so this should remain
default-off. The next AT step should be closer to the paper's training-group
logic: keep a local group best/SWA candidate, accept only improving poly
updates, and stop the group dynamically when it stops helping.

## 2026-06-01 AT Dynamic Stop Attempt

Added default-off `smartpaf_at_stop_after_rejected_poly_groups`. This is a
small proxy for the SMART-PAF scheduler's group-level stop behavior: after a
configured number of rejected poly groups, future AT poly phases are disabled
and the remaining training budget goes to weights. The proxy config sets the
threshold to `1`, because the previous runs showed every later poly group was
rejected and only consumed epochs.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_fast.yaml') as f:
    data = yaml.safe_load(f)
kw = data['models'][0]['trainer_kwargs']
assert kw['smartpaf_at_cycle_epochs'] == 2
assert kw['smartpaf_at_skip_rejected_poly_group'] is True
assert kw['smartpaf_at_stop_after_rejected_poly_groups'] == 1
print(data['models'][0]['name'])
PY
.venv/bin/python - <<'PY'
from trainers.base_trainer import Trainer
t = Trainer.__new__(Trainer)
t.smartpaf_alternate_training = True
t.smartpaf_at_initial_phase = 'poly'
t.smartpaf_at_cycle_epochs = 2
t._smartpaf_at_start_epoch = 5
t.smartpaf_at_skip_rejected_poly_group = True
t.smartpaf_at_stop_after_rejected_poly_groups = 1
t._smartpaf_skipped_poly_phase_idx = None
t._smartpaf_last_rejected_poly_phase_idx = None
t._smartpaf_consecutive_rejected_poly_groups = 0
t._smartpaf_poly_stopped_after_rejections = False
assert t._is_smartpaf_poly_phase(5)
assert t._is_smartpaf_poly_phase(6)
t._mark_rejected_poly_group(5)
assert not t._is_smartpaf_poly_phase(6)
assert not t._is_smartpaf_poly_phase(9)
print('stop phase smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_fast.status' < /dev/null &
```

Result summary:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-stop-b256` | 16 | 46.64 | 46.64 | 2.42 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Epochs | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-b256` | 16 | 46.64 | 46.64 | 2.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-skip-b256` | 16 | 43.52 | 43.52 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-dropout-b256` | 16 | 41.14 | 41.14 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-b256` | 16 | 38.76 | 38.76 | 0.34 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-related-b256` | 16 | 27.00 | 25.04 | 1.96 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 22.30 | 0 |
| 6 | weights | 25.12 | 0 |
| 7 | weights | 28.86 | 0 |
| 8 | weights | 26.84 | 0 |
| 9 | weights | 30.18 | 0 |
| 10 | weights | 33.92 | 0 |
| 11 | weights | 38.84 | 0 |
| 12 | weights | 40.38 | 0 |
| 13 | weights | 44.18 | 0 |
| 14 | weights | 46.46 | 0 |
| 15 | weights | 44.04 | 0 |
| 16 | weights | 46.64 | 0 |

Conclusion: dynamic stop is the strongest AT proxy so far. It improves the
best AT result from 43.52 to 46.64 by avoiding repeated rejected poly phases,
leaving only a 0.78 point gap to CT+SS and a 2.30 point gap to CT-only. This
also confirms why earlier AT variants underperformed: the current PAF poly
updates are not useful in this global epoch proxy, and the scheduler must stop
spending epochs on them once validation rejects a group. Keep the option
default-off; the remaining paper-faithful step is still a true per-layer group
loop with local best/SWA candidate restoration instead of global epoch
alternation.

## 2026-06-01 DS-to-SS Deployment Conversion Attempt

Implemented default-off `smartpaf_ds_to_ss_after_training` and
`smartpaf_ds_to_ss_use_best`. This adds the paper-style deployment step for
dynamic scale runs: after DS training, optionally load `best_model.pth`, copy
each dynamic StablePoly4 module's `running_absmax` into `static_absmax`, switch
the module to static scale, then validate and save `ds_to_ss_model.pth`.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ds_to_ss_fast.yaml') as f:
    data = yaml.safe_load(f)
kw = data['models'][0]['trainer_kwargs']
assert kw['poly4_scale_mode'] == 'dynamic'
assert kw['smartpaf_ds_to_ss_after_training'] is True
assert kw['smartpaf_ds_to_ss_use_best'] is True
print(data['models'][0]['name'])
PY
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.gate_net_cmp.block_def import StablePoly4
from trainers.base_trainer import Trainer

model = nn.Sequential(StablePoly4(scale_mode='dynamic'))
x = torch.randn(24, 3)
y = torch.arange(24) % 3
loader = DataLoader(TensorDataset(x, y), batch_size=6)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
trainer = Trainer(
    model=model,
    train_loader=loader,
    val_loader=loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=opt,
    device=torch.device('cpu'),
    result_dir='/tmp/smartpaf_ds_to_ss_smoke',
    epochs=1,
    save_checkpoints=False,
    poly4_scale_mode='dynamic',
    smartpaf_ds_to_ss_after_training=True,
    smartpaf_ds_to_ss_use_best=False,
)
model.train()
with torch.no_grad():
    model(torch.randn(8, 3) * 4)
result = trainer._run_smartpaf_ds_to_ss_evaluation()
assert result is not None
assert model[0].scale_mode == 'static'
assert float(model[0].static_absmax) > 1.0
assert trainer.history['smartpaf_phase'][-1] == 'ds_to_ss'
print('ds_to_ss smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ds_to_ss_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ds_to_ss_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ds_to_ss_fast.status' < /dev/null &
```

CT and DS->SS conversion details:

| Module | CT MSE | DS->SS static absmax | In scale |
| --- | ---: | ---: | ---: |
| `special_resnet.layers.0.act` | 0.0571138 | 8.70961 | 0.114816 |
| `special_resnet.layers.1.act` | 0.0326864 | 6.96405 | 0.143595 |

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ds-to-ss-b256` | 17 | 44.42 | 44.42 | 3.54 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ds-b256` | 16 | 44.58 | 42.98 | 5.52 | 0 | PASS |
| `imagenet100-96-pa-ct-ds-to-ss-b256` | 17 | 44.42 | 44.42 | 3.54 | 0 | PASS |

Tail rows:

| Epoch/Row | Phase | Val acc | Val loss |
| ---: | --- | ---: | ---: |
| 13 | disabled | 42.48 | 2.5357 |
| 14 | disabled | 44.42 | 2.4441 |
| 15 | disabled | 40.88 | 2.6122 |
| 16 | disabled | 43.00 | 2.5146 |
| 17 | ds_to_ss | 44.42 | 2.4441 |

Conclusion: DS-to-SS conversion works mechanically and preserves the best DS
checkpoint exactly on this proxy: epoch 14 dynamic validation and the converted
static validation are both 44.42%. This confirms the deployability path, but it
does not fix DS's accuracy gap: direct pre-calibrated SS remains 3.00 points
higher, and CT-only remains 4.52 points higher. Keep DS-to-SS default-off and
available for deployment-style runs; current accuracy work should continue to
focus on the per-layer group scheduler and PAF family choices rather than scale
conversion alone.

## 2026-06-01 AT Phase Best Restore Attempt

Implemented default-off `smartpaf_at_restore_phase_best` and
`smartpaf_at_restore_phase_min_delta`. This approximates another part of the
SMART-PAF training-group scheduler: at the start of each AT phase group, the
trainer snapshots the current model on CPU; within the group it tracks the
validation-best model; at the next group boundary it restores the group best if
it improves over the group start, otherwise it restores the group start. A final
`phase_restore` validation row records the restored last group model.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ss_at_group2_phasebest_fast.yaml') as f:
    data = yaml.safe_load(f)
kw = data['models'][0]['trainer_kwargs']
assert kw['smartpaf_at_cycle_epochs'] == 2
assert kw['smartpaf_at_restore_phase_best'] is True
assert kw['smartpaf_at_stop_after_rejected_poly_groups'] == 1
print(data['models'][0]['name'])
PY
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from trainers.base_trainer import Trainer
model = nn.Linear(1, 1, bias=False)
trainer = Trainer.__new__(Trainer)
trainer.model = model
trainer.resume_strict = True
trainer.smartpaf_alternate_training = True
trainer.smartpaf_at_initial_phase = 'poly'
trainer.smartpaf_at_cycle_epochs = 2
trainer._smartpaf_at_start_epoch = 5
trainer._smartpaf_poly_stopped_after_rejections = False
trainer._smartpaf_poly_param_ids = {id(model.weight)}
trainer.smartpaf_at_skip_rejected_poly_group = False
trainer._smartpaf_skipped_poly_phase_idx = None
trainer.smartpaf_at_restore_phase_best = True
trainer.smartpaf_at_restore_phase_min_delta = 0.0
trainer.best_acc = 20.0
trainer.history = {'val_acc': [20.0]}
trainer._smartpaf_restore_phase_idx = None
trainer._smartpaf_restore_phase_label = None
trainer._smartpaf_restore_phase_start_acc = None
trainer._smartpaf_restore_phase_start_state = None
trainer._smartpaf_restore_phase_best_acc = None
trainer._smartpaf_restore_phase_best_epoch = None
trainer._smartpaf_restore_phase_best_state = None
trainer._smartpaf_restore_phase_last_restored_acc = None
with torch.no_grad():
    model.weight.fill_(1.0)
trainer._prepare_smartpaf_phase_restore_group(5)
with torch.no_grad():
    model.weight.fill_(2.0)
trainer._update_smartpaf_phase_restore_group(5, 22.0)
with torch.no_grad():
    model.weight.fill_(3.0)
trainer._update_smartpaf_phase_restore_group(6, 21.0)
trainer.history['val_acc'].append(21.0)
trainer._prepare_smartpaf_phase_restore_group(7)
assert abs(float(model.weight.item()) - 2.0) < 1e-6
assert trainer._smartpaf_restore_phase_idx == 1
assert abs(trainer._smartpaf_restore_phase_start_acc - 22.0) < 1e-6
print('phase restore smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_phasebest_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_phasebest_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_phasebest_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-phasebest-b256` | 17 | 45.34 | 45.34 | 1.36 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-b256` | 16 | 46.64 | 46.64 | 2.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-phasebest-b256` | 17 | 45.34 | 45.34 | 1.36 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-skip-b256` | 16 | 43.52 | 43.52 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |

Phase details:

| Epoch/Row | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 23.10 | 0 |
| 6 | weights | 24.52 | 0 |
| 7 | weights | 29.54 | 0 |
| 8 | weights | 28.98 | 0 |
| 9 | weights | 30.08 | 0 |
| 10 | weights | 32.50 | 0 |
| 11 | weights | 37.08 | 0 |
| 12 | weights | 38.92 | 0 |
| 13 | weights | 41.74 | 0 |
| 14 | weights | 44.28 | 0 |
| 15 | weights | 42.92 | 0 |
| 16 | weights | 45.34 | 0 |
| 17 | phase_restore | 45.34 | 0 |

Group restore log summary:

| Group | Phase | Start | Best | Restored |
| ---: | --- | ---: | ---: | --- |
| 0 | poly | 23.10 | 24.52 | epoch 6 |
| 1 | weights | 24.52 | 29.54 | epoch 7 |
| 2 | weights | 29.54 | 32.50 | epoch 10 |
| 3 | weights | 32.50 | 38.92 | epoch 12 |
| 4 | weights | 38.92 | 44.28 | epoch 14 |
| 5 | weights | 44.28 | 45.34 | epoch 16 |

Conclusion: phase-local best restoration is mechanically correct and stable,
and it reduces max drop from the dynamic-stop run's 2.42 to 1.36 points. It is
nevertheless a negative accuracy result on this proxy: best/final accuracy is
45.34, which is 1.30 points below `group2-stop` and 2.08 points below CT+SS.
Restoring every two-epoch weights group appears too conservative here; the
better AT proxy remains dynamic poly stop plus uninterrupted weights recovery.
Keep phase restore default-off. A more faithful next step should make this
per-layer and compare a local SWA candidate, rather than applying short global
phase restores.

## 2026-06-01 AT Phase SWA Candidate Attempt

Added default-off `smartpaf_at_phase_swa`. When phase-local restoration is
enabled, each AT phase group now keeps an `AveragedModel` over the group epochs.
At the group boundary, the trainer validates the SWA candidate and restores it
if it beats the ordinary group best. This is a direct proxy for the SMART-PAF
paper's "best or SWA candidate" training-group acceptance rule.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ss_at_group2_phaseswa_fast.yaml') as f:
    data = yaml.safe_load(f)
kw = data['models'][0]['trainer_kwargs']
assert kw['smartpaf_at_restore_phase_best'] is True
assert kw['smartpaf_at_phase_swa'] is True
assert kw['smartpaf_at_stop_after_rejected_poly_groups'] == 1
print(data['models'][0]['name'])
PY
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from trainers.base_trainer import Trainer
model = nn.Linear(1, 1, bias=False)
trainer = Trainer.__new__(Trainer)
trainer.model = model
trainer.device = torch.device('cpu')
trainer.resume_strict = True
trainer.smartpaf_alternate_training = True
trainer.smartpaf_at_initial_phase = 'poly'
trainer.smartpaf_at_cycle_epochs = 2
trainer._smartpaf_at_start_epoch = 5
trainer._smartpaf_poly_stopped_after_rejections = False
trainer._smartpaf_poly_param_ids = {id(model.weight)}
trainer.smartpaf_at_skip_rejected_poly_group = False
trainer._smartpaf_skipped_poly_phase_idx = None
trainer.smartpaf_at_restore_phase_best = True
trainer.smartpaf_at_restore_phase_min_delta = 0.0
trainer.smartpaf_at_phase_swa = True
trainer.best_acc = 20.0
trainer.history = {'val_acc': [20.0]}
trainer._smartpaf_restore_phase_idx = None
trainer._smartpaf_restore_phase_label = None
trainer._smartpaf_restore_phase_start_acc = None
trainer._smartpaf_restore_phase_start_state = None
trainer._smartpaf_restore_phase_best_acc = None
trainer._smartpaf_restore_phase_best_epoch = None
trainer._smartpaf_restore_phase_best_state = None
trainer._smartpaf_restore_phase_last_restored_acc = None
trainer._smartpaf_restore_phase_swa_model = None
trainer._smartpaf_restore_phase_swa_updates = 0
trainer._smartpaf_restore_phase_swa_acc = None
trainer._validate_with_model = lambda model, epoch: (0.5, 23.0)
with torch.no_grad():
    model.weight.fill_(1.0)
trainer._prepare_smartpaf_phase_restore_group(5)
with torch.no_grad():
    model.weight.fill_(2.0)
trainer._update_smartpaf_phase_restore_group(5, 22.0)
with torch.no_grad():
    model.weight.fill_(4.0)
trainer._update_smartpaf_phase_restore_group(6, 21.0)
result = trainer._finalize_smartpaf_phase_restore_group()
assert result['best_source'] == 'swa'
assert result['swa_updates'] == 2
assert abs(float(model.weight.item()) - 3.0) < 1e-6
print('phase swa smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_phaseswa_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_phaseswa_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_phaseswa_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-phaseswa-b256` | 17 | 46.18 | 46.18 | 1.50 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-b256` | 16 | 46.64 | 46.64 | 2.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-phaseswa-b256` | 17 | 46.18 | 46.18 | 1.50 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-phasebest-b256` | 17 | 45.34 | 45.34 | 1.36 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-skip-b256` | 16 | 43.52 | 43.52 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-preguard-b256` | 16 | 41.42 | 41.42 | 0.00 | 0 | PASS |

Phase details:

| Epoch/Row | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 22.82 | 0 |
| 6 | weights | 24.26 | 0 |
| 7 | weights | 28.86 | 0 |
| 8 | weights | 27.36 | 0 |
| 9 | weights | 31.22 | 0 |
| 10 | weights | 34.36 | 0 |
| 11 | weights | 37.76 | 0 |
| 12 | weights | 39.64 | 0 |
| 13 | weights | 43.82 | 0 |
| 14 | weights | 45.26 | 0 |
| 15 | weights | 43.82 | 0 |
| 16 | weights | 46.18 | 0 |
| 17 | phase_restore | 46.18 | 0 |

Group candidate summary:

| Group | Phase | Start | Ordinary best | SWA acc | Restored |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | poly | 22.82 | 24.26 | 17.32 | epoch 6 |
| 1 | weights | 24.26 | 28.86 | 31.56 | SWA |
| 2 | weights | 31.56 | 34.36 | 27.52 | epoch 10 |
| 3 | weights | 34.36 | 39.64 | 35.46 | epoch 12 |
| 4 | weights | 39.64 | 45.26 | 44.08 | epoch 14 |
| 5 | weights | 45.26 | 46.18 | 44.84 | epoch 16 |

Conclusion: local phase SWA works and gives a positive delta over phase-best
restore, improving best/final from 45.34 to 46.18. The SWA candidate was useful
in group 1, where it recovered 31.56% versus the ordinary group's 28.86%.
However, it still trails the simpler dynamic-stop proxy by 0.46 points and
CT+SS by 1.24 points. Keep phase SWA default-off for global AT; it remains a
good component for the future true per-layer training-group scheduler, where
SWA candidates are evaluated around a single replacement rather than repeated
global weights-only recovery phases.

## 2026-06-01 AT Poly-Scoped Phase SWA Attempt

Added default-off `smartpaf_at_restore_phase_scope` with allowed values
`all`, `poly`, and `weights`. This keeps the existing phase-local best/SWA
machinery, but lets a proxy apply it only where it matches the experiment. The
new proxy scopes phase restore/SWA to `poly` groups only: the rejected or
accepted PAF coefficient group still gets a local best/SWA candidate, while
later weights recovery is not interrupted by short global phase restores.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
with open('configs/proxy_imagenet100_96_pa_ct_ss_at_group2_polyswa_fast.yaml') as f:
    data = yaml.safe_load(f)
kw = data['models'][0]['trainer_kwargs']
assert kw['smartpaf_at_restore_phase_best'] is True
assert kw['smartpaf_at_restore_phase_scope'] == 'poly'
assert kw['smartpaf_at_phase_swa'] is True
assert kw['smartpaf_at_stop_after_rejected_poly_groups'] == 1
print(data['models'][0]['name'])
PY
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from trainers.base_trainer import Trainer

model = nn.Linear(1, 1, bias=False)
trainer = Trainer.__new__(Trainer)
trainer.model = model
trainer.device = torch.device('cpu')
trainer.resume_strict = True
trainer.smartpaf_alternate_training = True
trainer.smartpaf_at_initial_phase = 'poly'
trainer.smartpaf_at_cycle_epochs = 2
trainer._smartpaf_at_start_epoch = 5
trainer._smartpaf_poly_stopped_after_rejections = False
trainer._smartpaf_poly_param_ids = {id(model.weight)}
trainer.smartpaf_at_skip_rejected_poly_group = False
trainer._smartpaf_skipped_poly_phase_idx = None
trainer.smartpaf_at_restore_phase_best = True
trainer.smartpaf_at_restore_phase_min_delta = 0.0
trainer.smartpaf_at_restore_phase_scope = 'poly'
trainer.smartpaf_at_phase_swa = True
trainer.best_acc = 20.0
trainer.history = {'val_acc': [20.0]}
trainer._smartpaf_restore_phase_idx = None
trainer._smartpaf_restore_phase_label = None
trainer._smartpaf_restore_phase_start_acc = None
trainer._smartpaf_restore_phase_start_state = None
trainer._smartpaf_restore_phase_best_acc = None
trainer._smartpaf_restore_phase_best_epoch = None
trainer._smartpaf_restore_phase_best_state = None
trainer._smartpaf_restore_phase_last_restored_acc = None
trainer._smartpaf_restore_phase_swa_model = None
trainer._smartpaf_restore_phase_swa_updates = 0
trainer._smartpaf_restore_phase_swa_acc = None
trainer._validate_with_model = lambda model, epoch: (0.5, 23.0)

with torch.no_grad():
    model.weight.fill_(1.0)
trainer._prepare_smartpaf_phase_restore_group(5)
with torch.no_grad():
    model.weight.fill_(2.0)
trainer._update_smartpaf_phase_restore_group(5, 22.0)
with torch.no_grad():
    model.weight.fill_(4.0)
trainer._update_smartpaf_phase_restore_group(6, 21.0)
result = trainer._prepare_smartpaf_phase_restore_group(7)
assert result['best_source'] == 'swa'
assert result['swa_updates'] == 2
assert abs(float(model.weight.item()) - 3.0) < 1e-6
assert trainer._smartpaf_restore_phase_idx is None
trainer._prepare_smartpaf_phase_restore_group(9)
assert trainer._smartpaf_restore_phase_idx == 2
print('poly-scoped phase restore smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_polyswa_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_polyswa_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_polyswa_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-polyswa-b256` | 16 | 46.54 | 46.54 | 1.66 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-b256` | 16 | 46.64 | 46.64 | 2.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-polyswa-b256` | 16 | 46.54 | 46.54 | 1.66 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-phaseswa-b256` | 17 | 46.18 | 46.18 | 1.50 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-phasebest-b256` | 17 | 45.34 | 45.34 | 1.36 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-skip-b256` | 16 | 43.52 | 43.52 | 0.58 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 23.00 | 0 |
| 6 | weights | 24.76 | 0 |
| 7 | weights | 28.06 | 0 |
| 8 | weights | 29.02 | 0 |
| 9 | weights | 30.30 | 0 |
| 10 | weights | 34.44 | 0 |
| 11 | weights | 37.34 | 0 |
| 12 | weights | 41.94 | 0 |
| 13 | weights | 44.72 | 0 |
| 14 | weights | 46.08 | 0 |
| 15 | weights | 44.42 | 0 |
| 16 | weights | 46.54 | 0 |

Group candidate summary:

| Group | Phase | Start | Ordinary best | SWA acc | Restored |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | poly | 23.00 | 24.76 | 17.58 | epoch 6 |

Conclusion: poly-scoped phase restore/SWA is the strongest phase-restore AT
variant so far. It avoids the negative effect seen when restoring every weights
group, improving from phase-SWA's 46.18 to 46.54 and almost matching the simpler
dynamic-stop proxy at 46.64. It still does not beat CT+SS and still trails the
normal Swish baseline. Keep it default-off; the evidence says the current global
AT proxy's only useful behavior is rejecting the first bad poly candidate and
then letting weights recover uninterrupted.

Paper techniques mentioned locally but still not faithfully applied:

- True per-layer training groups that replace exactly one non-polynomial layer,
  run a local convergence loop, then advance in inference order.
- AT scoped to the current replacement and its directly related linear layers,
  instead of global all-poly versus global non-poly phases.
- Full paper PAF family search and task-specific selection across `a7`,
  `2f12g1`, `f2g3`, `f2g2`, and `f1g2`; current proxies mostly exercise one
  StablePoly4-style family.
- Replacing all non-polynomial operators, especially MaxPooling, not only the
  activation modules currently covered by the proxy model.
- Deployment-oriented FHE latency/depth evaluation for the selected PAF family,
  rather than accuracy-only proxy training.

## 2026-06-01 AT Dynamic Stop + BN Recalibration Attempt

Added a proxy configuration that combines the strongest current AT path,
dynamic poly stop, with post-training BatchNorm recalibration. The run loads the
best checkpoint after normal training, recomputes BatchNorm running statistics
with 128 train batches, then validates and records a `bn_recal` row.

Validation commands:

```bash
.venv/bin/python - <<'PY'
import yaml
p = 'configs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_bnrecal_fast.yaml'
with open(p) as f:
    data = yaml.safe_load(f)
model = data['models'][0]
kw = model['trainer_kwargs']
assert model['name'] == 'imagenet100-96-pa-ct-ss-at-group2-stop-bnrecal-b256'
assert kw['smartpaf_alternate_training'] is True
assert kw['smartpaf_at_stop_after_rejected_poly_groups'] == 1
assert kw['bn_recalibrate_after_training'] is True
assert kw['bn_recalibrate_batches'] == 128
assert kw['bn_recalibrate_use_best'] is True
print(model['name'])
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_bnrecal_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_bnrecal_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_at_group2_stop_bnrecal_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-at-group2-stop-bnrecal-b256` | 17 | 46.76 | 46.56 | 1.52 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-bnrecal-b256` | 17 | 46.76 | 46.56 | 1.52 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-b256` | 16 | 46.64 | 46.64 | 2.42 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-polyswa-b256` | 16 | 46.54 | 46.54 | 1.66 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-phaseswa-b256` | 17 | 46.18 | 46.18 | 1.50 | 0 | PASS |

Phase details:

| Epoch/Row | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 5 | poly_rejected | 22.06 | 0 |
| 6 | weights | 23.62 | 0 |
| 7 | weights | 29.42 | 0 |
| 8 | weights | 28.66 | 0 |
| 9 | weights | 33.42 | 0 |
| 10 | weights | 35.52 | 0 |
| 11 | weights | 38.92 | 0 |
| 12 | weights | 39.26 | 0 |
| 13 | weights | 43.66 | 0 |
| 14 | weights | 45.94 | 0 |
| 15 | weights | 44.42 | 0 |
| 16 | weights | 46.76 | 0 |
| 17 | bn_recal | 46.56 | 0 |

Log highlights:

| Event | Value |
| --- | --- |
| Rejected poly candidate | 18.00 vs best 22.06 |
| Future poly groups stopped | after 1 rejected group |
| BN recalibration batches | 128 |
| BN recalibrated validation | 46.56 |

Conclusion: this run sets a new best AT proxy accuracy at 46.76, slightly above
the previous dynamic-stop result of 46.64. The added BN recalibration itself did
not improve the best checkpoint: recalibrated validation was 46.56, 0.20 points
below the normal epoch-16 best. Keep BN recalibration default-off for this AT
path. The useful behavior remains pre-guard poly rejection plus dynamic stop;
the next accuracy work should focus on the remaining paper-faithful gaps,
especially per-layer training groups or PAF family selection, rather than
expecting BN recalibration to recover the gap to CT+SS.

## 2026-06-01 ReLU-Targeted CT Attempt

Added default-off `smartpaf_ct_target` so coefficient tuning can fit different
local replacement targets instead of always fitting each module's warmup
activation. Supported targets are `warmup`, `relu`, `swish`/`silu`, `sigmoid`,
`tanh`, and `identity`. This is a small step toward task-specific PAF selection:
the training loop can now evaluate whether fitting the original ReLU replacement
target is better than fitting the warmup/Swish target.

This proxy keeps the rest of the CT+SS setup unchanged and sets
`smartpaf_ct_target: relu`.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
p = 'configs/proxy_imagenet100_96_pa_ct_ss_ctrelu_fast.yaml'
with open(p) as f:
    data = yaml.safe_load(f)
model = data['models'][0]
kw = model['trainer_kwargs']
assert model['name'] == 'imagenet100-96-pa-ctrelu-ss-b256'
assert kw['smartpaf_ct_init'] is True
assert kw['smartpaf_ct_target'] == 'relu'
assert kw['smartpaf_ss_calibrate'] is True
assert kw['smartpaf_alternate_training'] is False
print(model['name'])
PY
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from trainers.base_trainer import Trainer

class DummyPoly(nn.Module):
    def __init__(self):
        super().__init__()
        self.warmup_act = nn.SiLU()

trainer = Trainer.__new__(Trainer)
module = DummyPoly()
x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
trainer.smartpaf_ct_target = 'relu'
assert torch.equal(trainer._smartpaf_ct_target_eval(module, x), torch.relu(x))
trainer.smartpaf_ct_target = 'warmup'
assert torch.allclose(trainer._smartpaf_ct_target_eval(module, x), torch.nn.functional.silu(x))
trainer.smartpaf_ct_target = 'identity'
assert torch.equal(trainer._smartpaf_ct_target_eval(module, x), x)
print('ct target smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_ctrelu_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_ctrelu_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_ctrelu_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ctrelu-ss-b256` | 16 | 46.58 | 46.58 | 5.70 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-bnrecal-b256` | 17 | 46.76 | 46.56 | 1.52 | 0 | PASS |
| `imagenet100-96-pa-ctrelu-ss-b256` | 16 | 46.58 | 46.58 | 5.70 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 1 | disabled | 8.72 | 0 |
| 2 | disabled | 15.86 | 0 |
| 3 | disabled | 19.06 | 0 |
| 4 | disabled | 23.14 | 0 |
| 5 | disabled | 24.60 | 0 |
| 6 | disabled | 28.94 | 0 |
| 7 | disabled | 29.62 | 0 |
| 8 | disabled | 28.90 | 0 |
| 9 | disabled | 32.38 | 0 |
| 10 | disabled | 26.68 | 0 |
| 11 | disabled | 39.44 | 0 |
| 12 | disabled | 40.52 | 0 |
| 13 | disabled | 44.36 | 0 |
| 14 | disabled | 46.14 | 0 |
| 15 | disabled | 44.30 | 0 |
| 16 | disabled | 46.58 | 0 |

CT details:

| Module | Target | CT MSE |
| --- | --- | ---: |
| `special_resnet.layers.0.act` | relu | 0.0888795 |
| `special_resnet.layers.1.act` | relu | 0.0550250 |

Conclusion: configurable CT targets work mechanically and provide a useful
search hook for PAF-family experiments, but `relu` is a negative target choice
on this proxy. It finishes at 46.58, which is 0.84 points below default CT+SS
with the warmup/Swish CT target and has a larger max drop at epoch 10. Keep
`smartpaf_ct_target` available for future target/family searches, but keep the
default as `warmup`.

## 2026-06-01 StablePoly4 Output Scale 0.2 Attempt

Goal: test whether increasing the StablePoly4 polynomial branch output scale
from the module default 0.1 to 0.2 improves CT+SS recovery on the low-resolution
ImageNet-100 proxy.

Implementation:

- Added `poly4_output_scale` trainer kwarg.
- `None` preserves module defaults; positive float values override modules that
  expose `output_scale`.
- Added validation for non-positive values.
- Added config `configs/proxy_imagenet100_96_pa_ct_ss_outscale02_fast.yaml`.

Validation:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
p='configs/proxy_imagenet100_96_pa_ct_ss_outscale02_fast.yaml'
cfg=yaml.safe_load(open(p))
m=cfg['models'][0]
assert m['name'] == 'imagenet100-96-pa-ct-ss-outscale02-b256'
assert m['trainer_kwargs']['poly4_output_scale'] == 0.2
print('yaml ok')
PY
.venv/bin/python - <<'PY'
import torch
import torch.nn as nn
from trainers.base_trainer import Trainer

class DummyPoly(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_scale = 0.1
        self.range_args = None
        self.scale_mode = None
    def set_range_params(self, **kwargs):
        self.range_args = kwargs
    def set_scale_mode(self, mode, momentum, eps):
        self.scale_mode = (mode, momentum, eps)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = DummyPoly()

trainer = Trainer.__new__(Trainer)
trainer.model = DummyModel()
trainer.poly4_output_scale = 0.2
trainer.poly4_range_r = 2.0
trainer.poly4_range_lambda = 0.0
trainer.poly4_deriv_L = 3.0
trainer.poly4_deriv_lambda = 0.0
trainer.poly4_scale_mode = 'static'
trainer.poly4_dynamic_scale_momentum = 0.99
trainer.poly4_dynamic_scale_eps = 1e-6
trainer._configure_poly4_modules()
assert trainer.model.act.output_scale == 0.2
print('output scale smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_outscale02_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_outscale02_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_outscale02_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-outscale02-b256` | 16 | 47.58 | 47.58 | 0.58 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-outscale02-b256` | 16 | 47.58 | 47.58 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-at-group2-stop-bnrecal-b256` | 17 | 46.76 | 46.56 | 1.52 | 0 | PASS |
| `imagenet100-96-pa-ctrelu-ss-b256` | 16 | 46.58 | 46.58 | 5.70 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 1 | disabled | 8.60 | 0 |
| 2 | disabled | 15.16 | 0 |
| 3 | disabled | 19.90 | 0 |
| 4 | disabled | 22.58 | 0 |
| 5 | disabled | 23.30 | 0 |
| 6 | disabled | 29.22 | 0 |
| 7 | disabled | 31.30 | 0 |
| 8 | disabled | 30.94 | 0 |
| 9 | disabled | 34.00 | 0 |
| 10 | disabled | 38.98 | 0 |
| 11 | disabled | 40.54 | 0 |
| 12 | disabled | 42.70 | 0 |
| 13 | disabled | 44.44 | 0 |
| 14 | disabled | 47.32 | 0 |
| 15 | disabled | 46.74 | 0 |
| 16 | disabled | 47.58 | 0 |

CT and output-scale evidence:

| Item | Value |
| --- | ---: |
| Configured `output_scale` | 0.2 |
| `special_resnet.layers.0.act` CT MSE | 0.0566172 |
| `special_resnet.layers.1.act` CT MSE | 0.0325049 |

Conclusion: `poly4_output_scale=0.2` is a small positive CT+SS tweak on this
proxy. It improves final accuracy by 0.16 points over default CT+SS and reduces
max drop from 2.32 to 0.58, but it remains 1.36 points behind CT-only and 1.88
points behind the Swish baseline. Keep the kwarg and config as a useful search
knob; do not treat it as sufficient to close the baseline gap.

## 2026-06-01 StablePoly Degree 2 Attempt

Goal: start testing the paper's PAF forms/degrees direction by making the
existing StablePoly4 module configurable as a lower-degree polynomial while
preserving degree 4 as the default.

Implementation:

- Added default-off `StablePoly4(poly_degree=4)`.
- Added `set_poly_degree()` with supported degrees `2`, `3`, and `4`.
- Degree 2 masks the cubic and quartic terms in forward and derivative
  regularization while leaving parameter names/checkpoints compatible.
- Added trainer kwarg `poly4_degree`; `None` preserves module defaults.
- CT evaluation now uses the configured degree, so CT and training optimize the
  same polynomial family.
- Added config `configs/proxy_imagenet100_96_pa_ct_ss_degree2_fast.yaml`.

Validation:

```bash
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py trainers/base_trainer.py
.venv/bin/python - <<'PY'
import yaml
p='configs/proxy_imagenet100_96_pa_ct_ss_degree2_fast.yaml'
cfg=yaml.safe_load(open(p))
m=cfg['models'][0]
assert m['name'] == 'imagenet100-96-pa-ct-ss-degree2-b256'
assert m['trainer_kwargs']['poly4_degree'] == 2
assert m['trainer_kwargs']['poly4_scale_mode'] == 'static'
assert m['trainer_kwargs']['smartpaf_ct_init'] is True
assert m['trainer_kwargs']['smartpaf_ss_calibrate'] is True
print('yaml ok')
PY
.venv/bin/python - <<'PY'
import torch
from models.gate_net_cmp.block_def import StablePoly4

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
m = StablePoly4(output_scale=1.0, warmup_epochs=0, scale_mode='static')
m.set_poly_schedule(start_epoch=0, transition_epochs=0)
m.set_poly_degree(2)
m.static_absmax.fill_(1.0)
with torch.no_grad():
    m.a.fill_(0.01)
    m.b.fill_(0.1)
    m.c.fill_(0.5)
    m.d.fill_(1.0)
    m.e.fill_(0.25)
    m.set_epoch(1)
y2 = m(x)
expected2 = 0.5 * x.square() + x + 0.25
assert torch.allclose(y2, expected2)
m.set_poly_degree(4)
y4 = m(x)
expected4 = 0.01 * x**4 + 0.1 * x**3 + 0.5 * x**2 + x + 0.25
assert torch.allclose(y4, expected4)
print('stablepoly degree smoke ok')
PY
.venv/bin/python - <<'PY'
import torch.nn as nn
from trainers.base_trainer import Trainer

class DummyPoly(nn.Module):
    def __init__(self):
        super().__init__()
        self.poly_degree = 4
        self.output_scale = 0.1
        self.range_args = None
        self.scale_mode = None
    def set_poly_degree(self, degree):
        self.poly_degree = degree
    def set_range_params(self, **kwargs):
        self.range_args = kwargs
    def set_scale_mode(self, mode, momentum, eps):
        self.scale_mode = (mode, momentum, eps)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = DummyPoly()

trainer = Trainer.__new__(Trainer)
trainer.model = DummyModel()
trainer.poly4_output_scale = None
trainer.poly4_degree = 2
trainer.poly4_range_r = 2.0
trainer.poly4_range_lambda = 0.0
trainer.poly4_deriv_L = 3.0
trainer.poly4_deriv_lambda = 0.0
trainer.poly4_scale_mode = 'static'
trainer.poly4_dynamic_scale_momentum = 0.99
trainer.poly4_dynamic_scale_eps = 1e-6
trainer._configure_poly4_modules()
assert trainer.model.act.poly_degree == 2
assert trainer.model.act.scale_mode == ('static', 0.99, 1e-6)
print('trainer degree smoke ok')
PY
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_degree2_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_degree2_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_degree2_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-degree2-b256` | 16 | 47.74 | 47.64 | 2.16 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree2-b256` | 16 | 47.74 | 47.64 | 2.16 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-outscale02-b256` | 16 | 47.58 | 47.58 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 1 | disabled | 8.64 | 0 |
| 2 | disabled | 15.48 | 0 |
| 3 | disabled | 19.38 | 0 |
| 4 | disabled | 22.16 | 0 |
| 5 | disabled | 24.88 | 0 |
| 6 | disabled | 28.10 | 0 |
| 7 | disabled | 31.28 | 0 |
| 8 | disabled | 29.58 | 0 |
| 9 | disabled | 35.58 | 0 |
| 10 | disabled | 39.22 | 0 |
| 11 | disabled | 40.36 | 0 |
| 12 | disabled | 42.60 | 0 |
| 13 | disabled | 45.50 | 0 |
| 14 | disabled | 47.74 | 0 |
| 15 | disabled | 45.58 | 0 |
| 16 | disabled | 47.64 | 0 |

CT and degree evidence:

| Item | Value |
| --- | ---: |
| Configured `poly_degree` | 2 |
| `special_resnet.layers.0.act` CT MSE | 0.0695274 |
| `special_resnet.layers.1.act` CT MSE | 0.0404456 |
| Final logged `output_scale` | 0.1 |

Conclusion: degree 2 is a small positive PAF-family search result. It improves
CT+SS best accuracy by 0.32 points over the degree-4 default and by 0.16 points
over the outscale=0.2 tweak, although its final value is only 0.06 points above
outscale=0.2 and its max drop is larger. It still trails CT-only by 1.20 points
and the Swish baseline by 1.72 points. Keep `poly4_degree` as a useful search
knob; the next degree/form search should test degree 3 and possibly combine
degree 2 with the more stable `output_scale=0.2` setting.

## 2026-06-01 StablePoly Degree 3 Attempt

Goal: complete a direct degree-2/3/4 comparison for the current StablePoly
family after adding `poly4_degree`.

Implementation:

- Added config `configs/proxy_imagenet100_96_pa_ct_ss_degree3_fast.yaml`.
- No trainer/model code change in this step; this uses the already-validated
  `poly4_degree` path from the degree 2 attempt.

Validation:

```bash
.venv/bin/python - <<'PY'
import yaml
p='configs/proxy_imagenet100_96_pa_ct_ss_degree3_fast.yaml'
cfg=yaml.safe_load(open(p))
m=cfg['models'][0]
kw=m['trainer_kwargs']
assert m['name'] == 'imagenet100-96-pa-ct-ss-degree3-b256'
assert kw['poly4_degree'] == 3
assert kw['poly4_scale_mode'] == 'static'
assert kw['smartpaf_ct_init'] is True
assert kw['smartpaf_ss_calibrate'] is True
assert kw['smartpaf_alternate_training'] is False
print('yaml ok')
PY
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py trainers/base_trainer.py
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_degree3_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_degree3_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_degree3_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-degree3-b256` | 16 | 47.26 | 47.26 | 1.96 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree2-b256` | 16 | 47.74 | 47.64 | 2.16 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-outscale02-b256` | 16 | 47.58 | 47.58 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree3-b256` | 16 | 47.26 | 47.26 | 1.96 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 1 | disabled | 8.60 | 0 |
| 2 | disabled | 15.24 | 0 |
| 3 | disabled | 19.60 | 0 |
| 4 | disabled | 23.20 | 0 |
| 5 | disabled | 25.48 | 0 |
| 6 | disabled | 28.16 | 0 |
| 7 | disabled | 31.06 | 0 |
| 8 | disabled | 30.90 | 0 |
| 9 | disabled | 31.46 | 0 |
| 10 | disabled | 37.80 | 0 |
| 11 | disabled | 41.36 | 0 |
| 12 | disabled | 43.60 | 0 |
| 13 | disabled | 45.22 | 0 |
| 14 | disabled | 47.18 | 0 |
| 15 | disabled | 45.22 | 0 |
| 16 | disabled | 47.26 | 0 |

CT and degree evidence:

| Item | Value |
| --- | ---: |
| Configured `poly_degree` | 3 |
| `special_resnet.layers.0.act` CT MSE | 0.0695140 |
| `special_resnet.layers.1.act` CT MSE | 0.0404400 |
| Final logged `output_scale` | 0.1 |

Conclusion: degree 3 is stable but not useful on this proxy. It finishes at
47.26, which is 0.48 points below degree 2 and 0.16 points below the degree-4
default CT+SS run. Degree search now points to degree 2 as the only positive
lower-degree setting tested so far. The next useful search is a combination
test: degree 2 plus the more stable `output_scale=0.2` setting.

## 2026-06-01 StablePoly Degree 2 + Output Scale 0.2 Attempt

Goal: test whether the two positive CT+SS knobs found so far, degree 2 and
`output_scale=0.2`, compose into a better proxy result.

Implementation:

- Added config `configs/proxy_imagenet100_96_pa_ct_ss_degree2_outscale02_fast.yaml`.
- No trainer/model code change in this step; this combines the existing
  `poly4_degree` and `poly4_output_scale` knobs.

Validation:

```bash
.venv/bin/python - <<'PY'
import yaml
p='configs/proxy_imagenet100_96_pa_ct_ss_degree2_outscale02_fast.yaml'
cfg=yaml.safe_load(open(p))
m=cfg['models'][0]
kw=m['trainer_kwargs']
assert m['name'] == 'imagenet100-96-pa-ct-ss-degree2-outscale02-b256'
assert kw['poly4_degree'] == 2
assert kw['poly4_output_scale'] == 0.2
assert kw['poly4_scale_mode'] == 'static'
assert kw['smartpaf_ct_init'] is True
assert kw['smartpaf_ss_calibrate'] is True
assert kw['smartpaf_alternate_training'] is False
print('yaml ok')
PY
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py trainers/base_trainer.py
git diff --check
```

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_ss_degree2_outscale02_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_ss_degree2_outscale02_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_ss_degree2_outscale02_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-ss-degree2-outscale02-b256` | 16 | 47.76 | 47.76 | 0.56 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree2-outscale02-b256` | 16 | 47.76 | 47.76 | 0.56 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree2-b256` | 16 | 47.74 | 47.64 | 2.16 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-outscale02-b256` | 16 | 47.58 | 47.58 | 0.58 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-b256` | 16 | 47.42 | 47.42 | 2.32 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree3-b256` | 16 | 47.26 | 47.26 | 1.96 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 1 | disabled | 8.78 | 0 |
| 2 | disabled | 15.56 | 0 |
| 3 | disabled | 19.36 | 0 |
| 4 | disabled | 22.26 | 0 |
| 5 | disabled | 24.78 | 0 |
| 6 | disabled | 28.34 | 0 |
| 7 | disabled | 31.92 | 0 |
| 8 | disabled | 32.58 | 0 |
| 9 | disabled | 33.94 | 0 |
| 10 | disabled | 40.48 | 0 |
| 11 | disabled | 41.42 | 0 |
| 12 | disabled | 43.30 | 0 |
| 13 | disabled | 45.60 | 0 |
| 14 | disabled | 47.02 | 0 |
| 15 | disabled | 46.46 | 0 |
| 16 | disabled | 47.76 | 0 |

CT and config evidence:

| Item | Value |
| --- | ---: |
| Configured `poly_degree` | 2 |
| Configured `output_scale` | 0.2 |
| `special_resnet.layers.0.act` CT MSE | 0.0566409 |
| `special_resnet.layers.1.act` CT MSE | 0.0325149 |

Conclusion: degree 2 plus `output_scale=0.2` is the best CT+SS-family proxy so
far. The best accuracy improves only 0.02 points over degree 2 alone, but final
accuracy improves by 0.12 and max drop falls from 2.16 to 0.56. This confirms
the settings compose mostly as a stability improvement rather than a large
accuracy gain. It still trails CT-only by 1.18 points and Swish baseline by 1.70
points, so CT-only remains the stronger deployment candidate unless later PA/SS
work closes that gap.

## 2026-06-01 CT-Only Degree 2 + Output Scale 0.2 Attempt

Goal: test whether the best CT+SS knobs, degree 2 and `output_scale=0.2`, also
help the stronger learned-scale CT-only path.

Implementation:

- Added config `configs/proxy_imagenet100_96_pa_ct_degree2_outscale02_fast.yaml`.
- No trainer/model code change in this step; this uses existing `poly4_degree`
  and `poly4_output_scale` with `poly4_scale_mode: learned`.

Validation:

```bash
.venv/bin/python - <<'PY'
import yaml
p='configs/proxy_imagenet100_96_pa_ct_degree2_outscale02_fast.yaml'
cfg=yaml.safe_load(open(p))
m=cfg['models'][0]
kw=m['trainer_kwargs']
assert m['name'] == 'imagenet100-96-pa-ct-degree2-outscale02-b256'
assert kw['poly4_scale_mode'] == 'learned'
assert kw['poly4_degree'] == 2
assert kw['poly4_output_scale'] == 0.2
assert kw['smartpaf_ct_init'] is True
assert kw['smartpaf_alternate_training'] is False
assert 'smartpaf_ss_calibrate' not in kw
print('yaml ok')
PY
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py trainers/base_trainer.py
git diff --check
```

Proxy command:

```bash
setsid bash -lc 'mkdir -p logs; .venv/bin/python -u train.py --config configs/proxy_imagenet100_96_pa_ct_degree2_outscale02_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_pa_ct_degree2_outscale02_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_pa_ct_degree2_outscale02_fast.status' < /dev/null &
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-pa-ct-degree2-outscale02-b256` | 16 | 48.84 | 48.84 | 0.04 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-degree2-outscale02-b256` | 16 | 48.84 | 48.84 | 0.04 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree2-outscale02-b256` | 16 | 47.76 | 47.76 | 0.56 | 0 | PASS |
| `imagenet100-96-pa-ct-ss-degree2-b256` | 16 | 47.74 | 47.64 | 2.16 | 0 | PASS |

Phase details:

| Epoch | Phase | Recorded val acc | Guard |
| ---: | --- | ---: | ---: |
| 1 | disabled | 9.00 | 0 |
| 2 | disabled | 14.46 | 0 |
| 3 | disabled | 18.66 | 0 |
| 4 | disabled | 21.26 | 0 |
| 5 | disabled | 23.50 | 0 |
| 6 | disabled | 27.74 | 0 |
| 7 | disabled | 31.76 | 0 |
| 8 | disabled | 32.74 | 0 |
| 9 | disabled | 36.18 | 0 |
| 10 | disabled | 39.80 | 0 |
| 11 | disabled | 42.76 | 0 |
| 12 | disabled | 42.72 | 0 |
| 13 | disabled | 46.22 | 0 |
| 14 | disabled | 47.00 | 0 |
| 15 | disabled | 48.70 | 0 |
| 16 | disabled | 48.84 | 0 |

CT and config evidence:

| Item | Value |
| --- | ---: |
| Configured `poly4_scale_mode` | learned |
| Configured `poly_degree` | 2 |
| Configured `output_scale` | 0.2 |
| `special_resnet.layers.0.act` CT MSE | 0.000162552 |
| `special_resnet.layers.1.act` CT MSE | 0.0000537013 |

Conclusion: learned-scale CT-only remains stronger than CT+SS. The degree 2
plus `output_scale=0.2` variant is very stable and nearly matches CT-only, but
it finishes 0.10 points below the original CT-only run. The useful takeaway is
not an accuracy gain; it is that degree/output-scale tuning can preserve most
of CT-only's accuracy while sharply reducing drop. Keep original CT-only as the
best proxy result in this family, and keep the tuned variant as a stability
candidate for longer runs.

## 2026-06-01 AutoFHE Adaptive Precision Round

Goal: apply the AutoFHE-style layerwise mixed-precision idea to the current
StablePoly proxy. The repo already had a global `poly4_degree`; this round adds
`poly4_degrees`, which can assign degree 2/3/4 per StablePoly module by module
order or module name. The first AutoFHE proxy uses `[2, 2]` because the prior
degree search showed degree 2 retained CT-only accuracy while reducing
polynomial depth.

AutoFHE research notes and tooling were added under `autofhe/` with a local
`.gitignore`. The selection helper ranks candidates by accuracy and polynomial
depth, preferring lower depth within a 0.25 percentage point accuracy tolerance.

Validation commands:

```bash
.venv/bin/python -m py_compile trainers/base_trainer.py autofhe/select_precision.py
.venv/bin/python autofhe/select_precision.py --repo-root .
.venv/bin/python - <<'PY'
import yaml
p='configs/proxy_imagenet100_96_autofhe_adaptive_degree_fast.yaml'
cfg=yaml.safe_load(open(p))
kw=cfg['models'][0]['trainer_kwargs']
assert cfg['models'][0]['name'] == 'imagenet100-96-autofhe-adaptive-degree2-b256'
assert kw['poly4_degrees'] == [2, 2]
assert kw['poly4_scale_mode'] == 'learned'
assert kw['poly4_output_scale'] == 0.2
print('yaml ok')
PY
git diff --check
```

Proxy command:

```bash
bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_autofhe_adaptive_degree_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_autofhe_adaptive_degree_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_autofhe_adaptive_degree_fast.status'
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-autofhe-adaptive-degree2-b256` | 16 | 48.98 | 48.98 | 1.02 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-autofhe-adaptive-degree2-b256` | 16 | 48.98 | 48.98 | 1.02 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-pa-ct-degree2-outscale02-b256` | 16 | 48.84 | 48.84 | 0.04 | 0 | PASS |

CT and adaptive precision evidence:

| Item | Value |
| --- | --- |
| Configured `poly4_scale_mode` | learned |
| Configured `poly4_degrees` | `[2, 2]` |
| Configured `output_scale` | 0.2 |
| `special_resnet.layers.0.act` CT MSE | 0.000162552 |
| `special_resnet.layers.1.act` CT MSE | 0.0000537013 |
| AutoFHE selector recommendation | `imagenet100-96-autofhe-adaptive-degree2-b256` |

Conclusion: the AutoFHE-style adaptive precision path is applicable to this
project. On this two-activation proxy, the selected low-degree configuration is
both cheaper by polynomial depth and slightly more accurate than the previous
CT-only degree-4 run (48.98% vs 48.94%). AT, dynamic scale, and DS-to-SS remain
default-off experimental paths because their best results still trail CT-only.
The current recommended project setting is CT initialization with learned scale,
per-module degree selection, and degree 2 for both StablePoly sites on this
proxy.

## 2026-06-01 AutoFHE PAT Swish Backward Implementation

Goal: add an AutoFHE PAT-style backward path without adding a ReLU configuration
branch. Since this project uses Swish as the warmup/original smooth activation,
the implemented option is Swish-specific:

- `poly4_pat_swish_backward`

When enabled, the StablePoly branch keeps the same polynomial forward value,
but the activation input gradient follows Swish. Polynomial coefficients and
learned input scale still receive gradients through a detached-input polynomial
path, so PAT does not freeze coefficient learning.

Implementation sketch:

```python
swish_surrogate = poly_branch.detach() + silu(x) - silu(x).detach()
poly_param = polynomial(detached_x_or_detached_scaled_x, params)
poly_branch = swish_surrogate + poly_param - poly_param.detach()
```

Validation commands:

```bash
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py trainers/base_trainer.py
.venv/bin/python - <<'PY'
import torch
from models.gate_net_cmp.block_def import StablePoly4

x0 = torch.tensor([-1.0, 0.25, 1.5], requires_grad=True)
x1 = x0.detach().clone().requires_grad_(True)
base = StablePoly4(output_scale=1.0, warmup_epochs=0, scale_mode='learned', pat_swish_backward=False)
pat = StablePoly4(output_scale=1.0, warmup_epochs=0, scale_mode='learned', pat_swish_backward=True)
for m in (base, pat):
    m.train()
    m.set_poly_schedule(start_epoch=0, transition_epochs=0)
    with torch.no_grad():
        m.a.zero_(); m.b.zero_(); m.c.fill_(0.5); m.d.fill_(1.0); m.e.zero_(); m.log_in_scale.zero_()
with torch.no_grad():
    for name in ('a','b','c','d','e','log_in_scale'):
        getattr(pat, name).copy_(getattr(base, name))
y_base = base(x0)
y_pat = pat(x1)
assert torch.allclose(y_base, y_pat)
y_base.sum().backward()
y_pat.sum().backward()
sig = torch.sigmoid(x1.detach())
expected = sig + x1.detach() * sig * (1 - sig)
assert torch.allclose(x1.grad, expected, atol=1e-6)
assert not torch.allclose(x0.grad, x1.grad)
assert pat.c.grad is not None and abs(float(pat.c.grad)) > 0
assert pat.d.grad is not None and abs(float(pat.d.grad)) > 0
assert pat.log_in_scale.grad is not None and abs(float(pat.log_in_scale.grad)) > 0
print('pat swish learned smoke ok')
PY
git diff --check
```

Follow-up proxy config:

- `configs/proxy_imagenet100_96_autofhe_pat_swish_fast.yaml`

Conclusion: the Swish-specific PAT backward is mechanically ready and
default-off. It should be tested as an ablation against the existing
`imagenet100-96-autofhe-adaptive-degree2-b256` result before making it a
recommended default.

## 2026-06-01 AutoFHE PAT Swish Backward Proxy

Goal: test whether the Swish-specific PAT backward improves the selected
AutoFHE adaptive degree-2 proxy. The setup matches
`proxy_imagenet100_96_autofhe_adaptive_degree_fast.yaml` except that
`poly4_pat_swish_backward: true` is enabled.

Proxy command:

```bash
bash -lc '.venv/bin/python -u train.py --config configs/proxy_imagenet100_96_autofhe_pat_swish_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 96 --force > logs/proxy_imagenet100_96_autofhe_pat_swish_fast.log 2>&1; echo $? > logs/proxy_imagenet100_96_autofhe_pat_swish_fast.status'
```

Result summary:

| Model | Rows | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-autofhe-pat-swish-degree2-b256` | 16 | 48.22 | 48.22 | 0.00 | 0 | 0 | 0 | PASS |

Comparison:

| Model | Rows | Best | Final | Max drop | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-96-swish-baseline-b256` | 16 | 49.46 | 49.46 | 0.00 | 0 | PASS |
| `imagenet100-96-autofhe-adaptive-degree2-b256` | 16 | 48.98 | 48.98 | 1.02 | 0 | PASS |
| `imagenet100-96-pa-ct-b256` | 16 | 48.94 | 48.94 | 0.00 | 0 | PASS |
| `imagenet100-96-autofhe-pat-swish-degree2-b256` | 16 | 48.22 | 48.22 | 0.00 | 0 | PASS |

Phase details:

| Epoch | Val acc | Note |
| ---: | ---: | --- |
| 5 | 23.58 | first StablePoly starts transition; no early drop |
| 8 | 34.42 | better than no-PAT run at the same point |
| 10 | 38.34 | second StablePoly starts transition; falls behind no-PAT |
| 13 | 45.42 | lower than no-PAT 46.28 |
| 16 | 48.22 | final |

Conclusion: Swish PAT backward is numerically stable and avoids collapse, but it
is a negative result for the degree-2 proxy: final accuracy is 0.76 percentage
points below the no-PAT adaptive-degree run. Keep `poly4_pat_swish_backward`
default-off. The likely reason is that degree-2 polynomial gradients are already
well behaved after CT; replacing activation-input gradients with Swish helps
some early transition epochs but slows late adaptation after the second
StablePoly module activates. PAT may still be useful for degree-3/4 or more
aggressive PAF families where true polynomial gradients are less stable.

## 2026-06-01 CIFAR 224 AutoFHE Comparison Setup

Goal: compare the Swish baseline, AutoFHE adaptive degree-2 proxy, and the
Swish PAT-backward ablation on CIFAR-10/CIFAR-100 resized to ImageNet-style
224px inputs. This checks whether the AutoFHE strategy remains stable outside
the ImageNet100 96px proxy before attempting a larger ImageNet100 run.

Configs:

- `configs/large_cifar10_224_autofhe_compare.yaml`
- `configs/large_cifar100_224_autofhe_compare.yaml`

Planned commands:

```bash
.venv/bin/python -u train.py --config configs/large_cifar10_224_autofhe_compare.yaml --dataset cifar10 --train_dir ./data --val_dir ./data --download --result_dir ./results --gpus 1 2 3 --input_size 224 --force
.venv/bin/python -u train.py --config configs/large_cifar100_224_autofhe_compare.yaml --dataset cifar100 --train_dir ./data --val_dir ./data --download --result_dir ./results --gpus 1 2 3 --input_size 224 --force
```

First CIFAR-10 attempt with `./data` failed before training because the CIFAR
download was attempted concurrently by three workers and the configured proxy
returned connection errors. Retrying with local `./tmp/cifar-10-batches-py`
entered training but found a CT initialization bug at 224px: StablePoly CT/SS
activation sampling used `torch.linspace(...).long()` on CUDA for large flattened
feature maps. At 224px this can round the final index to `numel()`, causing
`index_select` to trigger a scatter/gather device-side assert. The sampling code
now uses integer arithmetic and a debug run confirmed CT initialization reaches
the first training batch for `cifar10-224-autofhe-degree2-b128`.

Run command:

```bash
.venv/bin/python -u train.py --config configs/large_cifar10_224_autofhe_compare.yaml --dataset cifar10 --train_dir ./tmp --val_dir ./tmp --result_dir ./results --gpus 1 2 3 --input_size 224 --force
```

The 40-epoch run was stopped intentionally after all three models passed the
first StablePoly transition window; the log status is therefore non-zero due to
manual termination, not model failure.

Result summary at stop:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar10-224-swish-baseline-b128` | 28 | 92.98 | 92.78 | 8.67 | 0 | 0 | 0 | PASS |
| `cifar10-224-autofhe-degree2-b128` | 22 | 92.41 | 92.41 | 2.78 | 0 | 0 | 0 | PASS |
| `cifar10-224-autofhe-pat-swish-degree2-b128` | 16 | 90.53 | 90.42 | 2.05 | 0 | 0 | 0 | PASS |

Transition observations:

- No-PAT AutoFHE crossed the first StablePoly transition at epoch 15 without
  collapse: epoch 14 was 89.68, epoch 15 rose to 90.77, and epoch 22 reached
  92.41.
- Swish PAT crossed the same transition stably but underperformed no-PAT:
  epoch 14 was 90.53, epoch 15 was 89.76, and epoch 16 was 90.42.
- The current CIFAR-10 224 proxy supports the ImageNet100 96px conclusion:
  keep Swish PAT backward as an ablation/default-off feature. The stronger
  recommended proxy is AutoFHE adaptive degree-2 without PAT.

CIFAR-100 data prep status:

- Standard torchvision `cifar-100-python` was not present locally.
- Concurrent torchvision download failed through the configured HTTP proxy.
- Direct no-proxy download timed out.
- `proxychains4` through the configured socks5 proxy returned connection
  refused.

Therefore CIFAR-100/ImageNet100 follow-up training is pending a usable
CIFAR-100 data root or a working external download path.

## 2026-06-01 ImageNet100 224 AutoFHE Comparison Setup

Goal: continue the large-resolution stability check on available ImageNet100
data while CIFAR-100 is blocked by missing data/download failures.

Dataset check:

- `train`: 128,982 images via symlinks into `imagenet_1000`
- `val`: 5,000 images via symlinks into `imagenet_1000`

Config:

- `configs/large_imagenet100_224_autofhe_compare.yaml`

Planned command:

```bash
.venv/bin/python -u train.py --config configs/large_imagenet100_224_autofhe_compare.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 1 2 3 --input_size 224 --no_memory_fs --force
```

First attempt:

- Config: `configs/large_imagenet100_224_autofhe_compare.yaml`
- Batch size: 128
- Workers/prefetch: 12/4 per model, three models in parallel

This did not OOM and loaded the dataset correctly, but it was CPU/I/O bound:
DataLoader workers consumed roughly 1000% CPU while GPUs were mostly idle except
for intermittent baseline batches. The run was stopped before any epoch result.

Follow-up fast proxy config:

- `configs/large_imagenet100_224_autofhe_fast.yaml`
- Models: Swish baseline and no-PAT AutoFHE degree-2
- Workers/prefetch reduced to 4/2 per model
- Epochs reduced to 12

The multi-model manager was still inefficient for this dataset/host
combination, so the fast proxy was run as two single-model processes on GPU1
and GPU2:

```bash
.venv/bin/python -u train.py --config configs/large_imagenet100_224_autofhe_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 1 --input_size 224 --no_memory_fs --models imagenet100-224-swish-baseline-fast-b128 --force
.venv/bin/python -u train.py --config configs/large_imagenet100_224_autofhe_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 224 --no_memory_fs --models imagenet100-224-autofhe-degree2-fast-b128 --force
```

Final 12-epoch result:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `imagenet100-224-autofhe-degree2-fast-b128` | 12 | 71.52 | 71.52 | 0.00 | 0 | 0 | 0 | PASS |
| `imagenet100-224-swish-baseline-fast-b128` | 12 | 71.40 | 71.40 | 0.00 | 0 | 0 | 0 | PASS |

Transition observations:

- AutoFHE crossed the first StablePoly transition at epoch 5 and the second at
  epoch 6 without collapse or accuracy drop: validation accuracy improved from
  47.62 at epoch 5 to 54.28 at epoch 6 and 58.44 at epoch 7.
- At epoch 8, AutoFHE and the Swish baseline were effectively tied: 62.78 vs
  62.84. At epoch 9, they remained within 0.20 percentage points: 66.86 vs
  67.06.
- The final AutoFHE proxy slightly exceeded the Swish baseline: 71.52 vs 71.40.
  Both runs completed with process status 0 and no nonfinite, skipped-batch, or
  activation guard events.

Conclusion: the no-PAT AutoFHE adaptive degree-2 strategy is stable at 224px on
the available ImageNet100 proxy and did not collapse after staged StablePoly
replacement. Together with the CIFAR-10 224 result, this supports keeping
degree-2 no-PAT as the recommended proxy path and keeping Swish PAT backward as
a default-off ablation.

## 2026-06-02 Evolution Rank-10 NAS AutoFHE Iteration

Goal: move from the hand-written ResNet proxy to an architecture produced by
regularized evolution, then iterate AutoFHE changes on that searched structure.

Searched architecture:

- Source copy: `configs/nas_variants/evolution_rank10.json`
- Original result source: `nas_result/best_models/10.json`
- Rank/generation: rank 10, generation 153
- Search metadata: 12.57M reported params, 1.174G FLOPs, FHE latency 10.92M,
  24 bootstraps, max depth 243
- Block choices: `[13, 9, 13, 11, 13, 13, 13, 13]`, mostly GatedMBConv4 with
  LearnableSwish, plus one GatedMBConv1 and one SE block.

Tracked configs:

- `configs/evolution_rank10_imagenet100_224_fast.yaml`
- `configs/evolution_rank10_poly4_imagenet100_224_fast.yaml`
- `configs/evolution_rank10_poly4_gate_sigmoid_imagenet100_224_fast.yaml`
- `tools/create_rank10_poly4_variant.py`

Commands:

```bash
.venv/bin/python -u train.py --config configs/evolution_rank10_imagenet100_224_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 1 --input_size 224 --no_memory_fs --models evolution-rank10-imagenet100-224-b128 --force
.venv/bin/python -u train.py --config configs/evolution_rank10_poly4_imagenet100_224_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 --input_size 224 --no_memory_fs --models evolution-rank10-poly4-imagenet100-224-b64 --force
```

Completed 12-epoch results:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `evolution-rank10-imagenet100-224-b128` | 12 | 68.96 | 68.96 | 0.00 | 0 | 0 | 0 | PASS |
| `evolution-rank10-poly4-imagenet100-224-b64` | 12 | 65.16 | 61.22 | 9.64 | 0 | 0 | 0 | PASS |

Observations:

- The original evolution architecture trains stably under the same 12-epoch
  ImageNet100 224 proxy, reaching 68.96.
- The full Poly4 block-mapped variant is stable through the first transition:
  epoch 5 was 48.52 and epoch 6 improved to 52.68.
- Late progressive replacement exposed a real weakness: best accuracy reached
  65.16 at epoch 10, then validation loss jumped to 151298 at epoch 11 and
  final accuracy fell to 61.22. No nonfinite, skipped-batch, or collapse-guard
  event fired, so this is a degradation rather than a hard numerical failure.
- Root cause hypothesis: mapping the GatedMBConv block id to Poly4 also changed
  the internal GatedDepthwise gate activation from Sigmoid to StablePoly4. That
  makes the gate no longer bounded like a probability and correlates with the
  late validation loss explosion.

Follow-up change:

- `models/gate_net_cmp/block_def.py` now treats `StablePoly4` as a Swish-family
  activation for GatedDepthwise gate selection, so a Poly4 MBConv keeps the gate
  activation as Sigmoid while only the main MBConv activation becomes
  StablePoly4.
- The conservative follow-up config
  `configs/evolution_rank10_poly4_gate_sigmoid_imagenet100_224_fast.yaml` has
  8 StablePoly4 modules instead of 16 and is running as the next iteration.

## 2026-06-02 Rank-10 Stability Follow-ups

Goal: isolate the late-collapse mechanism in the evolution rank-10 Poly4
variant and find a stable AutoFHE mapping for the searched architecture.

Additional tracked configs and variants:

- `configs/evolution_rank10_poly4_gate_sigmoid_imagenet100_224_fast.yaml`
- `configs/evolution_rank10_poly4_head5_imagenet100_224_fast.yaml`
- `configs/evolution_rank10_poly4_head4_static_imagenet100_224_fast.yaml`
- `configs/evolution_rank10_poly4_head5_static_imagenet100_224_fast.yaml`
- `configs/nas_variants/evolution_rank10_poly4_head5.json`
- `configs/nas_variants/evolution_rank10_poly4_head4.json`
- `tools/create_rank10_partial_poly4_variant.py`

Completed results:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `evolution-rank10-imagenet100-224-b128` | 12 | 68.96 | 68.96 | 0.00 | 0 | 0 | 0 | PASS |
| `evolution-rank10-poly4-head4-static-imagenet100-224-b96` | 12 | 65.40 | 65.24 | 0.16 | 0 | 0 | 0 | PASS |
| `evolution-rank10-poly4-imagenet100-224-b64` | 12 | 65.16 | 61.22 | 9.64 | 0 | 0 | 0 | PASS |
| `evolution-rank10-poly4-head5-static-imagenet100-224-b96` | 12 | 64.90 | 64.90 | 0.00 | 0 | 0 | 0 | PASS |
| `evolution-rank10-poly4-head5-imagenet100-224-b96` | 10 | 63.90 | 19.70 | 44.20 | 0 | 0 | 1 | COLLAPSE |
| `evolution-rank10-poly4-gatesigmoid-imagenet100-224-b96` | 10 | 62.46 | 8.40 | 54.06 | 0 | 0 | 1 | COLLAPSE |

Baseline comparison added during this iteration:

| Model | Params | Epochs | Best | Final | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `imagenet100-224-efficientnet-b0-fast-b128` | 4.14M | 12 | 72.74 | 72.74 | PASS |
| `imagenet100-224-resnet18-fast-b128` | 11.23M | 12 | 72.52 | 72.52 | PASS |

Observations:

- Gate-sigmoid alone did not fix collapse. It reached 62.46 at epoch 9 and
  collapsed at epoch 10 to 8.40 with the guard firing.
- Head5 partial replacement improved the pre-collapse peak to 63.90 at epoch 9,
  but still collapsed at epoch 10 to 19.70. This disproves the simple hypothesis
  that only adding the sixth StablePoly4 block caused the failure.
- Head4 with static Poly4 scale crossed the same epoch-10 region without a drop:
  epoch 9 was 62.58, epoch 10 was 64.22, epoch 11 was 65.40, and final was
  65.24. This is the best stable rank-10 Poly4 result so far and exceeds the
  full Poly4 best while avoiding its late degradation.
- Head5 with static Poly4 scale also crossed epoch 10 without collapse: epoch 9
  was 62.62, epoch 10 was 63.96, and final/best was 64.90. This supports
  learned-scale drift as a major contributor to the head5 collapse, but adding
  the fifth static Poly4 did not recover accuracy and underperformed head4
  static.
- Accuracy is still below the unmodified searched architecture: 65.40 vs 68.96.
  The next useful direction is to recover accuracy without reintroducing
  learned-scale collapse, for example head4 with learned scale plus
  restore/reduce-LR guard, or a head4 static run with a less suppressive
  polynomial output scale/transition schedule.

## 2026-06-03 Rank-10 Learned-Scale Recovery

Goal: keep the stable head4 partial StablePoly4 mapping, recover some of the
accuracy lost by static scaling, and avoid the learned-scale collapse seen in
head5/full mappings.

In-progress result:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `evolution-rank10-poly4-head4-learned-recover-imagenet100-224-b96` | 10 | 64.20 | 17.52 | 46.68 | 0 | 0 | 1 | COLLAPSE |

Current trajectory is slightly ahead of head4 static at matched epochs:

| Epoch | Head4 static | Head4 learned-recover |
| ---: | ---: | ---: |
| 5 | 50.38 | 50.76 |
| 6 | 54.62 | 55.32 |
| 7 | 58.92 | 59.70 |
| 8 | 60.82 | 62.18 |
| 9 | 62.58 | 64.20 |
| 10 | 64.22 | 17.52 |

The learned-scale run uses the AutoFHE-inspired stable recipe already found to
work well on proxy experiments: degree-2 StablePoly4, output scale 0.2, CT
initialization, progressive activation transition, and restore-best/reduce-LR
collapse guard. It has not triggered nonfinite batches, skipped batches, or the
collapse guard through epoch 9. At epoch 9 it was 1.62 percentage points ahead
of the static-scale head4 run at the same epoch and within 1.04 percentage
points of the static run's final 65.24. It then collapsed at epoch 10: val_acc
dropped from 64.20 to 17.52, with no nonfinite/skipped batches but with the
collapse guard firing once. The run saved `collapse_epoch_10.pth`, while
`best_model.pth` remained the epoch-9 checkpoint.

Follow-up change prepared while the run continues:

- `trainers/multi_gpu_manager.py` now supports `poly_scale_lr_mult`, a separate
  learning-rate multiplier for StablePoly4 `log_in_scale` parameters.
- Existing configs are backward compatible: if `poly_scale_lr_mult` is omitted,
  `log_in_scale` keeps using `poly_lr_mult`.
- New config:
  `configs/evolution_rank10_poly4_head4_learned_slow_scale_imagenet100_224_fast.yaml`.
  It matches the learned-recover setup but sets `poly_scale_lr_mult: 0.1`.
- Dry instantiation check confirmed the intended optimizer groups:
  `poly` has 20 coefficient parameters at LR 7e-04, while `poly_scale` has 4
  scale parameters at LR 7e-05.

Rationale: static scale is stable but suppresses accuracy, while unconstrained
learned scale can drift aggressively. The next controlled ablation is to keep
Poly4 coefficients fully trainable but slow only `log_in_scale` updates.

Epoch-10 collapse makes this ablation the next priority: restore/reduce-LR did
not prevent the learned-scale failure mode from appearing in the validation
history, so scale drift needs to be controlled before the guard fires.

Slow-scale run started:

```bash
.venv/bin/python -u train.py \
  --config configs/evolution_rank10_poly4_head4_learned_slow_scale_imagenet100_224_fast.yaml \
  --dataset imagenet100 \
  --train_dir /home/xuming/Documents/dataset/imagenet_100/train \
  --val_dir /home/xuming/Documents/dataset/imagenet_100/val \
  --result_dir ./results \
  --gpus 1 \
  --input_size 224 \
  --no_memory_fs \
  --models evolution-rank10-poly4-head4-learned-slow-scale-imagenet100-224-b96 \
  --force
```

Startup checks:

- GPU: physical GPU 1.
- Optimizer groups: `poly` 20 parameters at LR 7e-04, `poly_scale` 4
  `log_in_scale` parameters at LR 7e-05.
- CT init completed; first batch had finite logits and finite loss.
- The run is intended to test whether slower scale learning can keep the
  epoch-9 gain of learned scale while avoiding the epoch-10 collapse.

First checkpoint:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `evolution-rank10-poly4-head4-learned-slow-scale-imagenet100-224-b96` | 12 | 65.58 | 65.44 | 0 | 0 | 0 | PASS |

Epoch 1 matches the unconstrained learned-recover run exactly at 15.60, so the
separate lower LR for `log_in_scale` does not damage the initial trajectory.
Scale values after epoch 1 are still dominated by CT initialization
(`blocks.0.activation` in_scale about 1.485 and `blocks.2.activation` about
1.478), so the useful signal for this ablation will come later, especially
epochs 8-10.

Epoch 2 reached 27.90 versus 27.80 for the unconstrained learned-recover run.
`log_in_scale` remained essentially at its CT-initialized values, which is the
intended behavior for this slow-scale ablation in the early warmup phase.
Epoch 3 reached 38.54 versus 38.82 for unconstrained learned-recover. The
0.28-point gap is small enough that slow-scale remains a viable ablation; the
decisive comparison is still whether it can avoid the epoch-10 collapse while
retaining most of the epoch-9 gain.
Epoch 4 reached 43.16 versus 43.22 for unconstrained learned-recover, only
0.06 points lower. `log_in_scale` still matched the CT-initialized values, so
the reduced scale LR is preserving the early accuracy trajectory while holding
scale drift near zero through the warmup phase.
Epoch 5 reached 50.16. This is 0.60 points below unconstrained learned-recover
at epoch 5 (50.76) and 0.22 points below the static-scale head4 run (50.38).
The slow-scale run is becoming slightly more conservative near the start of the
Poly4 transition but remains stable with no nonfinite/skipped batches and no
collapse-guard event.
Epoch 6 reached 54.60, essentially matching static head4 at the same epoch
(54.62) and trailing unconstrained learned-recover by 0.72 points (55.32).
The scale-control hypothesis is supported by diagnostics: by epoch 5,
`blocks.0.activation` in_scale was about 1.278, while the unconstrained
learned-recover run had already driven it down to about 0.422 at epoch 5 and
continued toward the later collapse. The tradeoff so far is clear: slower scale
learning loses part of the learned-scale accuracy gain but strongly damps the
scale drift.
Epoch 7 reached 58.92, exactly matching static head4 at the same epoch and
trailing unconstrained learned-recover by 0.78 points (59.70). The run is now
best understood as a learned-scale stabilization candidate that tracks static
accuracy, rather than a clear accuracy-recovery candidate so far.
Epoch 8 reached 61.14, 0.32 points above static head4 at the same epoch
(60.82) but 1.04 points below unconstrained learned-recover (62.18). The run is
stable through the first post-warmup region with no nonfinite/skipped batches
and no collapse-guard event. Epochs 9-10 remain the decisive check.
Epoch 9 reached 62.36, 0.22 points below static head4 at the same epoch
(62.58) and 1.84 points below unconstrained learned-recover (64.20). The run
has not recovered the learned-scale accuracy gain, but it remains stable going
into the epoch-10 collapse checkpoint where unconstrained learned-recover fell
to 17.52.
Epoch 10 reached 64.28 with no collapse-guard event. This directly validates
the slow-scale hypothesis for stability: the same head4 learned-scale recipe
that collapsed to 17.52 with unconstrained `log_in_scale` reached 64.28 when
the scale LR was reduced by 10x. It also slightly exceeded static head4 at the
same epoch (64.22) and the unconstrained learned-recover best before collapse
(64.20), but it is still below the static run's final/best result
(65.24/65.40). The remaining question is whether epochs 11-12 can improve past
the static head4 best.
Epoch 11 reached 65.58, exceeding the previous best stable head4 static result
of 65.40. This makes slow-scale the current best stable rank-10 Poly4 variant.
It is still below the unmodified evolution rank-10 architecture at 68.96, so
the remaining target is accuracy recovery, not stability.
Epoch 12 finished at 65.44, with best remaining 65.58. No nonfinite batches,
skipped batches, or collapse-guard events occurred. This finalizes
slow-scale as the best stable rank-10 Poly4 variant so far:

| Variant | Best | Final | Status |
| --- | ---: | ---: | --- |
| Original evolution rank10 | 68.96 | 68.96 | PASS |
| Head4 learned slow-scale Poly4 | 65.58 | 65.44 | PASS |
| Head4 static Poly4 | 65.40 | 65.24 | PASS |
| Head5 static Poly4 | 64.90 | 64.90 | PASS |
| Head4 unconstrained learned Poly4 | 64.20 | 17.52 | COLLAPSE |

Remaining gap to the unmodified searched architecture is 3.38 percentage
points at best accuracy.

Next ablation:

- New config:
  `configs/evolution_rank10_poly4_head4_learned_scale02_imagenet100_224_fast.yaml`.
- It keeps the head4 learned slow-scale recipe but raises
  `poly_scale_lr_mult` from 0.1 to 0.2.
- Optimizer dry check confirmed `poly_scale` has 4 `log_in_scale` parameters at
  LR 1.4e-04, while Poly4 coefficients remain at LR 7e-04.
- Rationale: 0.1 is stable and now best among stable Poly4 variants, but it
  gives up much of the unconstrained learned-scale accuracy gain. A 0.2
  multiplier is the next conservative attempt to recover accuracy while keeping
  scale drift far below the unconstrained 1.0 run.
- Scale02 epoch 1 reached 15.62, matching the earlier learned-scale starts
  (15.60 for scale02=0.1 and 15.60 for unconstrained learned-recover) with no
  nonfinite/skipped batches and no guard event.
- Scale02 epoch 2 reached 27.88, essentially matching slow-scale 0.1 at the
  same epoch (27.90) and unconstrained learned-recover (27.80), with no
  nonfinite/skipped batches and no guard event.
- Scale02 epoch 3 reached 38.64, 0.10 points above slow-scale 0.1 (38.54).
- Scale02 epoch 4 reached 42.48, 0.68 points below slow-scale 0.1 (43.16).
- Scale02 epoch 5 reached 50.64, 0.48 points above slow-scale 0.1 (50.16)
  and 0.12 points below unconstrained learned-recover (50.76). Stability
  remains clean: no nonfinite batches, no skipped batches, and no collapse
  guard event through epoch 5.
- Scale diagnostics from the epoch-5 best checkpoint are still controlled:
  `blocks.0.activation` in_scale is about 1.109 and `blocks.2.activation` is
  about 1.478. This is far from the aggressive unconstrained epoch-5 drift
  previously observed around `blocks.0.activation` in_scale 0.422. The 0.2
  multiplier therefore remains worth continuing to epoch 10 before deciding
  whether it improves on the stable 0.1 run.
- Scale02 epoch 6 reached 54.94, 0.34 points above slow-scale 0.1 (54.60)
  and 0.32 points above static head4 (54.62).
- Scale02 epoch 7 reached 59.02, 0.10 points above slow-scale 0.1 (58.92)
  and still below unconstrained learned-recover at the same epoch (59.70).
- Scale02 epoch 8 reached 61.24, 0.10 points above slow-scale 0.1 (61.14)
  and 0.42 points above static head4 (60.82).
- Scale02 epoch 9 reached 62.74, 0.38 points above slow-scale 0.1 (62.36)
  and 0.16 points above static head4 (62.58).
- Scale02 epoch 10 reached 64.34 with no collapse-guard event. This passes the
  exact checkpoint where unconstrained learned-recover collapsed to 17.52, and
  is slightly ahead of slow-scale 0.1 at the same epoch (64.28). The advantage
  is small, so the final decision depends on epochs 11-12 and whether best
  accuracy can exceed the current stable Poly4 best of 65.58.
- Scale diagnostics from the epoch-10 best checkpoint remain controlled:
  `blocks.0.activation` in_scale is about 0.689 and `blocks.2.activation` is
  about 1.230. The scale is moving more than the 0.1 run, as intended, but it
  has not reproduced the unconstrained learned-scale collapse behavior.
- Scale02 finished 12 epochs with best 65.50 at epoch 11 and final 65.16.
  No nonfinite batches, skipped batches, or collapse-guard events occurred.
  This confirms `poly_scale_lr_mult: 0.2` is stable, but it does not improve on
  the current best stable Poly4 run (`poly_scale_lr_mult: 0.1`, best 65.58).
  The result also shows that simply increasing learned-scale LR is not the
  right next lever: it preserves stability at 0.2 but gives no accuracy gain.
  The next rank-10 ablation should instead keep the 0.1 scale control and test
  a different training signal, such as the Swish-specific PAT backward path.

Swish-PAT rank-10 ablation:

- New config:
  `configs/evolution_rank10_poly4_head4_learned_slow_scale_pat_swish_imagenet100_224_fast.yaml`.
- It matches the current best stable slow-scale setup and changes only
  `poly4_pat_swish_backward: true` plus the model/result names.
- Startup checks confirmed PAT was enabled, `poly` had 20 coefficient
  parameters at LR 7e-04, and `poly_scale` had 4 `log_in_scale` parameters at
  LR 7e-05.
- CT init completed with very small MSE for the two modules that received
  samples (`blocks.0.activation` 3.8e-09, `blocks.2.activation` 1.7e-12); the
  first batch had finite logits and finite loss.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Head4 learned slow-scale + Swish-PAT | 12 | 64.82 | 64.54 | 0 | 0 | 0 | PASS |
| Head4 learned slow-scale no-PAT | 12 | 65.58 | 65.44 | 0 | 0 | 0 | PASS |
| Head4 learned scale02 no-PAT | 12 | 65.50 | 65.16 | 0 | 0 | 0 | PASS |
| Head4 static Poly4 | 12 | 65.40 | 65.24 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | slow-scale no-PAT | scale02 no-PAT | slow-scale + PAT |
| ---: | ---: | ---: | ---: |
| 1 | 15.60 | 15.62 | 15.60 |
| 2 | 27.90 | 27.88 | 27.80 |
| 3 | 38.54 | 38.64 | 38.66 |
| 4 | 43.16 | 42.48 | 42.98 |
| 5 | 50.16 | 50.64 | 50.40 |
| 6 | 54.60 | 54.94 | 54.70 |
| 7 | 58.92 | 59.02 | 58.90 |
| 8 | 61.14 | 61.24 | 60.56 |
| 9 | 62.36 | 62.74 | 62.68 |
| 10 | 64.28 | 64.34 | 64.40 |
| 11 | 65.58 | 65.50 | 64.82 |
| 12 | 65.44 | 65.16 | 64.54 |

Conclusion: Swish-PAT is stable on the searched rank-10 architecture but is a
negative accuracy result, matching the earlier proxy trend. It slightly helps
around epoch 10 but loses the late-epoch recovery at epochs 11-12. Best
checkpoint scale diagnostics are controlled (`blocks.0.activation` in_scale
about 0.998, `blocks.2.activation` about 1.322), so the loss is not caused by
scale collapse. Keep PAT default-off for this rank-10 recipe. The best stable
Poly4 variant remains head4 learned slow-scale no-PAT with best 65.58, leaving
a 3.38-point gap to the unmodified searched rank10 model at 68.96.

Head3 structure ablation:

- New NAS variant:
  `configs/nas_variants/evolution_rank10_poly4_head3.json`.
- New config:
  `configs/evolution_rank10_poly4_head3_learned_slow_scale_imagenet100_224_fast.yaml`.
- It maps only the first three Swish/LearnableSwish body blocks to StablePoly4
  and keeps the best stable learned slow-scale recipe: degree 2, output scale
  0.2, CT init, progressive transition, and `poly_scale_lr_mult: 0.1`.
- Startup checks confirmed 3 StablePoly4 modules, `poly` 15 coefficient
  parameters at LR 7e-04, and `poly_scale` 3 `log_in_scale` parameters at
  LR 7e-05. The first batch had finite logits and finite loss.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Head4 learned slow-scale Poly4 | 12 | 65.58 | 65.44 | 0 | 0 | 0 | PASS |
| Head3 learned slow-scale Poly4 | 12 | 65.52 | 65.52 | 0 | 0 | 0 | PASS |
| Head4 learned scale02 Poly4 | 12 | 65.50 | 65.16 | 0 | 0 | 0 | PASS |
| Head4 static Poly4 | 12 | 65.40 | 65.24 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Head4 slow-scale | Head3 slow-scale |
| ---: | ---: | ---: |
| 1 | 15.60 | 15.62 |
| 2 | 27.90 | 27.92 |
| 3 | 38.54 | 38.50 |
| 4 | 43.16 | 42.86 |
| 5 | 50.16 | 50.28 |
| 6 | 54.60 | 54.52 |
| 7 | 58.92 | 58.86 |
| 8 | 61.14 | 60.90 |
| 9 | 62.36 | 62.90 |
| 10 | 64.28 | 64.72 |
| 11 | 65.58 | 65.46 |
| 12 | 65.44 | 65.52 |

Conclusion: head3 is stable and has the best final accuracy among stable Poly4
variants so far, but its best accuracy is still 0.06 points below head4
slow-scale. The late-epoch trajectory is useful evidence: replacing fewer
blocks slightly improves the post-transition region around epochs 9-12, but
head3 does not beat the current best checkpoint. Best checkpoint scale
diagnostics are controlled (`blocks.0.activation` in_scale about 1.007,
`blocks.2.activation` about 1.325). The next structure-side check is head2:
if reducing the number of StablePoly4 replacements continues to raise late
accuracy, head2 should move closer to the original rank10 baseline while
quantifying the accuracy/FHE-coverage tradeoff.

Head2 structure ablation:

- New NAS variant:
  `configs/nas_variants/evolution_rank10_poly4_head2.json`.
- New config:
  `configs/evolution_rank10_poly4_head2_learned_slow_scale_imagenet100_224_fast.yaml`.
- It maps only the first two Swish/LearnableSwish body blocks to StablePoly4
  and keeps the same learned slow-scale recipe as head3/head4.
- Startup checks confirmed 2 StablePoly4 modules, `poly` 10 coefficient
  parameters at LR 7e-04, and `poly_scale` 2 `log_in_scale` parameters at
  LR 7e-05.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Head2 learned slow-scale Poly4 | 12 | 66.82 | 66.82 | 0 | 0 | 0 | PASS |
| Head4 learned slow-scale Poly4 | 12 | 65.58 | 65.44 | 0 | 0 | 0 | PASS |
| Head3 learned slow-scale Poly4 | 12 | 65.52 | 65.52 | 0 | 0 | 0 | PASS |
| Head4 static Poly4 | 12 | 65.40 | 65.24 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Head4 slow-scale | Head3 slow-scale | Head2 slow-scale |
| ---: | ---: | ---: | ---: |
| 1 | 15.60 | 15.62 | 15.54 |
| 2 | 27.90 | 27.92 | 27.64 |
| 3 | 38.54 | 38.50 | 38.56 |
| 4 | 43.16 | 42.86 | 43.20 |
| 5 | 50.16 | 50.28 | 51.36 |
| 6 | 54.60 | 54.52 | 53.54 |
| 7 | 58.92 | 58.86 | 59.06 |
| 8 | 61.14 | 60.90 | 61.84 |
| 9 | 62.36 | 62.90 | 62.98 |
| 10 | 64.28 | 64.72 | 65.58 |
| 11 | 65.58 | 65.46 | 66.64 |
| 12 | 65.44 | 65.52 | 66.82 |

Conclusion: head2 is the new best stable Poly4 variant. It improves best
accuracy by 1.24 points over the previous head4 slow-scale best (66.82 vs
65.58) and reduces the gap to the unmodified searched rank10 architecture from
3.38 points to 2.14 points. The tradeoff is lower StablePoly4/FHE coverage:
only the first two body blocks are replaced. Scale diagnostics are controlled
at the best checkpoint (`blocks.0.activation` in_scale about 1.028; the other
head2 module remains at its default in_scale about 0.368). The structure trend
is now clear: fewer replacements recover accuracy. The next check is head1 to
quantify the upper end of the accuracy/coverage tradeoff.

Head1 structure ablation:

- New NAS variant:
  `configs/nas_variants/evolution_rank10_poly4_head1.json`.
- New config:
  `configs/evolution_rank10_poly4_head1_learned_slow_scale_imagenet100_224_fast.yaml`.
- It maps only the first body Swish/LearnableSwish block to StablePoly4 and
  keeps the same learned slow-scale recipe as the head2/head3/head4 runs:
  degree 2, output scale 0.2, CT init, progressive transition, and
  `poly_scale_lr_mult: 0.1`.
- Startup checks confirmed 1 StablePoly4 module, `poly` 5 coefficient
  parameters at LR 7e-04, and `poly_scale` 1 `log_in_scale` parameter at
  LR 7e-05.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Head1 learned slow-scale Poly4 | 12 | 66.92 | 66.92 | 0 | 0 | 0 | PASS |
| Head2 learned slow-scale Poly4 | 12 | 66.82 | 66.82 | 0 | 0 | 0 | PASS |
| Head4 learned slow-scale Poly4 | 12 | 65.58 | 65.44 | 0 | 0 | 0 | PASS |
| Head3 learned slow-scale Poly4 | 12 | 65.52 | 65.52 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Head2 slow-scale | Head1 slow-scale |
| ---: | ---: | ---: |
| 1 | 15.54 | 15.54 |
| 2 | 27.64 | 27.64 |
| 3 | 38.56 | 38.66 |
| 4 | 43.20 | 43.18 |
| 5 | 51.36 | 51.18 |
| 6 | 53.54 | 53.82 |
| 7 | 59.06 | 59.16 |
| 8 | 61.84 | 61.98 |
| 9 | 62.98 | 63.08 |
| 10 | 65.58 | 65.38 |
| 11 | 66.64 | 66.64 |
| 12 | 66.82 | 66.92 |

Conclusion: head1 is now the best stable Poly4 variant. It improves best
accuracy by 0.10 points over head2 and cuts the gap to the unmodified searched
rank10 architecture to 2.04 points. The tradeoff is even lower StablePoly4
coverage: only one body block is replaced. Best-checkpoint scale diagnostics
remain controlled (`blocks.0.activation` in_scale about 1.028). This confirms
the accuracy/coverage trend: fewer StablePoly4 replacements recover accuracy,
but still do not match the original searched rank10 network.

Head1 scale-LR ablation:

- New config:
  `configs/evolution_rank10_poly4_head1_learned_scale02_imagenet100_224_fast.yaml`.
- It keeps the best head1 learned slow-scale structure and recipe, but raises
  only `poly_scale_lr_mult` from 0.1 to 0.2. Coefficient LR remains 7e-04;
  the single `log_in_scale` parameter uses LR 1.4e-04.
- Data path was revalidated before launch: 128,982 train images, 5,000 val
  images, and 100 classes through the existing ImageNet-100 symlink subset.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Head1 learned slow-scale Poly4 (`poly_scale_lr_mult=0.1`) | 12 | 66.92 | 66.92 | 0 | 0 | 0 | PASS |
| Head1 learned scale02 Poly4 (`poly_scale_lr_mult=0.2`) | 12 | 66.82 | 66.82 | 0 | 0 | 0 | PASS |
| Head2 learned slow-scale Poly4 | 12 | 66.82 | 66.82 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Head1 slow-scale | Head1 scale02 |
| ---: | ---: | ---: |
| 1 | 15.54 | 15.60 |
| 2 | 27.64 | 27.76 |
| 3 | 38.66 | 38.50 |
| 4 | 43.18 | 43.14 |
| 5 | 51.18 | 51.26 |
| 6 | 53.82 | 54.28 |
| 7 | 59.16 | 59.58 |
| 8 | 61.98 | 61.92 |
| 9 | 63.08 | 63.40 |
| 10 | 65.38 | 65.26 |
| 11 | 66.64 | 66.20 |
| 12 | 66.92 | 66.82 |

Conclusion: increasing `poly_scale_lr_mult` to 0.2 is stable but does not
improve the best checkpoint. It helps immediately after Poly4 activation
(epochs 6-7) but loses the late-epoch recovery and finishes 0.10 points below
the 0.1 slow-scale run. Keep `poly_scale_lr_mult=0.1` as the current best
head1 recipe.

Single-position replacement ablation:

- Updated `tools/create_rank10_partial_poly4_variant.py` so it can generate
  exact body-block replacement variants with `--indices`, while preserving the
  existing `--head-blocks` mode.
- The rank10 body has 8 blocks with original choices:
  `[13, 9, 13, 11, 13, 13, 13, 13]`.
- `idx1` was generated but not trained to completion: its block id is
  `gated_mbconv1_lswish -> gated_mbconv1_poly4`, and CT init reported
  `skipped, no activation samples`. Inspection showed this is because
  `MBConvBlock` with expansion 1 does not call `self.activation`, so this is an
  ineffective replacement position.
- New effective NAS variant:
  `configs/nas_variants/evolution_rank10_poly4_idx2.json`.
- New config:
  `configs/evolution_rank10_poly4_idx2_learned_slow_scale_imagenet100_224_fast.yaml`.
- `idx2` maps only body block index 2 from `gated_mbconv4_lswish` to
  `gated_mbconv4_poly4`, using the current best learned slow-scale recipe.
  CT init collected 20,000 activation samples for `blocks.2.activation` with
  MSE about 1.74e-12.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Idx2 learned slow-scale Poly4 | 12 | 68.82 | 68.82 | 0 | 0 | 0 | PASS |
| Head1 learned slow-scale Poly4 | 12 | 66.92 | 66.92 | 0 | 0 | 0 | PASS |
| Head2 learned slow-scale Poly4 | 12 | 66.82 | 66.82 | 0 | 0 | 0 | PASS |
| Head1 learned scale02 Poly4 | 12 | 66.82 | 66.82 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Head1 slow-scale | Idx2 slow-scale |
| ---: | ---: | ---: |
| 1 | 15.54 | 15.54 |
| 2 | 27.64 | 28.32 |
| 3 | 38.66 | 39.42 |
| 4 | 43.18 | 44.26 |
| 5 | 51.18 | 52.22 |
| 6 | 53.82 | 56.86 |
| 7 | 59.16 | 61.54 |
| 8 | 61.98 | 62.92 |
| 9 | 63.08 | 65.36 |
| 10 | 65.38 | 66.82 |
| 11 | 66.64 | 68.30 |
| 12 | 66.92 | 68.82 |

Conclusion: single-position replacement is a stronger direction than simply
reducing the number of leading replacements. Replacing only body block index 2
recovers 1.90 points over head1 and leaves only a 0.14-point gap to the
unmodified searched rank10 architecture (68.82 vs 68.96), while still using
one effective StablePoly4 module. This also narrows the gap to ResNet-18 from
5.60 points to 3.70 points and to EfficientNet-B0 from 5.82 points to 3.92
points. Next position-side checks should prioritize effective expansion blocks
such as later `gated_mbconv4_lswish` blocks; skip expansion-1 positions where
the activation module is not used.

Idx4/idx5 single-position replacement ablation:

- Forward-hook validation was added before launch for single-index variants:
  idx0/head1, idx2, idx4, idx5, idx6, and idx7 each call their StablePoly4
  module once in a forward pass; idx1 and idx3 call it zero times and should
  be treated as ineffective replacement positions.
- New NAS variants:
  `configs/nas_variants/evolution_rank10_poly4_idx4.json` and
  `configs/nas_variants/evolution_rank10_poly4_idx5.json`.
- New config:
  `configs/evolution_rank10_poly4_idx45_learned_slow_scale_imagenet100_224_fast.yaml`.
- Both runs used the same learned slow-scale recipe as idx2 and were trained
  in parallel on GPU2/GPU3.
- CT init was valid for both replacements: idx4 collected 20,000 samples with
  MSE about 1.36e-10; idx5 collected 20,000 samples with MSE about 2.14e-10.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Idx2 learned slow-scale Poly4 | 12 | 68.82 | 68.82 | 0 | 0 | 0 | PASS |
| Idx5 learned slow-scale Poly4 | 12 | 68.78 | 68.78 | 0 | 0 | 0 | PASS |
| Idx4 learned slow-scale Poly4 | 12 | 68.32 | 68.24 | 0 | 0 | 0 | PASS |
| Head1 learned slow-scale Poly4 | 12 | 66.92 | 66.92 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Idx2 | Idx4 | Idx5 |
| ---: | ---: | ---: | ---: |
| 1 | 15.54 | 15.52 | 15.54 |
| 2 | 28.32 | 28.16 | 28.22 |
| 3 | 39.42 | 38.92 | 39.18 |
| 4 | 44.26 | 43.12 | 43.74 |
| 5 | 52.22 | 52.30 | 52.04 |
| 6 | 56.86 | 56.36 | 55.80 |
| 7 | 61.54 | 60.42 | 60.46 |
| 8 | 62.92 | 63.26 | 62.74 |
| 9 | 65.36 | 65.16 | 64.94 |
| 10 | 66.82 | 67.64 | 67.46 |
| 11 | 68.30 | 68.32 | 68.22 |
| 12 | 68.82 | 68.24 | 68.78 |

Conclusion: idx2 remains the best single-position StablePoly4 replacement, but
idx5 is effectively tied within 0.04 points and idx4 is also strong. The
single-position trend is now robust: replacing one effective late MBConv4
activation recovers nearly all of the original rank10 accuracy, while leading
replacement at idx0/head1 is much worse. Continue with idx6 and idx7 before
choosing a final single-position recipe or testing two-position combinations.

Idx6/idx7 single-position replacement ablation:

- New NAS variants:
  `configs/nas_variants/evolution_rank10_poly4_idx6.json` and
  `configs/nas_variants/evolution_rank10_poly4_idx7.json`.
- New config:
  `configs/evolution_rank10_poly4_idx67_learned_slow_scale_imagenet100_224_fast.yaml`.
- Both runs used the same learned slow-scale recipe as idx2/idx4/idx5 and were
  trained in parallel on GPU2/GPU3.
- CT init was valid for both replacements: idx6 collected 20,000 samples with
  MSE about 1.05e-10; idx7 collected 20,000 samples with MSE about 1.17e-10.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EfficientNet-B0 ImageNet-100 baseline | 12 | 72.74 | 72.74 | 0 | 0 | 0 | PASS |
| ResNet-18 ImageNet-100 baseline | 12 | 72.52 | 72.52 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Idx6 learned slow-scale Poly4 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Idx7 learned slow-scale Poly4 | 12 | 68.94 | 68.94 | 0 | 0 | 0 | PASS |
| Idx2 learned slow-scale Poly4 | 12 | 68.82 | 68.82 | 0 | 0 | 0 | PASS |
| Idx5 learned slow-scale Poly4 | 12 | 68.78 | 68.78 | 0 | 0 | 0 | PASS |
| Idx4 learned slow-scale Poly4 | 12 | 68.32 | 68.24 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Idx2 | Idx5 | Idx6 | Idx7 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.54 | 15.54 | 15.52 | 15.50 |
| 2 | 28.32 | 28.22 | 28.32 | 28.12 |
| 3 | 39.42 | 39.18 | 39.42 | 39.14 |
| 4 | 44.26 | 43.74 | 43.84 | 43.36 |
| 5 | 52.22 | 52.04 | 52.00 | 51.68 |
| 6 | 56.86 | 55.80 | 56.88 | 56.80 |
| 7 | 61.54 | 60.46 | 60.88 | 60.36 |
| 8 | 62.92 | 62.74 | 63.18 | 63.06 |
| 9 | 65.36 | 64.94 | 65.84 | 65.24 |
| 10 | 66.82 | 67.46 | 67.22 | 67.54 |
| 11 | 68.30 | 68.22 | 68.62 | 68.54 |
| 12 | 68.82 | 68.78 | 68.96 | 68.94 |

Conclusion: idx6 is now the best single-position StablePoly4 replacement and
matches the unmodified searched rank10 model at 68.96. Idx7 is effectively
tied at 68.94. This changes the next search direction from only `idx2+idx5`
or `idx2+idx4` to combinations involving the best late positions:
`idx2+idx6`, `idx6+idx7`, `idx2+idx7`, and `idx5+idx7`.

Two-position follow-up launched:

- `configs/nas_variants/evolution_rank10_poly4_idx2_7.json`
- `configs/nas_variants/evolution_rank10_poly4_idx5_7.json`
- `configs/evolution_rank10_poly4_idx27_idx57_learned_slow_scale_imagenet100_224_fast.yaml`
- `configs/nas_variants/evolution_rank10_poly4_idx2_6.json`
- `configs/nas_variants/evolution_rank10_poly4_idx6_7.json`
- `configs/evolution_rank10_poly4_idx26_idx67pair_learned_slow_scale_imagenet100_224_fast.yaml`

Early health checks: `idx2+idx7` and `idx2+idx6` both completed CT init with
finite first batches and valid coefficient fits. Full two-position results are
still running and should decide whether replacing two effective activations can
keep the no-loss behavior seen in single-position idx6.

Two-position follow-up results:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Idx6 learned slow-scale Poly4 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Idx7 learned slow-scale Poly4 | 12 | 68.94 | 68.94 | 0 | 0 | 0 | PASS |
| Idx5+idx7 learned slow-scale Poly4 | 12 | 68.52 | 68.52 | 0 | 0 | 0 | PASS |
| Idx6+idx7 learned slow-scale Poly4 | 12 | 68.50 | 68.50 | 0 | 0 | 0 | PASS |
| Idx2+idx7 learned slow-scale Poly4 | 12 | 68.12 | 68.12 | 0 | 0 | 0 | PASS |
| Idx2+idx6 learned slow-scale Poly4 | 12 | 68.08 | 68.08 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Idx6 | Idx7 | Idx2+idx6 | Idx2+idx7 | Idx5+idx7 | Idx6+idx7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.52 | 15.50 | 15.52 | 15.52 | 15.48 | 15.48 |
| 2 | 28.32 | 28.12 | 28.02 | 28.20 | 28.02 | 27.94 |
| 3 | 39.42 | 39.14 | 38.94 | 38.86 | 37.62 | 37.94 |
| 4 | 43.84 | 43.36 | 44.12 | 43.86 | 43.54 | 43.70 |
| 5 | 52.00 | 51.68 | 51.38 | 51.32 | 51.88 | 51.56 |
| 6 | 56.88 | 56.80 | 56.66 | 56.12 | 55.10 | 54.82 |
| 7 | 60.88 | 60.36 | 60.08 | 60.68 | 59.80 | 59.74 |
| 8 | 63.18 | 63.06 | 63.10 | 62.60 | 62.78 | 62.96 |
| 9 | 65.84 | 65.24 | 64.98 | 64.72 | 65.08 | 65.14 |
| 10 | 67.22 | 67.54 | 66.60 | 67.08 | 66.66 | 66.86 |
| 11 | 68.62 | 68.54 | 67.92 | 67.80 | 67.78 | 67.90 |
| 12 | 68.96 | 68.94 | 68.08 | 68.12 | 68.52 | 68.50 |

Conclusion: replacing two effective activations is stable under the CT +
learned slow-scale recipe, but it consistently lowers accuracy by 0.44 to 0.88
points versus the best single-position idx6 run. The current best FHE-friendly
recipe for the searched rank10 architecture is therefore a single StablePoly4
replacement at body block idx6. It matches the original searched rank10 model
at 68.96 on this ImageNet-100 12-epoch recipe, while using one effective
polynomial activation. Do not expand to three-position replacement unless a new
training technique is introduced; the next useful improvement direction is to
preserve idx6 and tune training or polynomial parameterization rather than
increase replacement coverage.

Idx6 tuning follow-up:

- New config:
  `configs/evolution_rank10_poly4_idx6_tuning_learned_slow_scale_imagenet100_224_fast.yaml`.
- Both runs keep `configs/nas_variants/evolution_rank10_poly4_idx6.json` and
  change only one knob from the current best idx6 recipe:
  - `scale005`: lower `poly_scale_lr_mult` from 0.1 to 0.05.
  - `outscale01`: lower `poly4_output_scale` from 0.2 to 0.1.
- CT and first-batch checks were valid for both runs. `scale005` kept the same
  CT fit quality as idx6, MSE about 1.05e-10. `outscale01` fit was worse but
  still finite, MSE about 1.27e-08.

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Idx6 learned slow-scale Poly4 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Original evolution rank10 | 12 | 68.96 | 68.96 | 0 | 0 | 0 | PASS |
| Idx6 outscale01 Poly4 | 12 | 68.78 | 68.78 | 0 | 0 | 0 | PASS |
| Idx6 scale005 Poly4 | 12 | 68.54 | 68.54 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Idx6 baseline | Idx6 scale005 | Idx6 outscale01 |
| ---: | ---: | ---: | ---: |
| 1 | 15.52 | 15.54 | 15.50 |
| 2 | 28.32 | 28.04 | 28.10 |
| 3 | 39.42 | 39.30 | 39.14 |
| 4 | 43.84 | 43.90 | 43.74 |
| 5 | 52.00 | 51.60 | 51.96 |
| 6 | 56.88 | 56.54 | 56.80 |
| 7 | 60.88 | 60.58 | 60.88 |
| 8 | 63.18 | 63.44 | 63.30 |
| 9 | 65.84 | 65.38 | 65.50 |
| 10 | 67.22 | 67.50 | 67.06 |
| 11 | 68.62 | 68.40 | 68.62 |
| 12 | 68.96 | 68.54 | 68.78 |

Conclusion: neither lower scale LR nor lower polynomial output scale improves
the best idx6 recipe. Both are stable, but late-epoch accuracy is lower. Keep
`poly_scale_lr_mult=0.1` and `poly4_output_scale=0.2` as the current best
single-position idx6 recipe. Since both coverage expansion and these first
parameter tweaks reduce accuracy, the next experiment should target training
length or schedule quality while keeping the idx6 structure and default Poly4
knobs fixed.

## 2026-06-04 Rank10 idx6 Matched 16-Epoch Comparison

Purpose: verify whether the current best FHE-friendly single-position idx6
StablePoly4 replacement still matches the original evolution rank10 architecture
under a longer matched 16-epoch ImageNet-100 run.

Important dataset boundary: all numbers in this section use the local
ImageNet-100 subset, not ImageNet-1k. The validation set has 5,000 images across
100 classes.

Command:

```bash
.venv/bin/python -u train.py \
  --config configs/evolution_rank10_idx6_matched16_imagenet100_224.yaml \
  --dataset imagenet100 \
  --train_dir /home/xuming/Documents/dataset/imagenet_100/train \
  --val_dir /home/xuming/Documents/dataset/imagenet_100/val \
  --result_dir ./results \
  --gpus 2 3 \
  --input_size 224 \
  --no_memory_fs \
  --models evolution-rank10-matched16-imagenet100-224-b96 evolution-rank10-poly4-idx6-matched16-imagenet100-224-b96 \
  --force
```

Run paths:

- Config: `configs/evolution_rank10_idx6_matched16_imagenet100_224.yaml`
- Log: `logs/evolution_rank10_idx6_matched16_imagenet100_224.log`
- Status: `logs/evolution_rank10_idx6_matched16_imagenet100_224.status` = 0
- Results: `results/evolution_rank10_idx6_matched16_imagenet100_224`

Architecture notes:

- `evolution-rank10-matched16-imagenet100-224-b96` uses the evolution-searched
  rank10 NAS json at `configs/nas_variants/evolution_rank10.json`.
- The searched candidate metadata records rank 10, generation 153, 12.57M
  parameters in the search score, 1.174G FLOPs, and block choices
  `[13, 9, 13, 11, 13, 13, 12, 13]`.
- `evolution-rank10-poly4-idx6-matched16-imagenet100-224-b96` uses
  `configs/nas_variants/evolution_rank10_poly4_idx6.json`, which keeps the same
  architecture and maps only body block index 6 from block id 13 to block id 12
  to replace one Swish/LearnableSwish activation with StablePoly4.

Idx6 StablePoly4 recipe:

- `poly4_scale_mode: learned`
- `poly4_output_scale: 0.2`
- `poly4_degree: 2`
- `poly_scale_lr_mult: 0.1`
- `smartpaf_ct_init: true`
- `smartpaf_ct_max_samples: 20000`
- `smartpaf_transition_epochs: 6`
- CT fit: `blocks.6.activation`, samples 20,000, MSE `1.05201e-10`

Result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original evolution rank10 matched16 b96 | 16 | 72.04 | 72.04 | 0 | 0 | 0 | PASS |
| Idx6 StablePoly4 matched16 b96 | 16 | 72.04 | 72.04 | 0 | 0 | 0 | PASS |

Matched-epoch trajectory:

| Epoch | Original rank10 | Idx6 StablePoly4 | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 14.96 | 15.52 | +0.56 |
| 2 | 27.18 | 28.08 | +0.90 |
| 3 | 38.42 | 39.36 | +0.94 |
| 4 | 44.90 | 43.92 | -0.98 |
| 5 | 51.96 | 52.14 | +0.18 |
| 6 | 56.74 | 56.66 | -0.08 |
| 7 | 59.36 | 60.18 | +0.82 |
| 8 | 63.22 | 62.34 | -0.88 |
| 9 | 65.30 | 64.58 | -0.72 |
| 10 | 66.86 | 66.54 | -0.32 |
| 11 | 68.04 | 68.60 | +0.56 |
| 12 | 69.04 | 68.98 | -0.06 |
| 13 | 70.92 | 70.66 | -0.26 |
| 14 | 71.58 | 71.08 | -0.50 |
| 15 | 71.74 | 71.58 | -0.16 |
| 16 | 72.04 | 72.04 | +0.00 |

Baseline context from the existing 12-epoch 224px ImageNet-100 runs:

| Model | Epochs | Best | Final | Notes |
| --- | ---: | ---: | ---: | --- |
| EfficientNet-B0 b128 | 12 | 72.74 | 72.74 | standard baseline |
| ResNet-18 b128 | 12 | 72.52 | 72.52 | standard baseline |
| Original evolution rank10 b128 | 12 | 68.96 | 68.96 | searched rank10 baseline |
| Original evolution rank10 matched16 b96 | 16 | 72.04 | 72.04 | longer matched run |
| Idx6 StablePoly4 matched16 b96 | 16 | 72.04 | 72.04 | one effective polynomial activation |

Conclusion: the idx6 StablePoly4 proxy recipe is stable and accuracy-neutral in
this matched 16-epoch ImageNet-100 comparison. It exactly matches the original
evolution rank10 final and best accuracy at 72.04 with no nonfinite batches,
skipped batches, or collapse guard triggers. The 16-epoch schedule also narrows
the gap to the 12-epoch ResNet-18 and EfficientNet-B0 baselines, but does not
exceed them. The current best direction remains a single idx6 StablePoly4
replacement; expanding replacement coverage to two positions was stable but
lower accuracy in the prior ablations.

## 2026-06-04 Rank10 idx6 Matched 24-Epoch Follow-up

Purpose: continue the best rank10/idx6 recipe by extending the matched schedule
from 16 to 24 epochs. The goal is to test whether longer training can push the
evolution rank10 architecture and the idx6 StablePoly4 proxy past the existing
12-epoch ResNet-18 and EfficientNet-B0 ImageNet-100 baselines.

New config:

- `configs/evolution_rank10_idx6_matched24_imagenet100_224.yaml`

Only the schedule length and model names changed from the matched16 config:

- epochs: 16 -> 24
- batch size: remains 96
- learning rate: remains 0.0007
- optimizer/scheduler: AdamW + cosine
- Poly4 recipe: remains learned scale, output scale 0.2, degree 2, CT init,
  slow scale LR multiplier 0.1, progressive transition.

Note: because the Poly4 run keeps `poly4_warmup_ratio: 0.35`, the single idx6
StablePoly4 module starts at epoch 8 and begins affecting training at epoch 9
in the 24-epoch run. This preserves the proportional schedule from matched16
rather than fixing the absolute matched16 start epoch.

Command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/evolution_rank10_idx6_matched24_imagenet100_224.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 3 --input_size 224 --no_memory_fs --models evolution-rank10-matched24-imagenet100-224-b96 evolution-rank10-poly4-idx6-matched24-imagenet100-224-b96 --force > logs/evolution_rank10_idx6_matched24_imagenet100_224.log 2>&1; echo $? > logs/evolution_rank10_idx6_matched24_imagenet100_224.status' < /dev/null &
```

Run state at startup:

- Parent PID: 3919379
- Train PID: 3919381
- Log: `logs/evolution_rank10_idx6_matched24_imagenet100_224.log`
- Status file when complete: `logs/evolution_rank10_idx6_matched24_imagenet100_224.status`
- Results: `results/evolution_rank10_idx6_matched24_imagenet100_224`
- Original rank10 uses GPU2; idx6 StablePoly4 uses GPU3.
- CT fit for `blocks.6.activation`: samples 20,000, MSE `1.05201e-10`.
- First-batch diagnostics for both models had finite logits and finite loss.

Final result:

| Model | Epochs | Best | Final | Max drop | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original evolution rank10 matched24 b96 | 24 | 74.08 | 74.08 | 0.16 | 0 | 0 | 0 | PASS |
| Idx6 StablePoly4 matched24 b96 | 24 | 73.96 | 73.96 | 0.10 | 0 | 0 | 0 | PASS |

Epoch-by-epoch validation accuracy:

| Epoch | Original rank10 | Idx6 StablePoly4 | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 14.96 | 15.56 | +0.60 |
| 2 | 27.08 | 28.10 | +1.02 |
| 3 | 38.46 | 39.10 | +0.64 |
| 4 | 44.90 | 43.84 | -1.06 |
| 5 | 52.04 | 51.88 | -0.16 |
| 6 | 56.72 | 55.88 | -0.84 |
| 7 | 59.46 | 60.16 | +0.70 |
| 8 | 62.68 | 61.90 | -0.78 |
| 9 | 64.76 | 62.66 | -2.10 |
| 10 | 66.08 | 66.00 | -0.08 |
| 11 | 67.20 | 67.66 | +0.46 |
| 12 | 68.18 | 67.74 | -0.44 |
| 13 | 70.20 | 70.30 | +0.10 |
| 14 | 70.60 | 70.38 | -0.22 |
| 15 | 72.14 | 71.42 | -0.72 |
| 16 | 71.98 | 71.60 | -0.38 |
| 17 | 72.26 | 72.22 | -0.04 |
| 18 | 72.90 | 73.12 | +0.22 |
| 19 | 73.36 | 73.28 | -0.08 |
| 20 | 73.66 | 73.18 | -0.48 |
| 21 | 73.56 | 73.46 | -0.10 |
| 22 | 73.72 | 73.64 | -0.08 |
| 23 | 73.82 | 73.74 | -0.08 |
| 24 | 74.08 | 73.96 | -0.12 |

Baseline context. All rows in this table are ImageNet-100 224px results, not
ImageNet-1k results:

| Model | Epochs | Best | Final | Notes |
| --- | ---: | ---: | ---: | --- |
| Original evolution rank10 matched24 b96 | 24 | 74.08 | 74.08 | best current rank10 result |
| Idx6 StablePoly4 matched24 b96 | 24 | 73.96 | 73.96 | one effective polynomial activation |
| EfficientNet-B0 b128 | 12 | 72.74 | 72.74 | standard baseline |
| ResNet-18 b128 | 12 | 72.52 | 72.52 | standard baseline |
| Original evolution rank10 matched16 b96 | 16 | 72.04 | 72.04 | previous matched schedule |
| Idx6 StablePoly4 matched16 b96 | 16 | 72.04 | 72.04 | previous matched Poly4 schedule |
| Original evolution rank10 b128 | 12 | 68.96 | 68.96 | short searched-arch baseline |

Observations:

- The original rank10 24-epoch run finishes at 74.08, beating the 12-epoch
  EfficientNet-B0 baseline by 1.34 points and the 12-epoch ResNet-18 baseline by
  1.56 points on ImageNet-100.
- The idx6 StablePoly4 24-epoch run finishes at 73.96, beating EfficientNet-B0 by
  1.22 points and ResNet-18 by 1.44 points on ImageNet-100.
- StablePoly4 remains very close to the original searched architecture: final
  gap is -0.12 points, with zero nonfinite batches, zero skipped batches, and
  zero collapse guard triggers.
- The idx6 run uses proportional Poly4 scheduling, so `poly4_warmup_ratio: 0.35`
  delays the single StablePoly4 module to start at epoch 8 / begin affecting
  epoch 9. Epoch 9 still shows a short disruption, but the model recovers and
  tracks the original closely through the end.

Conclusion: extending the matched rank10 schedule to 24 epochs is enough to move
both the original searched architecture and the single idx6 StablePoly4 variant
above the existing 12-epoch ResNet-18 and EfficientNet-B0 ImageNet-100 baselines.
The result should not be interpreted as ImageNet-1k accuracy.

## 2026-06-05 AESPA/HerPN Swish Proxy Setup

Purpose: test whether the AESPA paper's basis-wise Hermite polynomial
normalization idea can help this repo's Swish/LearnableSwish NAS models without
adding a ReLU-specific configuration path.

Implementation:

- Added `SwishHerPN` in `models/gate_net_cmp/block_def.py`.
- The activation uses degree-2 normalized probabilists' Hermite bases:
  `h1=x`, `h2=(x^2-1)/sqrt(2)`.
- Basis-wise normalization is implemented with `LazyBatchNorm2d(affine=False)`
  on each nonconstant basis, matching AESPA/HerPN's scale-control idea while
  still working with the current activation constructors that do not pass a
  channel count.
- Coefficients are initialized from the standard-normal projection of Swish:
  `c0≈0.20662`, `c1≈0.5`, `c2≈0.24860`. `c1/c2`, final `gamma`, and final
  `beta` are trainable during proxy training.
- `SwishHerPN` is treated as Swish-like by the gated blocks, so gate paths that
  previously used Sigmoid for Swish/LearnableSwish still use Sigmoid.

Architecture integration:

- Added `swish_herpn` to `ACTIVATION_TYPES`.
- Added optional `activation_override` to `BlockConfig`.
- The override changes only a block's activation class; it does not change
  `block_id`, convolution type, expansion factor, SE, gated depthwise conv, CT
  policies, or searched architecture metadata.
- Added `tools/create_rank10_partial_activation_variant.py`.
- Generated `configs/nas_variants/evolution_rank10_swish_herpn_idx6.json`,
  which keeps rank10 body block idx6 as `block_id=13` and sets
  `activation_override: swish_herpn` only for that block.

Verification:

- `python -m py_compile` passed for the touched Python files.
- Single-activation CPU forward/backward smoke test passed with finite outputs
  and gradients.
- NAS JSON build smoke test passed: the generated model contains exactly one
  `SwishHerPN` module at `blocks.6.activation`, and a batch-size-2 224x224
  forward/backward pass produced finite logits.

Proxy config prepared:

- `configs/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast.yaml`
- 12 epochs, batch size 96, learning rate 0.0007, AdamW + cosine, no AMP,
  label smoothing 0.05.
- Includes both:
  - `evolution-rank10-proxy12-imagenet100-224-b96`
  - `evolution-rank10-swish-herpn-idx6-proxy12-imagenet100-224-b96`

Proxy command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast.yaml --dataset imagenet100 --train_dir /home/xuming/Documents/dataset/imagenet_100/train --val_dir /home/xuming/Documents/dataset/imagenet_100/val --result_dir ./results --gpus 2 3 --input_size 224 --no_memory_fs --force > logs/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast.log 2>&1; echo $? > logs/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast.status' < /dev/null &
```

Run state at startup:

- Parent PID: 483195
- Train PID: 483197
- Log: `logs/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast.log`
- Status file when complete:
  `logs/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast.status`
- Results: `results/evolution_rank10_swish_herpn_idx6_proxy_imagenet100_224_fast`
- Original rank10 proxy12 uses GPU2; idx6 Swish-HerPN proxy12 uses GPU3.
- Both models started with finite first-batch setup and no status file yet.

Result summary:

| Model | Epochs | Best | Final | Avg epoch sec | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original evolution rank10 proxy12 b96 | 12 | 69.04 | 69.04 | 679.37 | 0 | 0 | 0 | PASS |
| Idx6 Swish-HerPN proxy12 b96 | 12 | 69.06 | 69.06 | 852.58 | 0 | 0 | 0 | PASS |

Epoch-by-epoch validation accuracy:

| Epoch | Original rank10 | Idx6 Swish-HerPN | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 14.98 | 14.84 | -0.14 |
| 2 | 27.04 | 30.24 | +3.20 |
| 3 | 38.60 | 41.06 | +2.46 |
| 4 | 44.80 | 46.38 | +1.58 |
| 5 | 51.76 | 51.26 | -0.50 |
| 6 | 56.94 | 57.82 | +0.88 |
| 7 | 60.70 | 60.98 | +0.28 |
| 8 | 64.02 | 64.18 | +0.16 |
| 9 | 65.70 | 66.24 | +0.54 |
| 10 | 67.48 | 67.60 | +0.12 |
| 11 | 68.74 | 68.92 | +0.18 |
| 12 | 69.04 | 69.06 | +0.02 |

Baseline context. All rows are ImageNet-100 224px results:

| Model | Epochs | Best | Final | Notes |
| --- | ---: | ---: | ---: | --- |
| Original evolution rank10 matched24 b96 | 24 | 74.08 | 74.08 | best current rank10 result |
| Idx6 StablePoly4 matched24 b96 | 24 | 73.96 | 73.96 | low-degree Poly4 replacement |
| EfficientNet-B0 b128 | 12 | 72.74 | 72.74 | standard baseline |
| ResNet-18 b128 | 12 | 72.52 | 72.52 | standard baseline |
| Idx6 Swish-HerPN proxy12 b96 | 12 | 69.06 | 69.06 | AESPA/HerPN-inspired activation override |
| Original evolution rank10 proxy12 b96 | 12 | 69.04 | 69.04 | direct proxy control |
| Original evolution rank10 b128 | 12 | 68.96 | 68.96 | prior short searched-arch baseline |

Observations:

- Swish-HerPN is stable in this idx6 proxy: no nonfinite batches, no skipped
  batches, and no collapse guard triggers.
- Accuracy is essentially neutral at 12 epochs: +0.02 points over the direct
  original proxy control. The early lead at epochs 2-4 shrinks by the end.
- Training cost is higher: average epoch time rises from 679.37s to 852.58s,
  about a 25.5% increase in this PyTorch training setup. In an FHE deployment,
  the basis-wise normalization would need folding or dedicated handling at
  inference time; the proxy only proves training stability and accuracy.

Conclusion: AESPA/HerPN's basis-wise Hermite normalization idea can be adapted to
the current Swish-based NAS model without a ReLU-specific configuration path, but
the single idx6 Swish-HerPN replacement is accuracy-neutral rather than clearly
better. It is worth keeping as an experimental activation override, while the
current strongest polynomial direction remains idx6 StablePoly4 matched24.

## 2026-06-05 CIFAR-100 Hermite Poly4 Quick Sweep

Purpose: test whether the AESPA/HerPN normalization idea is more useful when
applied directly to the Poly4 path, and whether replacing more than one Swish /
LearnableSwish activation can help. This is a deliberately small CIFAR-100
screening run using `/home/xuming/Documents/dataset/cifar-100-python.tar.gz`
extracted under `/home/xuming/Documents/dataset`.

Implementation:

- Added `HermitePoly4`, a `StablePoly4`-compatible activation that evaluates
  `a,b,c,d,e` as normalized probabilists' Hermite-basis coefficients instead of
  monomial coefficients.
- Added basis-wise `LazyBatchNorm2d(affine=False)` for 4D activation tensors.
- Kept the existing `StablePoly4` scheduling/CT/logging interface, so
  `poly4_warmup_ratio`, CT initialization, `poly_scale_lr_mult`, and summary
  tooling remain usable.
- Registered activation override key `poly4_herpn`.
- Generated activation-override NAS variants:
  - `configs/nas_variants/evolution_rank10_poly4_herpn_idx6.json`
  - `configs/nas_variants/evolution_rank10_poly4_herpn_idx5_6_7.json`
  - `configs/nas_variants/evolution_rank10_poly4_herpn_all8.json`

Validation:

```bash
.venv/bin/python -m py_compile models/gate_net_cmp/block_def.py network_gen/search_space.py trainers/base_trainer.py trainers/multi_gpu_manager.py
.venv/bin/python - <<'PY'
import torch
from models.json_registered import nas_json_model
from models.gate_net_cmp.block_def import HermitePoly4, StablePoly4

variants = [
    ('orig', 'configs/nas_variants/evolution_rank10.json', 0),
    ('stable_idx6', 'configs/nas_variants/evolution_rank10_poly4_idx6.json', None),
    ('herpn_idx6', 'configs/nas_variants/evolution_rank10_poly4_herpn_idx6.json', 1),
    ('herpn_idx567', 'configs/nas_variants/evolution_rank10_poly4_herpn_idx5_6_7.json', 3),
    ('herpn_all8', 'configs/nas_variants/evolution_rank10_poly4_herpn_all8.json', 8),
]
for name, path, expected in variants:
    model = nas_json_model(path, num_classes=100, pretrained=False)
    hcount = sum(1 for m in model.modules() if isinstance(m, HermitePoly4))
    scount = sum(1 for m in model.modules() if isinstance(m, StablePoly4) and not isinstance(m, HermitePoly4))
    if expected is not None:
        assert hcount == expected
    model.train()
    y = model(torch.randn(2, 3, 32, 32))
    y.float().sum().backward()
    print(name, torch.isfinite(y).all().item(), hcount, scount)
PY
git diff --check
```

6-epoch quick command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/evolution_rank10_poly4_herpn_cifar100_quick.yaml --dataset cifar100 --train_dir /home/xuming/Documents/dataset --val_dir /home/xuming/Documents/dataset --result_dir ./results --gpus 2 3 --input_size 32 --no_memory_fs --force > logs/evolution_rank10_poly4_herpn_cifar100_quick.log 2>&1; echo $? > logs/evolution_rank10_poly4_herpn_cifar100_quick.status' < /dev/null &
```

6-epoch result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar100-rank10-hermitepoly4-idx5-6-7-e6-b256` | 6 | 30.87 | 30.87 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-hermitepoly4-idx6-e6-b256` | 6 | 30.72 | 30.72 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-stablepoly4-idx6-e6-b256` | 6 | 30.27 | 30.27 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-original-e6-b256` | 6 | 30.25 | 30.25 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-hermitepoly4-all8-e6-b256` | 6 | 29.75 | 29.75 | 0 | 0 | 0 | PASS |

Observation: at 6 epochs, replacing idx5/6/7 with HermitePoly4 was the best
rank10 variant, +0.62 over the original. Replacing all eight body activations
was worse, suggesting that broad replacement is not automatically better.

12-epoch follow-up command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/evolution_rank10_poly4_herpn_cifar100_e12.yaml --dataset cifar100 --train_dir /home/xuming/Documents/dataset --val_dir /home/xuming/Documents/dataset --result_dir ./results --gpus 2 3 --input_size 32 --no_memory_fs --force > logs/evolution_rank10_poly4_herpn_cifar100_e12.log 2>&1; echo $? > logs/evolution_rank10_poly4_herpn_cifar100_e12.status' < /dev/null &
```

12-epoch rank10 result:

| Model | Epochs | Best | Final | Nonfinite | Skipped | Guard | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cifar100-rank10-original-e12-b256` | 12 | 36.77 | 36.77 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-hermitepoly4-idx6-e12-b256` | 12 | 36.11 | 36.11 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-stablepoly4-idx6-e12-b256` | 12 | 35.92 | 35.92 | 0 | 0 | 0 | PASS |
| `cifar100-rank10-hermitepoly4-idx5-6-7-e12-b256` | 12 | 35.66 | 34.53 | 0 | 0 | 0 | PASS |

Observation: the 6-epoch multi-position gain did not hold at 12 epochs.
HermitePoly4 idx5/6/7 hit 35.66 but ended at 34.53, and its validation loss
exploded late despite finite batches. Single idx6 HermitePoly4 was stable and
slightly better than single idx6 StablePoly4, but both trailed the original
rank10 control.

Standard-network baseline command:

```bash
setsid bash -lc '.venv/bin/python -u train.py --config configs/cifar100_standard_baselines_e12.yaml --dataset cifar100 --train_dir /home/xuming/Documents/dataset --val_dir /home/xuming/Documents/dataset --result_dir ./results --gpus 2 3 --input_size 32 --no_memory_fs --force > logs/cifar100_standard_baselines_e12.log 2>&1; echo $? > logs/cifar100_standard_baselines_e12.status' < /dev/null &
```

12-epoch standard-network comparison:

| Model | Params | Epochs | Best | Final | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `cifar100-resnet18-e12-b256` | 11.23M | 12 | 49.26 | 49.26 | ImageNet-style ResNet, 32px input |
| `cifar100-resnet56-e12-b256` | 0.86M | 12 | 44.05 | 44.05 | CIFAR-style ResNet |
| `cifar100-rank10-original-e12-b256` | 12.60M | 12 | 36.77 | 36.77 | searched rank10 control |
| `cifar100-resnet20-e12-b256` | 0.28M | 12 | 36.74 | 36.74 | CIFAR-style ResNet |
| `cifar100-rank10-hermitepoly4-idx6-e12-b256` | 12.60M | 12 | 36.11 | 36.11 | one HermitePoly4 override |
| `cifar100-rank10-stablepoly4-idx6-e12-b256` | 12.60M | 12 | 35.92 | 35.92 | one StablePoly4 block mapping |
| `cifar100-rank10-hermitepoly4-idx5-6-7-e12-b256` | 12.60M | 12 | 35.66 | 34.53 | three HermitePoly4 overrides |
| `cifar100-efficientnet-b0-e12-b256` | 4.14M | 12 | 30.66 | 30.66 | ImageNet-style EfficientNet-B0, 32px input |

Conclusion: on this small CIFAR-100 screen, applying AESPA/HerPN directly to
Poly4 is stable for one replacement and can show early gains, but it does not
beat the original rank10 control by 12 epochs. More importantly, all rank10
variants are far below the stronger standard baselines ResNet56 and ResNet18
under the same short CIFAR-100 schedule. Multi-position HermitePoly4 replacement
needs a slower or staged schedule before it is worth scaling back to ImageNet-100.
