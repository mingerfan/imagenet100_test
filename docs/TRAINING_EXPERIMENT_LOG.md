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

- SWA: not implemented. SMART-PAF uses SWA in PA/AT recovery.
- BN recalibration: partially implemented. BN is frozen during poly phase, but
  there is no post-training BN running-stat recalibration.
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
