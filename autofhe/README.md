# AutoFHE Applicability Notes

This folder keeps the AutoFHE-specific research notes and lightweight tooling
separate from the main SmartPAF code path.

Sources checked:

- USENIX Security 2024 paper page:
  https://www.usenix.org/conference/usenixsecurity24/presentation/ao
- Paper PDF:
  https://www.usenix.org/system/files/usenixsecurity24-ao.pdf
- arXiv abstract:
  https://arxiv.org/abs/2310.08012

## What Carries Over

AutoFHE searches CNN adaptations under two objectives: keep accuracy high and
reduce FHE evaluation cost, especially bootstrapping pressure. The parts that
fit this repo are:

- Coefficient tuning before normal training. This repo already has
  `smartpaf_ct_init`, and it is the strongest stable technique so far.
- Layerwise polynomial precision/form selection. This repo now supports
  `poly4_degrees` so different StablePoly modules can use degree 2, 3, or 4.
- Dynamic scale to static scale conversion. This repo has
  `smartpaf_ds_to_ss_after_training`, but experiments show it is deployable
  rather than accuracy-improving.
- Candidate acceptance/rejection around alternate training. The current
  epoch-level proxy has the rejection pieces, but it is still weaker than
  CT-only.

## Current Recommendation

For this project's ImageNet-100 96px proxy, the best AutoFHE-inspired setting
is:

- `poly4_scale_mode: learned`
- `poly4_degrees: [2, 2]`
- `poly4_output_scale: 0.2`
- `smartpaf_ct_init: true`
- `smartpaf_alternate_training: false`

Evidence:

- Swish baseline: 49.46%.
- CT learned-scale degree 4 default: 48.94%.
- CT learned-scale degree 2 with output scale 0.2: 48.84%.
- CT+SS degree 2: 47.74%.
- CT+SS degree 3: 47.26%.
- CT+SS degree 4 default: 47.42%.
- Best AT proxy: 46.76%, still below CT-only.

The degree-2 learned-scale run is only 0.10 percentage points below CT-only and
has lower polynomial depth, so it is the best current accuracy/cost tradeoff.

## Reproduction

Summarize the available precision experiments:

```bash
.venv/bin/python autofhe/select_precision.py --repo-root .
```

Run the selected adaptive-degree proxy:

```bash
.venv/bin/python -u train.py \
  --config configs/proxy_imagenet100_96_autofhe_adaptive_degree_fast.yaml \
  --dataset imagenet100 \
  --train_dir /home/xuming/Documents/dataset/imagenet_100/train \
  --val_dir /home/xuming/Documents/dataset/imagenet_100/val \
  --result_dir ./results \
  --gpus 2 \
  --input_size 96 \
  --force
```

## Deferred Work

The full AutoFHE method is broader than the current implementation. The main
remaining gaps are:

- Replace all non-polynomial operators, not only the current StablePoly
  activation sites.
- Add a real FHE latency/bootstrap evaluator into the search objective.
- Search PAF families beyond this repo's StablePoly form.
- Use a per-layer training-group loop instead of global epoch-level AT.
