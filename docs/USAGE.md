# Plant Disease Classification Experiments

A config-driven PyTorch codebase for the plant-disease experiments that started out as
Kaggle notebooks (in [`notebooks/`](../notebooks/)). It covers:

- **Supervised** training of EfficientNet-B0 and ViT-Tiny on PlantVillage variants
  (AutoAugment, CutMix, CCLR, ECCLR), with cross-domain testing on PlantDoc.
- **Few-shot** 8-way 5/10-shot classification with Prototypical Networks, using the
  supervised models as backbones.

Each experiment runs for several seeds with one command and reports **mean ± std**, plus
**time and memory** use, confusion matrices, and **Grad-CAM / t-SNE** figures.

---

## Contents

1. [Installation](#1-installation)
2. [Data setup](#2-data-setup)
3. [Quick start](#3-quick-start)
4. [Commands for each experiment](#4-commands-for-each-experiment)
5. [Seeds and mean ± std](#5-seeds-and-mean--std)
6. [Changing hyperparameters and paths](#6-changing-hyperparameters-and-paths)
7. [Outputs: metrics, time and memory, figures](#7-outputs)
8. [Evaluating a trained checkpoint](#8-evaluating-a-trained-checkpoint)
9. [Using the code from a notebook](#9-using-the-code-from-a-notebook)
10. [Running on Kaggle or Colab](#10-running-on-kaggle--colab)
11. [Project structure](#11-project-structure)
12. [Differences from the original notebooks](#12-differences-from-the-original-notebooks)
13. [Troubleshooting](#13-troubleshooting)

For the design and how to add models, augmentations or experiment types, see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Installation

You need Python 3.10 or newer. A CUDA GPU is recommended; Apple-silicon (MPS) and CPU also work.

```bash
cd CCLR-ECCLR
python -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                                 # installs the `plantdisease` command
plantdisease list                                       # sanity check: lists models, transforms, ...
pytest                                                  # ~1 min test suite on synthetic data
```

To install a CUDA build of PyTorch, follow <https://pytorch.org/get-started/locally/> first,
then run `pip install -e .`.

## 2. Data setup

By default the datasets are read from `data/`. Each dataset root holds one sub-folder per
class:

| Config key           | Default location                                     | Folder names                                   |
|----------------------|------------------------------------------------------|------------------------------------------------|
| `plantvillage`       | `data/plantvillage/`                                 | original names, e.g. `Apple___Cedar_apple_rust` |
| `plantvillage_cclr`  | `data/plantvillage_cclr/`                            | canonical, e.g. `Apple___rust`                  |
| `plantvillage_ecclr` | `data/plant_village_ecclr/`                          | canonical                                      |
| `plantdoc` (target)  | `data/plantdoc/train/` + `data/plantdoc/test/` (merged) | PlantDoc names, e.g. `Apple_rust_leaf`       |

Folder names are mapped to the 8 canonical classes by the `class_map` entries in
[configs/base.yaml](../configs/base.yaml). Unmapped folders (other crops) are ignored, so you can
point a root at the full PlantVillage `color/` directory. You do not need to copy or resize
images first.

Check that a dataset is found before training:

```bash
plantdisease check-data configs/experiments/supervised/exp3_efficientnet_cclr.yaml
```

This prints the train/val/test/target sizes and per-class counts, and flags empty classes.

## 3. Quick start

```bash
# Short test run: 1 seed, 2 epochs
plantdisease run configs/experiments/supervised/exp3_efficientnet_cclr.yaml \
    --seeds 0 --set train.epochs=2

# The real thing: every seed in configs/base.yaml (default 0 1 2 3 4), mean ± std at the end
plantdisease run configs/experiments/supervised/exp3_efficientnet_cclr.yaml
```

Results are written to `outputs/exp3_efficientnet_cclr/summary.md`.

> **PlantDoc not downloaded yet?** Add `--set data.target=null` to any command. The target
> split is then skipped and everything else runs normally.

## 4. Commands for each experiment

Run the commands from the repository root. Every command trains and evaluates **all seeds**
listed in `configs/base.yaml` and writes the mean ± std summary. Append `--seeds 0 1 2` to
choose seeds, or `--set key=value` to change settings (see [§6](#6-changing-hyperparameters-and-paths)).

### 4.1 Supervised experiments

Each model is trained on a PlantVillage variant (70/20/10 train/val/test split) and evaluated
on that variant's `val` and `test` splits and on **PlantDoc** (`target`).

| # | Model | Training data / augmentation | Command |
|---|-------|------------------------------|---------|
| 1 | EfficientNet-B0 | PlantVillage + AutoAugment | `plantdisease run configs/experiments/supervised/exp1_efficientnet_autoaugment.yaml` |
| 2 | EfficientNet-B0 | PlantVillage + AutoAugment + CutMix | `plantdisease run configs/experiments/supervised/exp2_efficientnet_cutmix.yaml` |
| 3 | EfficientNet-B0 | CCLR PlantVillage + AutoAugment | `plantdisease run configs/experiments/supervised/exp3_efficientnet_cclr.yaml` |
| 4 | EfficientNet-B0 | ECCLR PlantVillage + AutoAugment | `plantdisease run configs/experiments/supervised/exp4_efficientnet_ecclr.yaml` |
| 5 | ViT-Tiny/16 | PlantVillage + AutoAugment | `plantdisease run configs/experiments/supervised/exp5_vit_autoaugment.yaml` |
| 6 | ViT-Tiny/16 | PlantVillage + timm augmentation + CutMix | `plantdisease run configs/experiments/supervised/exp6_vit_cutmix.yaml` |
| 7 | ViT-Tiny/16 | CCLR PlantVillage + AutoAugment | `plantdisease run configs/experiments/supervised/exp7_vit_cclr.yaml` |
| 8 | ViT-Tiny/16 | ECCLR PlantVillage + AutoAugment (RRC scale 0.5–1) | `plantdisease run configs/experiments/supervised/exp8_vit_ecclr.yaml` |

Training recipes come from the notebooks and live in [configs/models/](../configs/models/):

| | EfficientNet-B0 | ViT-Tiny |
|---|---|---|
| Optimizer | Adam, lr 1e-3 | AdamW, lr 1e-4 |
| LR schedule | StepLR (step 5, γ 0.1) | constant |
| Early stopping | `val_loss`, patience 5 | `val_acc`, patience 5 |
| Epochs / batch | 25 / 32 | 25 / 32 |

### 4.2 Few-shot experiments (Prototypical Networks)

Each few-shot run loads the backbone checkpoint **for the same seed** from the matching
supervised experiment (`outputs/<exp>/seed_<seed>/checkpoints/best.pt`), so
**run the supervised experiment first**. It evaluates 1000 random 8-way episodes (10 queries
per class) on the source `test` split and on PlantDoc, and reports accuracy ± 95% CI over
episodes, then mean ± std over seeds.

| Backbone (run this first) | Shots | Command |
|---------------------------|-------|---------|
| exp3 EfficientNet CCLR  | 5  | `plantdisease run configs/experiments/fewshot/cclr_efficientnet_5shot.yaml` |
| exp3 EfficientNet CCLR  | 10 | `plantdisease run configs/experiments/fewshot/cclr_efficientnet_10shot.yaml` |
| exp4 EfficientNet ECCLR | 5  | `plantdisease run configs/experiments/fewshot/ecclr_efficientnet_5shot.yaml` |
| exp4 EfficientNet ECCLR | 10 | `plantdisease run configs/experiments/fewshot/ecclr_efficientnet_10shot.yaml` |
| exp7 ViT CCLR           | 5  | `plantdisease run configs/experiments/fewshot/cclr_vit_5shot.yaml` |
| exp7 ViT CCLR           | 10 | `plantdisease run configs/experiments/fewshot/cclr_vit_10shot.yaml` |
| exp8 ViT ECCLR          | 5  | `plantdisease run configs/experiments/fewshot/ecclr_vit_5shot.yaml` |
| exp8 ViT ECCLR          | 10 | `plantdisease run configs/experiments/fewshot/ecclr_vit_10shot.yaml` |

Useful variations:

```bash
# Use weights you already trained in a notebook instead of a supervised run from this repo
plantdisease run configs/experiments/fewshot/cclr_vit_5shot.yaml --seeds 0 \
    --set model.checkpoint=/path/to/best_vit_model.pth

# Other episode settings
plantdisease run configs/experiments/fewshot/ecclr_vit_5shot.yaml \
    --set fewshot.n_way=5 fewshot.n_query=15 fewshot.n_eval_tasks=600

# Meta-train the backbone episodically before evaluating (off by default, as in the notebooks)
plantdisease run configs/experiments/fewshot/ecclr_vit_5shot.yaml \
    --set fewshot.n_train_tasks=500 fewshot.n_val_tasks=100 train.epochs=10
```

### 4.3 Everything, then comparison tables

```bash
bash scripts/run_all.sh                       # all supervised runs, then all few-shot runs
bash scripts/run_all.sh --set train.epochs=5  # overrides are passed to every run
```

`--resume` is enabled in the script, so re-running it skips seeds that already finished.
It ends by writing `outputs/comparison/supervised.md` and `outputs/comparison/fewshot.md`.
To build your own comparison table:

```bash
plantdisease summarize outputs/exp1_efficientnet_autoaugment outputs/exp3_efficientnet_cclr \
    --metrics test/accuracy target/accuracy target/macro_f1 train/wall_time_s \
    --output outputs/comparison/my_table.md     # .md or .csv
```

## 5. Seeds and mean ± std

- `seeds:` in [configs/base.yaml](../configs/base.yaml) defaults to `[0, 1, 2, 3, 4]`. Override
  it per run with `--seeds 0 1 2`.
- For every seed, `run` seeds Python, NumPy and PyTorch, the DataLoader workers, the shuffling
  and the few-shot episode sampler. It then trains, tests and explains, and saves
  `seed_<s>/metrics.json`.
- At the end it writes `summary.md`, `summary.csv` and `summary.json` with **mean, std
  (sample std, ddof=1), min, max and n** for every metric, plus `per_seed.csv`.
- **Splits:** `data.split_seed: 42` keeps the train/val/test split fixed, as in the
  notebooks, so the variance you measure comes from initialization and data order. Set
  `--set data.split_seed=null` to re-split with every seed and include split variance too.
- `--resume` skips seeds that already have `metrics.json`. Use it to add seeds later or to
  continue after a crash.
- For bit-exact repeatability on the same GPU, add `--set deterministic=true` (this is slower).

## 6. Changing hyperparameters and paths

Every value in the YAML configs can be overridden from the command line with
`--set key=value` (dot notation, several at once). Unknown keys are rejected, so typos fail
immediately. To see the fully resolved config:

```bash
plantdisease show-config configs/experiments/supervised/exp7_vit_cclr.yaml --set train.epochs=40
```

Common overrides:

| What | Override |
|---|---|
| All dataset paths at once | `paths.data_root=/mnt/datasets` |
| One dataset's path | `"datasets.plantvillage_cclr.roots=[/mnt/cclr]"` |
| PlantDoc in a different place | `"datasets.plantdoc.roots=[/x/plantdoc/train,/x/plantdoc/test]"` |
| Raw PlantVillage `color/` folder | `"datasets.plantvillage.roots=[/x/plantvillage/color]"` |
| Disable the target dataset | `data.target=null` |
| Epochs, LR, weight decay | `train.epochs=40 train.optimizer.lr=3e-4 train.optimizer.weight_decay=0.05` |
| Optimizer / scheduler | `train.optimizer.name=sgd train.scheduler.name=cosine` |
| Early stopping | `train.early_stopping.patience=10 train.early_stopping.monitor=val_acc` |
| Batch size / workers | `data.batch_size=64 data.num_workers=8` |
| Mixed precision (CUDA) | `train.amp=true` |
| Label smoothing, grad clipping | `train.label_smoothing=0.1 train.grad_clip_norm=1.0` |
| Augmentation | `data.train_transform=timm_train data.batch_mixer=mixup data.rrc_scale=[0.5,1.0]` |
| Split ratios / stratified | `data.split_ratios=[0.8,0.1,0.1] data.stratify=true` |
| Model variant | `model.variant=efficientnet_b3` or `model.variant=vit_small_patch16_224` |
| Device | `device=cuda:1`, `device=cpu`, `device=mps` |
| Output folder / run name | `output_dir=/scratch/runs name=exp3_lr3e-4` |
| Skip XAI figures | `xai.enabled=false` (or `xai.gradcam=false`, `xai.tsne=false`) |

Run `plantdisease list` for every registered model, transform, batch mixer, optimizer and
scheduler name.

**Tip:** when sweeping a hyperparameter, also set `name=` so each setting gets its own output
folder, e.g. `--set train.optimizer.lr=3e-4 name=exp7_lr3e-4`.

To make a permanent variant, create a new YAML file that inherits from an existing one:

```yaml
# configs/experiments/supervised/exp7_vit_cclr_long.yaml
_base_: exp7_vit_cclr.yaml
name: exp7_vit_cclr_long
train:
  epochs: 60
  scheduler: {name: cosine}
```

## 7. Outputs

```
outputs/<experiment name>/
├── summary.md / summary.csv / summary.json   mean ± std over seeds for every metric
├── per_seed.csv                              one row per seed
└── seed_<s>/
    ├── config.yaml                           exact resolved config of this run
    ├── run.log                               full log
    ├── metrics.json                          all scalar metrics of this seed
    ├── class_counts.json                     images per class per split
    ├── history.csv                           per-epoch loss/acc, lr, time, RAM, GPU memory
    ├── checkpoints/best.pt                   best weights (plain state_dict)
    ├── reports/<split>_classification_report.txt, <split>_metrics.json
    └── figures/
        ├── samples_train.png  training_curves.png  episode_<split>.png (few-shot)
        ├── confusion_matrix_<split>.png
        ├── tsne_<split>.png   gradcam_<split>.png
```

Metrics in `metrics.json` (all of them are aggregated over seeds):

| Group | Keys |
|---|---|
| Classification (per split: `val`, `test`, `target`) | `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `loss`, few-shot: `accuracy_ci95` |
| Training time | `train/wall_time_s`, `train/cpu_time_s`, `train/mean_epoch_time_s`, `train/epochs_run`, `train/best_epoch` |
| Training memory | `train/ram_peak_mb`, `train/gpu_peak_allocated_mb`, `train/gpu_peak_reserved_mb` |
| Evaluation cost | `<split>/eval_wall_time_s`, `<split>/eval_throughput_img_s`, `<split>/eval_ram_peak_mb`, `<split>/eval_gpu_peak_allocated_mb` |
| Inference | `inference/per_image_ms` (batch 1), `inference/throughput_img_s` |
| Model | `model/params_total`, `model/params_trainable`, `model/size_mb` |
| Whole run | `total/wall_time_s`, `total/ram_peak_mb`, ... |

GPU memory numbers are PyTorch's peak allocated/reserved memory on CUDA, or driver memory on
Apple MPS. RAM is the process's peak resident memory, sampled every 0.2 s.

The profiling tools also work on their own:

```python
from plantdisease.utils import ResourceMonitor, count_parameters, measure_inference_latency

with ResourceMonitor("my_block", device) as mon:
    ...  # anything
print(mon.usage)              # [my_block] 1m02.3s wall, RAM peak 2310 MB, GPU peak 1840 MB
print(mon.usage.as_dict())
```

## 8. Evaluating a trained checkpoint

`evaluate` skips training. It runs testing (reports and confusion matrices) and the XAI
figures for a checkpoint:

```bash
# Default checkpoint: outputs/<name>/seed_<seed>/checkpoints/best.pt
plantdisease evaluate configs/experiments/supervised/exp8_vit_ecclr.yaml --seed 0

# Any weights file, including best_model.pth / best_vit_model.pth from the old notebooks
plantdisease evaluate configs/experiments/supervised/exp4_efficientnet_ecclr.yaml \
    --checkpoint /path/to/best_model.pth --set output_dir=outputs/notebook_weights
```

Checkpoints from the original notebooks load directly, because the model attribute names
(`efficientnet.*`, `model.*`) are unchanged.

## 9. Using the code from a notebook

The Python API is what the CLI calls, so notebooks stay short:

```python
from plantdisease import load_config, ExperimentRunner, SupervisedExperiment

cfg = load_config(
    "configs/experiments/supervised/exp7_vit_cclr.yaml",
    overrides=["train.epochs=10", "xai.show=true"],   # xai.show=true displays figures inline
)

# Several seeds + mean ± std
summary = ExperimentRunner(cfg, seeds=[0, 1, 2]).run()
summary["test/accuracy"]          # {'mean': ..., 'std': ..., 'min': ..., 'max': ..., 'n': 3}

# Or a single run, step by step
exp = SupervisedExperiment(cfg, seed=0, run_dir="outputs/scratch/seed_0")
exp.build(); exp.train(); exp.test(); exp.explain()
exp.metrics
```

Individual components can also be used on their own (`DataModule`, `Trainer`, `Evaluator`,
`Tester`, `XAIVisualizer`, `GradCAM`, ...). See [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## 10. Running on Kaggle / Colab

```bash
!git clone https://github.com/afaroo01/CCLR-ECCLR && cd CCLR-ECCLR && pip install -q -e .
!plantdisease run configs/experiments/supervised/exp7_vit_cclr.yaml \
    --set "datasets.plantvillage_cclr.roots=[/kaggle/input/rembg-2-0-segmented-plantvillage-dataset/data]" \
          "datasets.plantvillage_ecclr.roots=[/kaggle/input/plantvillage-cclr-yolo-augmented-dataset/data]" \
          "datasets.plantvillage.roots=[/kaggle/input/plantvillage-dataset/color]" \
          "datasets.plantdoc.roots=[/kaggle/input/plantdoc-dataset/train,/kaggle/input/plantdoc-dataset/test]" \
          output_dir=/kaggle/working/outputs
```

## 11. Project structure

```
Experiments/
├── configs/
│   ├── base.yaml                 dataset paths + class maps, data/XAI defaults, seeds
│   ├── models/                   architecture + its training recipe
│   └── experiments/
│       ├── supervised/exp1..8    one file per supervised experiment
│       └── fewshot/              8 few-shot experiments (+ shared _fewshot_base.yaml)
├── src/plantdisease/
│   ├── config/      schema.py (typed dataclasses), loader.py (_base_ inheritance, overrides)
│   ├── data/        datasets.py, splits.py, transforms.py, mixing.py (CutMix/MixUp),
│   │                episodic.py (N-way K-shot sampler), datamodule.py (DataModule)
│   ├── models/      base.py (BaseClassifier interface), efficientnet.py, vit.py,
│   │                protonet.py, factory.py
│   ├── engine/      trainer.py (Trainer), evaluator.py (Evaluator), tester.py (Tester),
│   │                steps.py (batch strategies), callbacks.py, optim.py
│   ├── metrics/     classification metrics, confidence intervals
│   ├── xai/         gradcam.py, embeddings.py (t-SNE), visualizer.py (XAIVisualizer)
│   ├── experiments/ base.py, supervised.py, fewshot.py, runner.py (multi-seed), results.py
│   ├── utils/       profiling.py (time/memory), seed.py, device.py, io.py, logging.py
│   └── cli.py       `plantdisease` command
├── scripts/run_all.sh
├── tests/           pytest suite on synthetic data
├── notebooks/       original notebooks + quickstart.ipynb
└── data/            datasets (not versioned)
```

## 12. Differences from the original notebooks

The configs reproduce the notebooks' settings. Fixes and deliberate changes:

1. **CutMix loss (exp2, exp6).** The notebooks took `argmax` of the CutMix labels before the
   loss, which discards the label mixing. The loss now uses the soft labels; accuracy still
   uses the argmax.
2. **exp6 training transform.** In the notebook, all three subsets shared one `ImageFolder`,
   so the last assignment (eval transform) won and training ran **without** the timm
   augmentation. Each split now keeps its own transform.
3. **Few-shot confusion matrices.** The notebooks built them from episode-local labels
   (0..7 are re-shuffled in every episode), so the rows did not mean classes. Predictions are
   now mapped back to the real class ids.
4. **Few-shot backbones.** The `cclr-efficientnet` notebooks loaded the **ECCLR** checkpoint,
   and the ViT few-shot notebooks used raw PlantVillage as the source data. Now every
   few-shot config uses its matching backbone (CCLR→exp3/exp7, ECCLR→exp4/exp8) and source
   dataset. To reproduce the old pairing, override `model.checkpoint` / `data.source`.
5. **Grad-CAM.** One implementation now serves CNNs and ViTs, with per-image normalization.
   The ViT notebook version registered two forward hooks and no backward hook.
6. **No preprocessing copy.** `make_ds` copied and resized every image into a working folder.
   Images are now read in place and LANCZOS-resized on load (`data.preresize`), which gives
   the same pixels.
7. **Kept as in the notebooks but worth reviewing:** the few-shot transform
   (`fewshot_basic`) applies no ImageNet normalization (the backbones were trained with it)
   and a *random* crop at evaluation time. Use `--set fewshot.transform=fewshot_standard`
   for a deterministic, normalized few-shot evaluation.

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` when downloading pretrained weights (python.org Python on macOS) | Run `/Applications/Python\ 3.x/Install\ Certificates.command`, or `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` |
| `Dataset root '...' does not exist` | Fix the path in `configs/base.yaml` or with `--set paths.data_root=...`; use `data.target=null` if PlantDoc is missing |
| `Checkpoint not found: outputs/exp.../seed_3/...` (few-shot) | Run the matching supervised experiment for that seed first, or pass `model.checkpoint=` |
| `n_way=8 but only k classes have at least n_shot + n_query images` | Reduce `fewshot.n_query` / `fewshot.n_way`, or evaluate on a larger split |
| DataLoader worker errors / slow startup on macOS or Windows | `--set data.num_workers=0` |
| Out of GPU memory | `data.batch_size=16`, `train.amp=true` |
| MPS op not implemented | `export PYTORCH_ENABLE_MPS_FALLBACK=1` or `--set device=cpu` |
