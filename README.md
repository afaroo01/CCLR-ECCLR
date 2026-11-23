# Class-Consistent Pre-Augmentation Schemes for Single-Source Domain Generalisation in Plant Disease Classification

**Adil Farooq** and **Rudi Villing**
Department of Electronic Engineering, Maynooth University, Co. Kildare, Ireland

Code and data lists for the paper of the same title, submitted to *Computers and Electronics in
Agriculture* (under review). This repository implements CCLR and ECCLR, the two class-consistent
pre-augmentation schemes proposed in the paper.

Models are trained on PlantVillage (laboratory images) and tested on PlantDoc (field images)
for eight classes, without using any PlantDoc image for training or model selection.
**CCLR** segments each PlantVillage leaf and pastes it onto a natural background retrieved for
the same crop and health or disease status. **ECCLR** also transforms, rescales and randomly
places the leaves. Both are built once, offline, before training.

## Repository contents

| Folder | Contents |
|---|---|
| `src/plantdisease/` | Training and evaluation package (supervised and few-shot), installed as the `plantdisease` command |
| `configs/` | Base settings, model recipes and one config per experiment (`exp1` to `exp8`, few-shot) |
| `scripts/` | Batch runs, statistics, near-duplicate check, clean PlantDoc set, background-swap and ablation datasets |
| `notebooks/dataset_generation/` | Kaggle notebooks that generated the CCLR and ECCLR datasets |
| `notebooks/fewshot_dinov2/` | Kaggle notebooks for the DINOv2 few-shot rows of Tables 7 and 8 (frozen backbone, outputs removed) |
| `lists/` | Background search log, near-duplicate pairs and the PlantDoc images excluded from the clean test set |
| `tests/` | Test suite on synthetic data |
| `docs/` | Detailed usage guide and code architecture |

Experiment codes used throughout:

| | EfficientNet-B0 | ViT-Tiny |
|---|---|---|
| PlantVillage + AutoAugment | exp1 | exp5 |
| PlantVillage + CutMix | exp2 | exp6 |
| CCLR | exp3 | exp7 |
| ECCLR | exp4 | exp8 |

## Installation

Python 3.10 or newer. Install a CUDA build of PyTorch first if you have a GPU
(<https://pytorch.org/get-started/locally/>), then:

```bash
git clone https://github.com/afaroo01/CCLR-ECCLR
cd CCLR-ECCLR
pip install -e ".[analysis]"
plantdisease list        # check the installation
pytest                   # about one minute, synthetic data
```

The paper used Python 3.11, PyTorch 2.11.0 (CUDA 12.8) and timm 1.0.31 on an NVIDIA RTX 5060.

## Data

Put the datasets under `data/` (one sub-folder per class):

| Dataset | Source | Location |
|---|---|---|
| PlantVillage | <https://github.com/spMohanty/PlantVillage-Dataset> (`color/`) | `data/plantvillage/` |
| PlantDoc | <https://github.com/pratikkayal/PlantDoc-Dataset> (`train/` and `test/`) | `data/plantdoc/` |
| CCLR | Zenodo, <https://doi.org/10.5281/zenodo.24484377> (`CCLR.zip`) | `data/plantvillage_cclr/` |
| ECCLR | Zenodo, <https://doi.org/10.5281/zenodo.24484377> (`ECCLR.zip`) | `data/plant_village_ecclr/` |
| Background pool (optional) | re-downloaded from `lists/background_search_results.csv` | `data/backgrounds/` |

Only the eight classes used in the paper are read; other folders are ignored. Check a dataset with

```bash
plantdisease check-data configs/experiments/supervised/exp3_efficientnet_cclr.yaml
```

### Clean PlantDoc test set

94 PlantDoc images have a near duplicate in the web-retrieved background pool. They are listed in
`lists/plantdoc_excluded.csv`, and the supervised results in the paper use the remaining 511
images. To build the clean set:

```bash
python scripts/make_clean_plantdoc.py --flagged lists/near_duplicate_pairs.csv \
    --plantdoc data/plantdoc --out data/plantdoc_clean
```

The file names in these lists follow the renamed PlantDoc copy used for the experiments
(`data/plantdoc/<class>/plantdoc_<name>_<index>.jpg`). `lists/plantdoc_excluded_original_names.csv`
maps them to the file names of the original PlantDoc release, and
`scripts/map_excluded_to_plantdoc.py` recreates that mapping.

`lists/near_duplicate_pairs.csv` gives every flagged pair (background image, PlantDoc image,
pHash Hamming distance, cosine similarity); paths are relative to the class folders. The check
itself is `scripts/near_duplicates.py`, and `scripts/trace_backgrounds.py` finds which flagged
backgrounds were used in the training composites.

### Background pool

`lists/background_search_results.csv` is the DuckDuckGo search log (class, query, title, image
URL, page URL): 13 queries per class, up to 200 results per query. The images are third-party
web content and are not redistributed; `scripts/download_scraped.py` re-downloads those still
online:

```bash
python scripts/download_scraped.py --csv lists/background_search_results.csv --out data/backgrounds
```

## Reproducing the paper

All runs use seeds 0 to 4 on a fixed 70/20/10 split (split seed 42) and report mean ± standard
deviation. On Windows, `scripts/*.bat` run each block end to end from the repository root;
`scripts/run_all.sh` is the Linux equivalent of `run_all.bat`.

| Paper item | Command |
|---|---|
| Supervised results (Table 6) | `scripts\run_all.bat`, or `plantdisease run configs/experiments/supervised/<exp>.yaml` per experiment |
| Clean PlantDoc evaluation | `scripts\eval_clean_plantdoc.bat` then `python scripts/clean_pd_compare.py` |
| Few-shot evaluation (Tables 7 and 8) | `plantdisease run configs/experiments/fewshot/<config>.yaml` after the matching supervised run (frozen backbone, 1,000 episodes) |
| Statistical tests (Welch, Holm, Hedges' g) | `python scripts/stats_tests.py` |
| Background-swap analysis and size-matched ECCLR (Table 11) | `scripts\run_background_swap.bat` |
| ECCLR component ablation (Table 11) | `scripts\run_ecclr_ablation.bat` |
| Comparison with single-source DG methods (Table 10) | `scripts/dg_baselines.py` and `scripts/gen_dafusion.py` |

Datasets for the background-swap analysis (same-class, random-class and other-class backgrounds)
are built by `scripts/make_control_datasets.py`, the size-matched ECCLR set by
`scripts/make_sizematched_ecclr.py`, and the ablation variants by `scripts/make_ecclr_variants.py`.

Settings can be changed from the command line without editing files, for example

```bash
plantdisease run configs/experiments/supervised/exp3_efficientnet_cclr.yaml --seeds 0 \
    --set train.epochs=2 "datasets.plantdoc.roots=[data/plantdoc_clean]"
```

See [docs/USAGE.md](docs/USAGE.md) for all options, outputs (metrics, time and memory,
confusion matrices, Grad-CAM and t-SNE) and running on Kaggle or Colab, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the code structure.

## Training recipes

| | EfficientNet-B0 | ViT-Tiny |
|---|---|---|
| Weights | torchvision `IMAGENET1K_V1` | timm `vit_tiny_patch16_224.augreg_in21k_ft_in1k` |
| Optimiser | Adam, lr 1e-3, StepLR (step 5, gamma 0.1) | AdamW, lr 1e-4, constant |
| Early stopping | validation loss, patience 5 | validation accuracy, patience 5 |
| Epochs, batch size | up to 25, 32 | up to 25, 32 |
| Online augmentation | AutoAugment (ImageNet policy) + random resized crop | as for EfficientNet-B0; scale 0.5 to 1.0 for exp8; timm default augmentation for exp6 |

## Notes on the original notebooks

The reported results come from this package, not from the notebooks. The main differences are
listed in [docs/USAGE.md](docs/USAGE.md#12-differences-from-the-original-notebooks). In short,
every experiment is repeated over five seeds, few-shot evaluation uses 1,000 episodes instead
of 100, and the CCLR EfficientNet-B0 few-shot runs use the CCLR backbone (one original notebook
loaded the ECCLR backbone by mistake).

## Funding

This publication has emanated from research conducted with the financial support of Research
Ireland under Grant number 18/CRT/6222 (ADVANCE CRT).

## Citation

See `CITATION.cff`. The full reference will be added on publication.

## Licence

Code: MIT (see `LICENSE`). PlantVillage, PlantDoc and the web images keep their own licences.
