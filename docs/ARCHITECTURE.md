# Architecture and extension guide

## Layers and dependencies

```
cli.py ──► experiments/runner.py (ExperimentRunner: seeds loop, ResultAggregator)
                │
                ▼
        experiments/{supervised,fewshot}.py   (Experiment subclasses: build/train/test)
                │ uses
   ┌────────────┼──────────────┬──────────────┬────────────────┐
   ▼            ▼              ▼              ▼                ▼
 data/       models/        engine/         xai/          utils/profiling
 DataModule  BaseClassifier Trainer         XAIVisualizer  ResourceMonitor
             + registry     Evaluator       GradCAM
                            Tester          t-SNE
                            BatchProcessor
```

Each layer depends only on the layers below it. `config/schema.py` (typed dataclasses) is the
contract every layer reads.

## Responsibilities (single responsibility)

| Class | Only job |
|---|---|
| `LeafImageDataset` | find images under one or more roots and map folders to canonical labels |
| `TransformSubset` | give a split its own transform without rescanning the folders |
| `DatasetSplitter` | produce deterministic train/val/test indices |
| `DataModule` | turn the config into datasets and standard or episodic `DataLoader`s |
| `TaskSampler` | sample N-way K-shot episodes and collate them |
| `BaseClassifier` (+ EfficientNet / ViT) | logits, embeddings, Grad-CAM target layer |
| `PrototypicalNetwork` | few-shot scoring from support/query embeddings |
| `BatchProcessor` | unpack a batch, run the model, return logits/targets/labels/predictions |
| `Trainer` | epoch loop, AMP, gradient clipping, scheduler, early stopping, best checkpoint |
| `Evaluator` | no-grad pass over a loader, collecting loss, predictions and resource use |
| `Tester` | run the `Evaluator` on several splits, write reports, return flat metrics |
| `XAIVisualizer` | every figure (samples, curves, confusion matrix, t-SNE, Grad-CAM, episodes) |
| `GradCAM` | heatmap computation only; no plotting |
| `ResourceMonitor` | wall/CPU time, peak RAM, peak GPU memory of a code block (nest-safe) |
| `Experiment` | template method: seed → build → train → test → profile → explain → save |
| `ExperimentRunner` | repeat an `Experiment` over seeds, with resume |
| `ResultAggregator` | mean / std / min / max over seeds, written as JSON, CSV and Markdown |

## No duplicated loops (strategy pattern)

Supervised training and few-shot meta-training share **one** `Trainer`. Supervised and
episodic evaluation share **one** `Evaluator`. The only difference is the batch format, which
lives in `engine/steps.py`:

- `StandardBatchProcessor`: `(images, labels)`, where labels are indices or CutMix soft labels.
- `EpisodicBatchProcessor`: `(support, support_labels, query, query_labels, class_ids)`.
  It maps episode-local predictions back to global class ids.

## Extending (open/closed via registries)

Components are chosen by name from YAML through small registries, so adding one never
requires editing the existing code.

### New architecture

```python
# src/plantdisease/models/convnext.py
import timm, torch
from .base import BaseClassifier
from .registry import MODELS

@MODELS.register("convnext")
class ConvNeXtClassifier(BaseClassifier):
    def __init__(self, cfg, num_classes):
        super().__init__(num_classes)
        self.model = timm.create_model(cfg.variant, pretrained=cfg.pretrained,
                                       num_classes=num_classes)
    def forward_features(self, x):
        return self.model.forward_head(self.model.forward_features(x), pre_logits=True)
    def forward_head(self, f):
        return self.model.get_classifier()(f)
    def gradcam_target_layer(self):
        return self.model.stages[-1]
```

Import it in `models/__init__.py`, add `configs/models/convnext_tiny.yaml`, and reference it
from an experiment with `_base_`. The trainer, few-shot code, Grad-CAM and t-SNE then work
with it unchanged.

### New augmentation

```python
# in data/transforms.py
@TRANSFORMS.register("randaugment")
def randaugment(cfg, model_cfg):
    return v2.Compose([*_to_float(), v2.RandAugment(), _normalize(cfg),
                       _random_resized_crop(cfg)])
```

Select it with `data.train_transform: randaugment`. Batch-level mixes (like CutMix) go in
`data/mixing.py` under `BATCH_MIXERS`.

### New dataset

Add an entry under `datasets:` in `configs/base.yaml`, with `roots` and an optional
`class_map`, and set `data.source` or `data.target` to its key. No code is needed.

### New experiment type

Subclass `Experiment`, implement `build`, `train`, `test` and the `backbone` property, and
decorate the class with `@EXPERIMENTS.register("my_type")`. Configs then select it with
`type: my_type`, and multi-seed running and aggregation work automatically.

### New optimizer / scheduler

Register a builder in `engine/optim.py` (`OPTIMIZERS` / `SCHEDULERS`).

## Configuration

- `config/schema.py` defines every key with its type and default. Unknown keys and wrong
  types are errors.
- YAML files compose with `_base_: [parent1.yaml, parent2.yaml]`. Later files override
  earlier ones, and the child overrides all of them.
- `${paths.data_root}` interpolation keeps all paths relative to one root.
- `{seed}` in `model.checkpoint` is filled in per run, which links few-shot runs to the
  supervised checkpoint of the same seed.
