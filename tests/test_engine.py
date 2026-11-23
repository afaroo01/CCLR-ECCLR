import pytest
import torch

from plantdisease.config import ModelConfig
from plantdisease.engine.callbacks import EarlyStopping
from plantdisease.experiments.results import ResultAggregator
from plantdisease.models import MODELS, PrototypicalNetwork
from plantdisease.utils import ResourceMonitor
from plantdisease.xai import GradCAM


def test_early_stopping_min_and_max():
    es = EarlyStopping("min", patience=2)
    for epoch, v in enumerate([1.0, 0.5, 0.6, 0.7], 1):
        es.step(v, epoch)
    assert es.best_epoch == 2 and es.should_stop
    es = EarlyStopping("max", patience=0)
    es.step(0.1, 1), es.step(0.9, 2)
    assert es.best == 0.9 and not es.should_stop


def test_aggregator_mean_and_sample_std():
    s = ResultAggregator.aggregate({0: {"acc": 0.8}, 1: {"acc": 0.9}, 2: {"acc": 1.0}})
    assert s["acc"]["mean"] == pytest.approx(0.9) and s["acc"]["std"] == pytest.approx(0.1)


@pytest.mark.parametrize("name,variant,size", [
    ("efficientnet", "efficientnet_b0", 64), ("vit", "vit_tiny_patch16_224", 224),
])
def test_models_embeddings_gradcam_protonet(name, variant, size):
    model = MODELS.build(name, ModelConfig(name=name, variant=variant, pretrained=False), 8)
    x = torch.randn(2, 3, size, size)
    assert model(x).shape == (2, 8) and model.embed(x).ndim == 2
    with GradCAM(model) as cam:
        heat = cam(x)
    assert heat.shape == (2, size, size) and heat.min() >= 0 and heat.max() <= 1
    scores = PrototypicalNetwork(model)(x, torch.tensor([0, 1]), x)
    assert scores.shape == (2, 2)


def test_resource_monitor_records_time_and_ram():
    with ResourceMonitor("x") as mon:
        sum(range(10**5))
    assert mon.usage.wall_time_s > 0 and mon.usage.ram_peak_mb > 0
