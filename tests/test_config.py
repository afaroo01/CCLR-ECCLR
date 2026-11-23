from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from plantdisease.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ALL = sorted(p for p in CONFIGS.glob("experiments/**/*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_every_experiment_config_resolves(path):
    cfg = load_config(path)
    assert cfg.name and cfg.type in ("supervised", "fewshot")
    assert cfg.data.source in cfg.datasets
    assert len(cfg.data.classes) == 8


def test_data_root_override_propagates():
    cfg = load_config(ALL[0], ["paths.data_root=/mnt/x"])
    assert cfg.datasets["plantdoc"].roots == ["/mnt/x/plantdoc/train", "/mnt/x/plantdoc/test"]


def test_model_recipe_inherited():
    vit = load_config(CONFIGS / "experiments/supervised/exp8_vit_ecclr.yaml")
    assert vit.train.optimizer.name == "adamw" and vit.data.rrc_scale == [0.5, 1.0]
    eff = load_config(CONFIGS / "experiments/supervised/exp2_efficientnet_cutmix.yaml")
    assert eff.train.scheduler.name == "step" and eff.data.batch_mixer == "cutmix"


def test_unknown_key_rejected():
    with pytest.raises(ConfigKeyError):
        load_config(ALL[0], ["train.epochz=3"])
