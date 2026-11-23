"""Full pipeline on synthetic data: multi-seed supervised run -> few-shot on its checkpoints."""

from plantdisease.cli import main
from plantdisease.experiments import ExperimentRunner
from plantdisease.utils.io import load_json


def test_supervised_multiseed_then_fewshot(load, tmp_path):
    cfg = load("experiments/supervised/exp1_efficientnet_autoaugment.yaml",
               "name=sup", "data.source=plantvillage")
    summary = ExperimentRunner(cfg, seeds=[0, 1]).run()
    exp_dir = tmp_path / "outputs" / "sup"
    assert summary["target/accuracy"]["n"] == 2
    for f in ("summary.md", "summary.csv", "per_seed.csv"):
        assert (exp_dir / f).is_file()
    run = exp_dir / "seed_0"
    for f in ("checkpoints/best.pt", "history.csv", "metrics.json", "run.log",
              "figures/gradcam_target.png", "figures/tsne_test.png",
              "figures/confusion_matrix_target.png", "reports/target_classification_report.txt"):
        assert (run / f).is_file(), f
    assert "train/wall_time_s" in load_json(run / "metrics.json")

    fs = load("experiments/fewshot/cclr_efficientnet_5shot.yaml", "name=fs",
              "data.source=plantvillage",
              f"model.checkpoint={exp_dir}/seed_{{seed}}/checkpoints/best.pt",
              "fewshot.n_shot=2", "fewshot.n_query=3", "fewshot.n_eval_tasks=3",
              "fewshot.eval_splits=[train,target]")
    fs_summary = ExperimentRunner(fs, seeds=[0, 1]).run()
    assert "target/accuracy_ci95" in fs_summary

    # resume skips finished seeds; summarize compares experiments
    ExperimentRunner(cfg, seeds=[0, 1], resume=True).run()
    main(["summarize", str(exp_dir), str(tmp_path / "outputs" / "fs"),
          "--output", str(tmp_path / "cmp.md")])
    assert "sup" in (tmp_path / "cmp.md").read_text()


def test_vit_cutmix_and_episodic_meta_training(load):
    cfg = load("experiments/supervised/exp6_vit_cutmix.yaml", "data.image_size=224",
               "xai.enabled=false", "data.target=null")
    ExperimentRunner(cfg, seeds=[0]).run()

    fs = load("experiments/fewshot/ecclr_vit_5shot.yaml", "data.source=plantvillage",
              "model.checkpoint=null", "data.image_size=224", "fewshot.n_shot=1",
              "fewshot.n_query=1", "fewshot.n_way=3", "fewshot.n_train_tasks=2",
              "fewshot.n_val_tasks=1", "fewshot.n_eval_tasks=2", "fewshot.eval_splits=[test]",
              "xai.enabled=false")
    summary = ExperimentRunner(fs, seeds=[0]).run()
    assert summary["train/epochs_run"]["mean"] == 1
