"""Command-line interface.

    plantdisease run         CONFIG [--seeds 0 1 2] [--set key=value ...] [--resume]
    plantdisease evaluate    CONFIG [--seed S] [--checkpoint PATH] [--set ...]
    plantdisease check-data  CONFIG [--set ...]
    plantdisease show-config CONFIG [--set ...]
    plantdisease summarize   EXPERIMENT_DIR [...] [--metrics ...] [--output FILE]
    plantdisease list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# Command-line runs only save figures. The default Windows backend (TkAgg) can abort the
# process with "Tcl_AsyncDelete: async handler deleted by the wrong thread" when figures are
# garbage-collected from a background thread, so use the non-interactive Agg backend.
matplotlib.use("Agg", force=True)

from .config import config_to_yaml, load_config  # noqa: E402
from .experiments.results import (  # noqa: E402
    DEFAULT_REPORT_METRICS,
    compare_experiments,
    rows_to_markdown,
)
from .utils.io import save_csv  # noqa: E402
from .utils.logging import get_logger  # noqa: E402


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", type=Path, help="experiment YAML, e.g. configs/experiments/...")
    p.add_argument(
        "--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE",
        help="override config values, e.g. --set train.epochs=10 paths.data_root=/mnt/data",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plantdisease", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="train + test + explain for one or more seeds; aggregate")
    _add_config_args(p)
    p.add_argument("--seeds", type=int, nargs="+", help="override `seeds` from the config")
    p.add_argument("--resume", action="store_true", help="skip seeds that already have metrics")

    p = sub.add_parser("evaluate", help="test + explain an already-trained checkpoint")
    _add_config_args(p)
    p.add_argument("--seed", type=int, default=None, help="default: first seed in config")
    p.add_argument("--checkpoint", default=None,
                   help="weights file (default: <output>/<name>/seed_<seed>/checkpoints/best.pt)")

    p = sub.add_parser("check-data", help="verify dataset paths/classes and print split sizes")
    _add_config_args(p)

    p = sub.add_parser("show-config", help="print the fully resolved config")
    _add_config_args(p)

    p = sub.add_parser("summarize", help="compare several experiments (mean ± std table)")
    p.add_argument("experiment_dirs", nargs="+", type=Path)
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_REPORT_METRICS))
    p.add_argument("--output", type=Path, default=None, help=".md or .csv file to write")

    sub.add_parser("list", help="list registered models, transforms, mixers, optimizers ...")
    return parser


def _cmd_run(args) -> None:
    from .experiments import ExperimentRunner

    cfg = load_config(args.config, args.overrides)
    ExperimentRunner(cfg, seeds=args.seeds, resume=args.resume).run()


def _cmd_evaluate(args) -> None:
    from .experiments import ExperimentRunner

    cfg = load_config(args.config, args.overrides)
    runner = ExperimentRunner(cfg)
    seed = args.seed if args.seed is not None else cfg.seeds[0]
    checkpoint = args.checkpoint
    if checkpoint is None:
        default = runner.run_dir(seed) / "checkpoints" / "best.pt"
        checkpoint = str(default) if default.is_file() else None  # few-shot: config backbone
    runner.build_experiment(seed).evaluate(checkpoint)


def _cmd_check_data(args) -> None:
    from .data import DataModule

    cfg = load_config(args.config, args.overrides)
    log = get_logger()
    for key in filter(None, (cfg.data.source, cfg.data.target)):
        log.info("%s -> %s", key, cfg.datasets[key].roots)
    counts = DataModule(cfg, seed=cfg.seeds[0]).class_counts()
    for split, per_class in counts.items():
        log.info("%-6s %6d images", split, sum(per_class.values()))
        for cls, n in per_class.items():
            log.info("         %-32s %6d%s", cls, n, "   <-- EMPTY" if n == 0 else "")


def _cmd_summarize(args) -> None:
    rows = compare_experiments(args.experiment_dirs, args.metrics)
    table = rows_to_markdown(rows)
    print(table)
    if args.output:
        if args.output.suffix == ".csv":
            save_csv(rows, args.output)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(table)


def _cmd_list(_args) -> None:
    from .data import BATCH_MIXERS, TRANSFORMS
    from .engine import OPTIMIZERS, SCHEDULERS
    from .experiments import EXPERIMENTS
    from .models import MODELS

    for reg in (EXPERIMENTS, MODELS, TRANSFORMS, BATCH_MIXERS, OPTIMIZERS, SCHEDULERS):
        print(f"{reg.name:<12} {', '.join(reg.keys())}")


COMMANDS = {
    "run": _cmd_run,
    "evaluate": _cmd_evaluate,
    "check-data": _cmd_check_data,
    "show-config": lambda a: print(config_to_yaml(load_config(a.config, a.overrides))),
    "summarize": _cmd_summarize,
    "list": _cmd_list,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (FileNotFoundError, KeyError, ValueError) as err:
        get_logger().error("%s: %s", type(err).__name__, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
