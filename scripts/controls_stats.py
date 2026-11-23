"""Tables and tests for the background-swap analysis and size-matched ECCLR (run after run_background_swap.bat).

Usage:  python scripts/controls_stats.py outputs outputs_cleanpd
The original exp3/exp4/exp7/exp8 results are read from the second folder (clean PlantDoc
re-evaluation), so that every condition is scored on the same clean PlantDoc set.
Writes outputs/comparison/controls.md and controls.csv.

Comparisons (Welch's t-test with Holm correction within each backbone, Hedges' g, 95% CI):
  background controls : agnostic, swap and bgonly against the re-generated class-consistent CCLR
  size-matched ECCLR  : against CCLR (exp3/exp7) and full-size ECCLR (exp4/exp8)
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import json  # noqa: E402

from stats_tests import compare, holm, load_per_seed, metric  # noqa: E402


def load_any(root: Path, exp: str) -> dict[int, dict]:
    """summary.json if present, otherwise the per-seed metrics.json files."""
    try:
        return load_per_seed(root, exp)
    except FileNotFoundError:
        out = {int(d.name.split("_")[1]): json.loads((d / "metrics.json").read_text())
               for d in sorted((root / exp).glob("seed_*")) if (d / "metrics.json").is_file()}
        if not out:
            raise
        return out

BACKBONES = {"EfficientNet-B0": ("effnet", "exp3_efficientnet_cclr", "exp4_efficientnet_ecclr"),
             "ViT-Tiny": ("vit", "exp7_vit_cclr", "exp8_vit_ecclr")}
METRICS = [("target/accuracy", "PD accuracy"), ("target/macro_f1", "PD macro F1")]


def fmt(v: list[float]) -> str:
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else "n/a")


def main(out_root: str = "outputs", orig_root: str | None = None) -> None:
    root = Path(out_root)
    orig = Path(orig_root) if orig_root else root
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    summary, tests = [], []

    for bb, (tag, cclr, ecclr) in BACKBONES.items():
        runs = {
            "CCLR, original (exp)": cclr,
            "Class consistent (regenerated)": f"ctrl_{tag}_consistent",
            "Class agnostic backgrounds": f"ctrl_{tag}_agnostic",
            "Cross class swapped backgrounds": f"ctrl_{tag}_swap",

            "ECCLR, size matched to CCLR": f"ecclr_sizematched_{tag}",
            "ECCLR, full size (exp)": ecclr,
        }
        data = {}
        for label, exp in runs.items():
            try:
                data[label] = load_any(orig if exp in (cclr, ecclr) else root, exp)
            except FileNotFoundError:
                print(f"skip {exp}: not run yet")
        for label, ps in data.items():
            summary.append({"backbone": bb, "condition": label, "n": len(ps),
                            **{m_label: fmt(list(metric(ps, m).values())) for m, m_label in METRICS}})

        pairs = [("Class agnostic backgrounds", "Class consistent (regenerated)"),
                 ("Cross class swapped backgrounds", "Class consistent (regenerated)"),
                 ("ECCLR, size matched to CCLR", "CCLR, original (exp)"),
                 ("ECCLR, size matched to CCLR", "ECCLR, full size (exp)")]
        for m, m_label in METRICS:
            fam = []
            for a, b in pairs:
                if a in data and b in data:
                    r = compare(metric(data[a], m), metric(data[b], m))
                    fam.append({"backbone": bb, "metric": m_label, "comparison": f"{a} vs {b}", **r})
            for r, p in zip(fam, holm([r["p_welch"] for r in fam]), strict=True):
                r["p_holm"] = p
            tests += fam

    lines = ["## Conditions (PlantDoc, mean ± SD over seeds)", "",
             "| Backbone | Condition | n | PD accuracy | PD macro F1 |", "|---|---|---|---|---|"]
    lines += [f"| {r['backbone']} | {r['condition']} | {r['n']} | {r['PD accuracy']} | {r['PD macro F1']} |"
              for r in summary]
    lines += ["", "## Tests (Welch's t-test, Holm correction within backbone and metric)", "",
              "| Backbone | Metric | Comparison | Difference | 95% CI | Hedges' g | p (Holm) |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['backbone']} | {r['metric']} | {r['comparison']} | {r['diff']:+.4f} | "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['g']:.2f} | {r['p_holm']:.4g} |" for r in tests]
    lines += ["", "All conditions scored on the clean PlantDoc set (near duplicates of the background pool removed)."]
    (out / "controls.md").write_text("\n".join(lines) + "\n")
    with open(out / "controls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "condition", "n", "PD accuracy", "PD macro F1"])
        w.writeheader()
        w.writerows(summary)
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
