"""Tables and tests for the ECCLR component ablation.

Usage:  python scripts/ablation_stats.py outputs
Chain per backbone: rebuilt CCLR (ctrl_*_consistent) -> ECCLR one leaf (abl_*_ecclr_1leaf)
-> ECCLR 1 to 5 leaves (abl_*_ecclr_full). Welch's t test, Holm correction within backbone and metric.
Writes outputs/comparison/ecclr_ablation.md.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from controls_stats import load_any  # noqa: E402
from stats_tests import compare, holm, metric  # noqa: E402

STEPS = [("CCLR rebuilt (1 leaf, no augmentation)", "ctrl_{b}_consistent"),
         ("ECCLR, 1 leaf (augmentation and placement)", "abl_{b}_ecclr_1leaf"),
         ("ECCLR, 1 to 5 leaves (augmentation and placement)", "abl_{b}_ecclr_full")]
PAIRS = [(1, 0, "effect of augmentation and placement"), (2, 1, "effect of extra leaves"),
         (2, 0, "total")]
METRICS = [("target/accuracy", "PD accuracy"), ("target/macro_f1", "PD macro F1")]


def main(root: str = "outputs") -> None:
    root_p = Path(root)
    lines = ["## Conditions (clean PlantDoc, mean ± SD over seeds)", "",
             "| Backbone | Condition | n | PD accuracy | PD macro F1 |", "|---|---|---|---|---|"]
    tests = ["", "## Tests (Welch, Holm within backbone and metric)", "",
             "| Backbone | Metric | Comparison | Difference | 95% CI | Hedges' g | p (Holm) |",
             "|---|---|---|---|---|---|---|"]
    for b, name in (("effnet", "EfficientNet-B0"), ("vit", "ViT-Tiny")):
        data = {}
        for label, pat in STEPS:
            try:
                data[label] = load_any(root_p, pat.format(b=b))
            except FileNotFoundError:
                print(f"missing: {pat.format(b=b)}")
        for label, _ in STEPS:
            if label in data:
                cells = []
                for m, _ in METRICS:
                    v = list(metric(data[label], m).values())
                    cells.append(f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}")
                lines.append(f"| {name} | {label} | {len(data[label])} | {cells[0]} | {cells[1]} |")
        for m, mlabel in METRICS:
            fam = []
            for i, j, what in PAIRS:
                a, c = STEPS[i][0], STEPS[j][0]
                if a in data and c in data:
                    va, vc = metric(data[a], m), metric(data[c], m)
                    if len(va) > 1 and len(vc) > 1:
                        fam.append((what, compare(va, vc)))
            for (what, r), ph in zip(fam, holm([r["p_welch"] for _, r in fam]), strict=True):
                tests.append(f"| {name} | {mlabel} | {what} | {r['diff']:+.4f} | [{r['ci_lo']:+.4f}, "
                             f"{r['ci_hi']:+.4f}] | {r['g']:.2f} | {ph:.3g} |")
    text = "\n".join(lines + tests) + "\n"
    out = root_p / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ecclr_ablation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
