"""Statistical tests over the multi-seed results.

Usage:  python scripts/stats_tests.py outputs
Writes outputs/comparison/stats_tests.md and stats_tests.csv, stats_domain_shift.md/.csv.

What it computes
1. Domain shift per experiment: DS = source test acc - target acc and relative
   DS = DS / source test acc, each per seed, with mean, sample SD and a t-based 95% CI.
2. Pairwise comparisons of each method against the baselines, separately for each backbone
   and metric (target accuracy, target macro F1, DS):
   - Welch's t-test (independent samples). Primary test: the runs being compared use
     different training data, so pairing by seed is a design choice, not a true pairing.
   - Paired t-test by seed (same seed and the same fixed split, seed 42). Secondary.
   - Holm correction within each family (one backbone x one metric x one test type).
   - Effect size: Hedges' g (bias-corrected Cohen's d, pooled SD) and the 95% CI of the
     mean difference.
3. Few-shot: ECCLR vs CCLR backbone for each model and shot count (target accuracy, macro F1).

With n = 5 per group, an exact Wilcoxon signed-rank test cannot reach p < 0.05 (minimum
two-sided p = 0.0625), so it is not reported.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

SUPERVISED = {
    "EfficientNet-B0": {
        "AutoAugment": "exp1_efficientnet_autoaugment",
        "CutMix": "exp2_efficientnet_cutmix",
        "CCLR": "exp3_efficientnet_cclr",
        "ECCLR": "exp4_efficientnet_ecclr",
    },
    "ViT-Tiny": {
        "AutoAugment": "exp5_vit_autoaugment",
        "CutMix": "exp6_vit_cutmix",
        "CCLR": "exp7_vit_cclr",
        "ECCLR": "exp8_vit_ecclr",
    },
}
# (method, baseline) pairs, tested within each backbone
COMPARISONS = [
    ("CCLR", "AutoAugment"),
    ("CCLR", "CutMix"),
    ("ECCLR", "AutoAugment"),
    ("ECCLR", "CutMix"),
    ("ECCLR", "CCLR"),
]
FEWSHOT = {
    ("EfficientNet-B0", 5): ("fewshot_cclr_efficientnet_5shot", "fewshot_ecclr_efficientnet_5shot"),
    ("EfficientNet-B0", 10): ("fewshot_cclr_efficientnet_10shot", "fewshot_ecclr_efficientnet_10shot"),
    ("ViT-Tiny", 5): ("fewshot_cclr_vit_5shot", "fewshot_ecclr_vit_5shot"),
    ("ViT-Tiny", 10): ("fewshot_cclr_vit_10shot", "fewshot_ecclr_vit_10shot"),
}


# ---------------------------------------------------------------------------------- loading
def load_per_seed(root: Path, exp: str) -> dict[int, dict]:
    f = root / exp / "summary.json"
    if not f.is_file():
        raise FileNotFoundError(f"{f} not found. Run the experiment first.")
    raw = json.loads(f.read_text())["per_seed"]
    return {int(k): v for k, v in raw.items()}


def metric(per_seed: dict[int, dict], name: str) -> dict[int, float]:
    """Per-seed values of a metric; 'ds' and 'rel_ds' are derived."""
    out = {}
    for s, m in per_seed.items():
        if name in ("ds", "rel_ds"):
            src, tgt = m["test/accuracy"], m["target/accuracy"]
            out[s] = src - tgt if name == "ds" else (src - tgt) / src
        else:
            out[s] = m[name]
    return out


# ------------------------------------------------------------------------------ statistics
def mean_ci(x: list[float], conf: float = 0.95) -> tuple[float, float, float, float]:
    n = len(x)
    m = st.mean(x)
    sd = st.stdev(x) if n > 1 else 0.0
    h = stats.t.ppf((1 + conf) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return m, sd, m - h, m + h


def hedges_g(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * st.variance(a) + (nb - 1) * st.variance(b)) / (na + nb - 2))
    if sp == 0:
        return float("inf") if st.mean(a) != st.mean(b) else 0.0
    d = (st.mean(a) - st.mean(b)) / sp
    return d * (1 - 3 / (4 * (na + nb) - 9))


def welch_diff_ci(a: list[float], b: list[float], conf: float = 0.95) -> tuple[float, float]:
    va, vb, na, nb = st.variance(a), st.variance(b), len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        d = st.mean(a) - st.mean(b)
        return d, d
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    h = stats.t.ppf((1 + conf) / 2, df) * se
    d = st.mean(a) - st.mean(b)
    return d - h, d + h


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def safe_p(res) -> float:
    p = float(res.pvalue)
    return 1.0 if math.isnan(p) else p  # identical constant samples


def compare(a_map: dict[int, float], b_map: dict[int, float]) -> dict:
    a, b = list(a_map.values()), list(b_map.values())
    common = sorted(set(a_map) & set(b_map))
    welch = safe_p(stats.ttest_ind(a, b, equal_var=False))
    paired = safe_p(stats.ttest_rel([a_map[s] for s in common], [b_map[s] for s in common]))
    lo, hi = welch_diff_ci(a, b)
    return {
        "mean_a": st.mean(a), "mean_b": st.mean(b), "diff": st.mean(a) - st.mean(b),
        "ci_lo": lo, "ci_hi": hi, "g": hedges_g(a, b),
        "p_welch": welch, "p_paired": paired, "n": f"{len(a)}/{len(b)}",
    }


# ----------------------------------------------------------------------------------- output
def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def write(rows: list[dict], header: list[str], path: Path, title: str, note: str) -> None:
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n\n" + note + "\n")
    print("\n".join(lines) + "\n")


def f4(x: float) -> str:
    return f"{x:.4f}"


def main(out_root: str = "outputs") -> None:
    root = Path(out_root)
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    # 1. domain shift with CIs
    ds_rows = []
    for backbone, exps in SUPERVISED.items():
        for method, exp in exps.items():
            ps = load_per_seed(root, exp)
            row = {"backbone": backbone, "method": method, "n": len(ps)}
            for name, label in [("ds", "DS"), ("rel_ds", "relative DS")]:
                m, sd, lo, hi = mean_ci(list(metric(ps, name).values()))
                row[f"{label} mean ± SD"] = f"{m:.4f} ± {sd:.4f}"
                row[f"{label} 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
            ds_rows.append(row)
    write(ds_rows, list(ds_rows[0]), out / "stats_domain_shift", "Domain shift with 95% CIs",
          "DS = source test accuracy - target accuracy; relative DS = DS / source test accuracy. "
          "CIs are t-based over seeds (df = n - 1).")

    # 2. supervised pairwise tests, Holm within backbone x metric x test
    rows = []
    for backbone, exps in SUPERVISED.items():
        data = {k: load_per_seed(root, v) for k, v in exps.items()}
        for name, label in [("target/accuracy", "target accuracy"),
                            ("target/macro_f1", "target macro F1"), ("ds", "DS")]:
            family = []
            for a, b in COMPARISONS:
                r = compare(metric(data[a], name), metric(data[b], name))
                family.append({"backbone": backbone, "metric": label,
                               "comparison": f"{a} vs {b}", **r})
            for key in ("p_welch", "p_paired"):
                for r, p in zip(family, holm([r[key] for r in family]), strict=True):
                    r[key + "_holm"] = p
            rows += family

    header = ["backbone", "metric", "comparison", "n", "mean A", "mean B", "difference",
              "95% CI of difference", "Hedges' g", "p Welch", "p Welch (Holm)", "",
              "p paired", "p paired (Holm)"]
    fmt = [{
        "backbone": r["backbone"], "metric": r["metric"], "comparison": r["comparison"],
        "n": r["n"], "mean A": f4(r["mean_a"]), "mean B": f4(r["mean_b"]),
        "difference": f"{r['diff']:+.4f}", "95% CI of difference": f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
        "Hedges' g": f"{r['g']:.2f}", "p Welch": f"{r['p_welch']:.4g}",
        "p Welch (Holm)": f"{r['p_welch_holm']:.4g}", "": stars(r["p_welch_holm"]),
        "p paired": f"{r['p_paired']:.4g}", "p paired (Holm)": f"{r['p_paired_holm']:.4g}",
    } for r in rows]
    write(fmt, header, out / "stats_tests", "Supervised: method vs baseline",
          "A is the first method named in the comparison. Welch's t-test is the primary test; the "
          "paired t-test (by seed) is shown for reference. Holm correction is applied within each "
          "backbone and metric (5 comparisons). Stars refer to Holm-adjusted Welch p: * < 0.05, "
          "** < 0.01, *** < 0.001. For DS a negative difference means A has the smaller shift.")

    # 3. few-shot: ECCLR vs CCLR backbone
    fs_rows = []
    for name, label in [("target/accuracy", "target accuracy"), ("target/macro_f1", "target macro F1")]:
        family = []
        for (backbone, shots), (cclr, ecclr) in FEWSHOT.items():
            r = compare(metric(load_per_seed(root, ecclr), name),
                        metric(load_per_seed(root, cclr), name))
            family.append({"backbone": backbone, "shots": shots, "metric": label, **r})
        for r, p in zip(family, holm([r["p_welch"] for r in family]), strict=True):
            r["p_welch_holm"] = p
        fs_rows += family
    fs_fmt = [{
        "backbone": r["backbone"], "shots": r["shots"], "metric": r["metric"],
        "ECCLR": f4(r["mean_a"]), "CCLR": f4(r["mean_b"]), "difference": f"{r['diff']:+.4f}",
        "95% CI of difference": f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
        "Hedges' g": f"{r['g']:.2f}", "p Welch": f"{r['p_welch']:.4g}",
        "p Welch (Holm)": f"{r['p_welch_holm']:.4g}", "": stars(r["p_welch_holm"]),
    } for r in fs_rows]
    write(fs_fmt, list(fs_fmt[0]), out / "stats_fewshot", "Few-shot: ECCLR vs CCLR backbone",
          "Mean over seeds of the per-seed episode mean (1000 episodes). Holm correction across the "
          "4 backbone x shot settings within each metric.")
    print(f"Written to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
