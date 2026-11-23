"""Compare PlantDoc results with and without the images duplicated in the scraped web pool.

Usage:  python scripts/clean_pd_compare.py outputs outputs_cleanpd
Writes outputs_cleanpd/comparison/clean_plantdoc.md and .csv.

For every supervised experiment and seed, the SAME trained checkpoint is evaluated on the full
PlantDoc set (outputs/) and on the clean set (outputs_cleanpd/), so differences are paired by
seed (paired t-test). Method comparisons on the clean set use Welch's t-test with Holm correction,
as in stats_tests.py.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from stats_tests import COMPARISONS, SUPERVISED, compare, holm  # noqa: E402

METRICS = [("target/accuracy", "PD accuracy"), ("target/macro_f1", "PD macro F1")]


def per_seed(root: Path, exp: str, key: str) -> dict[int, float]:
    out = {}
    for d in sorted((root / exp).glob("seed_*")):
        f = d / "metrics.json"
        if f.is_file():
            m = json.loads(f.read_text())
            if key in m:
                out[int(d.name.split("_")[1])] = m[key]
    return out


def ms(v: list[float]) -> str:
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else "n/a")


def main(full_root: str = "outputs", clean_root: str = "outputs_cleanpd") -> None:
    full, clean = Path(full_root), Path(clean_root)
    out = clean / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    lines += ["## Same checkpoints, full PlantDoc against PlantDoc without duplicated images", "",
              "| Backbone | Method | Metric | Full PD | Clean PD | Change (paired) | p (paired t) |",
              "|---|---|---|---|---|---|---|"]
    for backbone, exps in SUPERVISED.items():
        for method, exp in exps.items():
            for key, label in METRICS:
                f, c = per_seed(full, exp, key), per_seed(clean, exp, key)
                seeds = sorted(set(f) & set(c))
                if not seeds:
                    continue
                d = [c[s] - f[s] for s in seeds]
                p = (stats.ttest_rel([c[s] for s in seeds], [f[s] for s in seeds]).pvalue
                     if len(seeds) > 1 and st.stdev(d) > 0 else float("nan"))
                row = {"backbone": backbone, "method": method, "metric": label, "n": len(seeds),
                       "full": ms([f[s] for s in seeds]), "clean": ms([c[s] for s in seeds]),
                       "change": f"{st.mean(d):+.4f} ± {st.stdev(d) if len(d) > 1 else 0:.4f}",
                       "p_paired": f"{p:.3g}"}
                rows.append(row)
                lines.append(f"| {backbone} | {method} | {label} | {row['full']} | {row['clean']} | "
                             f"{row['change']} | {row['p_paired']} |")

    lines += ["", "## Method against baselines on the clean PlantDoc set (Welch, Holm)", "",
              "| Backbone | Metric | Comparison | Difference | 95% CI | Hedges' g | p (Holm) |",
              "|---|---|---|---|---|---|---|"]
    for backbone, exps in SUPERVISED.items():
        for key, label in METRICS:
            fam = []
            for a, b in COMPARISONS:
                va, vb = per_seed(clean, exps[a], key), per_seed(clean, exps[b], key)
                if len(va) > 1 and len(vb) > 1:
                    fam.append({"cmp": f"{a} vs {b}", **compare(va, vb)})
            for r, ph in zip(fam, holm([r["p_welch"] for r in fam]), strict=True):
                lines.append(f"| {backbone} | {label} | {r['cmp']} | {r['diff']:+.4f} | "
                             f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['g']:.2f} | {ph:.3g} |")

    (out / "clean_plantdoc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "clean_plantdoc.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["backbone"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
