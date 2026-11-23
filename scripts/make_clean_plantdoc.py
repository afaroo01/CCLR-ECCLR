"""PlantDoc without the images that have a near-duplicate in the scraped web pool.

Reads one or more flagged_pairs.csv files from near_duplicates.py and copies (hard links) every
PlantDoc image that is NOT flagged into a new folder with the same class sub-folders.
By default an image is excluded when its pHash Hamming distance is <= 8 (near-identical);
add --include-cosine to also exclude pairs flagged only by deep-feature similarity.

Usage
  python scripts/make_clean_plantdoc.py --flagged outputs/near_duplicates_scraped/flagged_pairs.csv \
      --plantdoc "<data>/plantdoc" --out "<data>/plantdoc_clean"
Writes <out>/excluded.csv listing the removed images.
"""
from __future__ import annotations

import argparse
import csv
import ntpath
import os
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", type=Path, nargs="+", required=True)
    ap.add_argument("--plantdoc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hamming", type=int, default=8)
    ap.add_argument("--include-cosine", action="store_true")
    a = ap.parse_args()

    excluded: dict[str, tuple[str, int, float]] = {}       # "class/file" -> (query, hamming, cosine)
    for f in a.flagged:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ham, cos = int(r["hamming"]), float(r["cosine"])
                if ham <= a.max_hamming or a.include_cosine:
                    p = r["plantdoc"]
                    key = f"{ntpath.basename(ntpath.dirname(p))}/{ntpath.basename(p)}"
                    if key not in excluded or ham < excluded[key][1]:
                        excluded[key] = (r["query"], ham, cos)

    if a.out.exists():
        shutil.rmtree(a.out)
    kept, removed = Counter(), Counter()
    for cls_dir in sorted(d for d in a.plantdoc.iterdir() if d.is_dir()):
        (a.out / cls_dir.name).mkdir(parents=True)
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT or p.name.startswith("._"):
                continue
            if f"{cls_dir.name}/{p.name}" in excluded:
                removed[cls_dir.name] += 1
                continue
            try:
                os.link(p, a.out / cls_dir.name / p.name)
            except OSError:
                shutil.copy2(p, a.out / cls_dir.name / p.name)
            kept[cls_dir.name] += 1

    with open(a.out / "excluded.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["plantdoc_image", "matched_scraped_image", "hamming", "cosine"])
        for k, (q, h, c) in sorted(excluded.items()):
            w.writerow([k, q, h, f"{c:.4f}"])
    for c in sorted(kept | removed):
        print(f"{c}: kept {kept[c]}, removed {removed[c]}")
    print(f"Total kept {sum(kept.values())}, removed {sum(removed.values())} -> {a.out}")


if __name__ == "__main__":
    main()
