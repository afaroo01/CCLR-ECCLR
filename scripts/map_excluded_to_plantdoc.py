"""Map the excluded PlantDoc images to the file names of the original PlantDoc download.

lists/plantdoc_excluded.csv names images as they appear in the local, renamed PlantDoc copy
used for the experiments (data/plantdoc/<class>/plantdoc_<name>_<index>.jpg). This script
matches each of those images to the original PlantDoc release (train/ and test/ folders) by
perceptual hash, so the clean test set can be rebuilt from a fresh download.

Usage
  python scripts/map_excluded_to_plantdoc.py --excluded lists/plantdoc_excluded.csv \
      --local data/plantdoc --original path/to/PlantDoc-Dataset \
      --out lists/plantdoc_excluded_original_names.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imagehash
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(path: Path):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excluded", type=Path, required=True)
    ap.add_argument("--local", type=Path, required=True, help="renamed PlantDoc copy used in the experiments")
    ap.add_argument("--original", type=Path, required=True, help="original PlantDoc release (train/ and test/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hamming", type=int, default=2)
    a = ap.parse_args()

    originals = []
    for p in sorted(a.original.rglob("*")):
        if p.suffix.lower() in EXTS:
            h = phash(p)
            if h is not None:
                originals.append((p.relative_to(a.original).as_posix(), h))

    with open(a.excluded, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out, missing = [], 0
    for r in rows:
        h = phash(a.local / r["plantdoc_image"])
        best = min(((h - oh, name) for name, oh in originals), default=(None, ""))
        if h is None or best[0] is None or best[0] > a.max_hamming:
            missing += 1
            out.append([r["plantdoc_image"], "", ""])
        else:
            out.append([r["plantdoc_image"], best[1], best[0]])

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["local_image", "original_plantdoc_image", "hamming"])
        w.writerows(out)
    print(f"{len(rows) - missing} of {len(rows)} excluded images matched; written to {a.out}")


if __name__ == "__main__":
    main()
