"""Size-matched ECCLR.

ECCLR has about 27% more images than CCLR. This makes an ECCLR subset with exactly the CCLR
number of images in every class (random, seeded), so CCLR and ECCLR can be compared at equal
dataset size. Files are hard-linked (no extra disk space); if that fails they are copied.

Usage
  python scripts/make_sizematched_ecclr.py --cclr "<data>/plantvillage_cclr" \
      --ecclr "<data>/plant_village_ecclr" --out "<data>/ecclr_sizematched"
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cclr", type=Path, required=True)
    ap.add_argument("--ecclr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    rng = random.Random(a.seed)
    total = 0
    for cls in sorted(d.name for d in a.cclr.iterdir() if d.is_dir()):
        n = len(images(a.cclr / cls))
        pool = images(a.ecclr / cls)
        if len(pool) < n:
            raise SystemExit(f"{cls}: ECCLR has {len(pool)} images, fewer than CCLR's {n}")
        (a.out / cls).mkdir(parents=True)
        for p in rng.sample(pool, n):
            dst = a.out / cls / p.name
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
        print(f"{cls}: {n} of {len(pool)}")
        total += n
    print(f"Total {total} images -> {a.out}")


if __name__ == "__main__":
    main()
