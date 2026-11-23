"""ECCLR component ablation datasets.

Rebuilds ECCLR with the leaf recipe of the ECCLR generation notebook (ecclr-rembg-2-0-segmented-plantvillage-yolo-aug),
reusing the u2net leaves cached by make_control_datasets.py and the same filtered background pool, so the
variants can be compared with the rebuilt class consistent CCLR (controls/cclr_consistent).

Variants (one composite per PlantVillage image, 9,360 images each, same size as CCLR):
  ecclr_1leaf : one leaf with the ECCLR geometric augmentation, rescaling and random placement
  ecclr_full  : the anchor leaf plus 0 to 4 extra leaves of the same class (1 to 5, uniform), each augmented

Leaf recipe (from the notebook): RandomHorizontalFlip(0.1), RandomVerticalFlip(0.1),
RandomPerspective(0.6, p=1), RandomAffine(degrees 30 to 70, translate 0.1 to 0.3, scale 0.5 to 0.75),
RandomRotation(0 to 180); leaf then resized by 1.5 to 2.0 and pasted at a random position on the class
consistent background resized to 512 x 512 (up to half the leaf may fall outside the frame).
The notebook's file name bug is not reproduced: each output is named after its anchor PlantVillage image.

Usage
  python scripts/make_ecclr_variants.py --leaves "<data>/controls/leaves_rgba" --backgrounds "<bg folder>" ^
      --exclude outputs/near_duplicates_bg/flagged_pairs.csv --out "<data>/controls"
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).parent))
from make_control_datasets import CLASSES, images, load_excluded, readable  # noqa: E402

CANVAS = 512
AUG = v2.Compose([
    v2.RandomHorizontalFlip(p=0.1),
    v2.RandomVerticalFlip(p=0.1),
    v2.RandomPerspective(distortion_scale=0.6, p=1.0),
    v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
    v2.RandomRotation(degrees=(0, 180)),
])


def compose_one(bg_path: Path, leaf_paths: list[Path], rng: random.Random) -> Image.Image:
    bg = ImageOps.exif_transpose(Image.open(bg_path)).convert("RGBA").resize((CANVAS, CANVAS))
    for lp in leaf_paths:
        leaf = AUG(Image.open(lp).convert("RGBA"))
        f = rng.uniform(1.5, 2.0)
        w, h = int(leaf.width * f), int(leaf.height * f)
        leaf = leaf.resize((w, h), Image.LANCZOS)
        x = rng.randint(-w // 2, CANVAS - w // 2)
        y = rng.randint(-h // 2, CANVAS - h // 2)
        bg.paste(leaf, (x, y), leaf)
    return bg.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaves", type=Path, required=True)
    ap.add_argument("--backgrounds", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["ecclr_1leaf", "ecclr_full"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    excluded = load_excluded(a.exclude)
    bgs = {}
    for c in CLASSES:
        allp = images(a.backgrounds / c)
        bgs[c] = readable([p for p in allp if f"{c}/{p.name}" not in excluded])
        print(f"[backgrounds] {c}: {len(bgs[c])} kept of {len(allp)}")

    for var in a.variants:
        rng = random.Random(a.seed)
        torch.manual_seed(a.seed)
        n_leaves = []
        for c in CLASSES:
            leaves = images(a.leaves / c)
            if not leaves:
                raise SystemExit(f"No cached leaves in {a.leaves / c}; run make_control_datasets.py first")
            dst = a.out / var / c
            dst.mkdir(parents=True, exist_ok=True)
            for lp in leaves:
                out = dst / (lp.stem + ".jpg")
                k = rng.randint(0, 4) if var == "ecclr_full" else 0
                if out.exists():
                    n_leaves.append(k + 1)
                    continue
                extra = rng.sample(leaves, k) if k else []
                compose_one(rng.choice(bgs[c]), [lp] + extra, rng).save(out, quality=95)
                n_leaves.append(k + 1)
            print(f"[{var}] {c}: {len(leaves)} images")
        dist = {n: n_leaves.count(n) for n in sorted(set(n_leaves))}
        print(f"[{var}] total {len(n_leaves)} images; leaves per image: {dist}")


if __name__ == "__main__":
    main()
