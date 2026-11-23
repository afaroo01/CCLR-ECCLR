"""Datasets for the background-swap analysis (same-class, random-class and other-class backgrounds).

All conditions are generated with the SAME pipeline, so they differ only in which backgrounds are used:

  consistent : each leaf on a background of its own class          (re-generated CCLR, the reference)
  agnostic   : each leaf on a background drawn from the pooled backgrounds of all classes
  swap       : each leaf on a background of a fixed different class of a different crop
  bgonly     : backgrounds alone, labelled with their class (no leaf)

If PlantDoc accuracy for "agnostic" stays close to "consistent", the gain does not come from
class cues in the backgrounds. If "swap" is close too, the model is not using background class
cues even when they point to the wrong class. If "bgonly" is near chance (1/8 = 0.125) on
PlantDoc, the backgrounds alone carry no transferable class information.

Steps
  1. segment : rembg (u2net) removes the background of every PlantVillage image -> RGBA leaves (cached)
  2. compose : one composite per PlantVillage image for each condition (same counts as CCLR)

Layout (--layout)
  cclr   (default) as in the original CCLR notebook: the whole background is resized to the
         PlantVillage frame and the segmented leaf is pasted at full size in its original position.
  random leaf scaled to --scale and placed at a random position on a random background crop.

Backgrounds that cannot be decoded are skipped, and backgrounds listed in --exclude
(flagged_pairs.csv from near_duplicates.py: near duplicates of PlantDoc test images) are removed
from the pool, so the controls carry no PlantDoc leakage.

Usage
  pip install rembg onnxruntime
  python scripts/make_control_datasets.py --pv "<data>/plantvillage" --backgrounds "<backgrounds>" --out "<data>/controls"

The backgrounds folder must have one sub-folder per class with the canonical names
(Apple___healthy, Apple___rust, ...), i.e. the scraped DuckDuckGo backgrounds used for CCLR.
"""
from __future__ import annotations

import argparse
import csv
import ntpath
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

CLASSES = [
    "Apple___healthy", "Apple___rust", "Bell_pepper___bacterial_spot", "Bell_pepper___healthy",
    "Grape___black_rot", "Grape___healthy", "Tomato___healthy", "Tomato___septoria_leaf_spot",
]
# swap: shift by 2 in the list above, so every class gets the backgrounds of a different crop
SWAP = {c: CLASSES[(i + 2) % len(CLASSES)] for i, c in enumerate(CLASSES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def readable(paths: list[Path]) -> list[Path]:
    ok = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(p)
        except Exception:
            pass
    return ok


def load_excluded(files: list[Path]) -> set[str]:
    """'class/file' keys of every background listed in flagged_pairs.csv files."""
    out = set()
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                q = r["query"]
                out.add(f"{ntpath.basename(ntpath.dirname(q))}/{ntpath.basename(q)}")
    return out


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))


def segment(pv: Path, leaves: Path) -> None:
    """rembg on every PlantVillage image; skips files already done."""
    from rembg import new_session, remove  # imported here so compose-only runs need no rembg

    session = new_session("u2net")
    for c in CLASSES:
        src, dst = pv / c, leaves / c
        if not src.is_dir():
            raise SystemExit(f"Missing class folder: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        todo = [p for p in images(src) if not (dst / (p.stem + ".png")).exists()]
        print(f"[segment] {c}: {len(todo)} to do")
        for i, p in enumerate(todo, 1):
            out = remove(Image.open(p).convert("RGB"), session=session)
            out.save(dst / (p.stem + ".png"))
            if i % 250 == 0:
                print(f"   {i}/{len(todo)}")


def fit_background(bg: Image.Image, size: int, rng: random.Random) -> Image.Image:
    """Random square crop (at least 60% of the short side), resized to size x size."""
    bg = ImageOps.exif_transpose(bg).convert("RGB")
    w, h = bg.size
    side = int(min(w, h) * rng.uniform(0.6, 1.0))
    x, y = rng.randint(0, w - side), rng.randint(0, h - side)
    return bg.crop((x, y, x + side, y + side)).resize((size, size), Image.LANCZOS)


def place_leaf(canvas: Image.Image, leaf: Image.Image, scale: tuple[float, float],
               rng: random.Random) -> Image.Image:
    leaf = leaf.crop(leaf.getbbox() or (0, 0, *leaf.size))           # tight box around the leaf
    size = canvas.size[0]
    target = int(size * rng.uniform(*scale))
    k = target / max(leaf.size)
    leaf = leaf.resize((max(1, int(leaf.width * k)), max(1, int(leaf.height * k))), Image.LANCZOS)
    if rng.random() < 0.5:
        leaf = ImageOps.mirror(leaf)
    x, y = rng.randint(0, size - leaf.width), rng.randint(0, size - leaf.height)
    canvas.paste(leaf, (x, y), leaf)
    return canvas


def cclr_composite(bg_path: Path, leaf_path: Path, size: int) -> Image.Image:
    """Original CCLR layout: background resized to the frame, leaf pasted at full size."""
    leaf = Image.open(leaf_path).convert("RGBA").resize((size, size), Image.LANCZOS)
    bg = ImageOps.exif_transpose(Image.open(bg_path)).convert("RGB").resize((size, size), Image.LANCZOS)
    bg.paste(leaf, (0, 0), leaf)
    return bg


def compose(leaves: Path, backgrounds: Path, out: Path, modes: list[str], size: int,
            scale: tuple[float, float], seed: int, layout: str = "cclr",
            exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    bgs, dropped = {}, {}
    for c in CLASSES:
        allp = images(backgrounds / c)
        keep = readable([p for p in allp if f"{c}/{p.name}" not in exclude])
        bgs[c], dropped[c] = keep, len(allp) - len(keep)
        if not keep:
            raise SystemExit(f"No usable backgrounds for {c} in {backgrounds / c}")
    pool = [p for c in CLASSES for p in bgs[c]]
    print("[backgrounds kept] " + ", ".join(f"{c}: {len(bgs[c])} (dropped {dropped[c]})" for c in CLASSES))

    for mode in modes:
        root = out / f"cclr_{mode}"
        if root.exists():
            shutil.rmtree(root)
        rng = random.Random(f"{seed}-{mode}")                       # same seed -> same placements
        for c in CLASSES:
            (root / c).mkdir(parents=True)
            if mode == "bgonly":
                for i, p in enumerate(bgs[c]):
                    fit_background(Image.open(p), size, rng).save(root / c / f"bg_{i:05d}.jpg", quality=95)
                continue
            for p in images(leaves / c):
                if mode == "consistent":
                    bg_path = rng.choice(bgs[c])
                elif mode == "agnostic":
                    bg_path = rng.choice(pool)
                elif mode == "swap":
                    bg_path = rng.choice(bgs[SWAP[c]])
                else:
                    raise SystemExit(f"Unknown mode {mode}")
                if layout == "cclr":
                    img = cclr_composite(bg_path, p, size)
                else:
                    img = place_leaf(fit_background(Image.open(bg_path), size, rng),
                                     Image.open(p).convert("RGBA"), scale, rng)
                img.save(root / c / (p.stem + ".jpg"), quality=95)
        n = sum(len(images(root / c)) for c in CLASSES)
        print(f"[compose] {mode}: {n} images -> {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pv", type=Path, help="PlantVillage root with the 8 canonical class folders")
    ap.add_argument("--backgrounds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--leaves", type=Path, help="RGBA leaves (default: <out>/leaves_rgba)")
    ap.add_argument("--modes", nargs="+", default=["consistent", "agnostic", "swap"])
    ap.add_argument("--layout", choices=["cclr", "random"], default="cclr")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="flagged_pairs.csv files; listed backgrounds are not used")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--scale", type=float, nargs=2, default=[0.55, 0.9],
                    help="leaf longest side as a fraction of the image side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-segment", action="store_true", help="leaves already segmented")
    a = ap.parse_args()
    leaves = a.leaves or a.out / "leaves_rgba"
    if not a.skip_segment:
        if a.pv is None:
            raise SystemExit("--pv is required unless --skip-segment is given")
        segment(a.pv, leaves)
    compose(leaves, a.backgrounds, a.out, a.modes, a.size, tuple(a.scale), a.seed,
            a.layout, load_excluded(a.exclude))
    print("Swap mapping: " + "; ".join(f"{k} <- {v}" for k, v in SWAP.items()))


if __name__ == "__main__":
    main()
