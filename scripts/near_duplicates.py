"""Near-duplicate check between the scraped backgrounds (or any image set) and PlantDoc.

Two complementary tests for every (query image, PlantDoc image) pair:
  1. Perceptual hash (pHash, 64 bit): Hamming distance <= --hash-thresh flags near-identical
     images (resized, recompressed, small colour changes).
  2. Deep features (ImageNet EfficientNet-B0, global pooled): cosine similarity >= --cos-thresh
     flags images that are crops or edits of each other, which pHash can miss.
Flagged pairs are written to a CSV and shown side by side in a contact sheet for manual review.
A pair counts as a duplicate only after visual confirmation.

Usage
  pip install imagehash
  python scripts/near_duplicates.py --query "<backgrounds>" --plantdoc "<data>/plantdoc" --out outputs/near_duplicates
  (also useful: --query "<data>/plantvillage_cclr" to check the training composites)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imagehash
import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def all_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))


def load(p: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def phashes(paths: list[Path]) -> tuple[np.ndarray, list[Path], list[str]]:
    """pHash every readable image; unreadable files are skipped and returned separately."""
    hashes, ok, bad = [], [], []
    for p in tqdm(paths, desc="pHash", unit="img"):
        try:
            hashes.append(imagehash.phash(load(p)).hash.flatten())
            ok.append(p)
        except Exception as e:  # corrupt, truncated, or not really an image
            bad.append(f"{p}\t{type(e).__name__}: {e}")
    return np.array(hashes, dtype=bool), ok, bad


@torch.no_grad()
def features(paths: list[Path], device: str, bs: int = 64) -> torch.Tensor:
    w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = tvm.efficientnet_b0(weights=w)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    tf = w.transforms()
    out = []
    for i in tqdm(range(0, len(paths), bs), desc="Features", unit="batch"):
        x = torch.stack([tf(load(p)) for p in paths[i:i + bs]]).to(device)
        out.append(torch.nn.functional.normalize(model(x), dim=1).cpu())
    return torch.cat(out)


def sheet(pairs: list[dict], out: Path, n: int = 40) -> None:
    pairs = pairs[:n]
    if not pairs:
        return
    t = 180
    img = Image.new("RGB", (2 * t + 10, len(pairs) * (t + 24)), "white")
    d = ImageDraw.Draw(img)
    for i, r in enumerate(pairs):
        y = i * (t + 24)
        for j, key in enumerate(("query", "plantdoc")):
            img.paste(load(Path(r[key])).resize((t, t)), (j * (t + 10), y + 20))
        d.text((2, y + 2), f"hamming {r['hamming']}  cosine {r['cosine']:.3f}", fill="black")
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=Path, required=True, help="backgrounds or other images to check")
    ap.add_argument("--plantdoc", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/near_duplicates"))
    ap.add_argument("--hash-thresh", type=int, default=8)
    ap.add_argument("--cos-thresh", type=float, default=0.95)
    ap.add_argument("--no-features", action="store_true", help="pHash only (no deep features)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q, pd = all_images(a.query), all_images(a.plantdoc)
    print(f"{len(q)} query images, {len(pd)} PlantDoc images")
    (hq, q, bad_q), (hp, pd, bad_pd) = phashes(q), phashes(pd)
    bad = bad_q + bad_pd
    if bad:
        (a.out / "unreadable.txt").write_text("\n".join(bad) + "\n", encoding="utf-8")
        print(f"Skipped {len(bad)} unreadable files (listed in unreadable.txt)")
    if a.no_features:
        fq, fp = torch.zeros(len(q), 1), torch.zeros(len(pd), 1)
        a.cos_thresh = float("inf")
    else:
        fq, fp = features(q, device), features(pd, device)

    rows, closest_h, closest_c = [], [], []
    for i in tqdm(range(len(q)), desc="Comparing", unit="img"):
        ham = (hq[i] != hp).sum(axis=1)
        cos = (fp @ fq[i]).numpy()
        closest_h.append(int(ham.min()))
        closest_c.append(float(cos.max()))
        for j in np.where((ham <= a.hash_thresh) | (cos >= a.cos_thresh))[0]:
            rows.append({"query": str(q[i]), "plantdoc": str(pd[j]),
                         "hamming": int(ham[j]), "cosine": float(cos[j])})
    rows.sort(key=lambda r: (r["hamming"], -r["cosine"]))

    with open(a.out / "flagged_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "plantdoc", "hamming", "cosine"])
        w.writeheader()
        w.writerows(rows)
    sheet(rows, a.out / "flagged_pairs.jpg")
    summary = (
        f"Query images: {len(q)}\nPlantDoc images: {len(pd)}\n"
        f"Unreadable files skipped: {len(bad)}\n"
        f"Thresholds: pHash Hamming <= {a.hash_thresh} of 64 bits, cosine >= {a.cos_thresh}\n"
        f"Flagged pairs: {len(rows)} ({len({r['query'] for r in rows})} query images, "
        f"{len({r['plantdoc'] for r in rows})} PlantDoc images)\n"
        f"Closest pHash distance per query image: min {min(closest_h)}, median {int(np.median(closest_h))}\n"
        f"Highest cosine per query image: max {max(closest_c):.3f}, median {np.median(closest_c):.3f}\n"
        "Review flagged_pairs.jpg: only visually confirmed pairs are duplicates.\n")
    (a.out / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
