"""Find which training composites used a given background image .

CCLR composites were made by resizing a background to the PlantVillage frame and pasting the
segmented leaf on top, so outside the leaf every pixel IS the background. For each background
listed in flagged_pairs.csv (the ones that near-duplicate a PlantDoc test image), this script
compares it with every composite of the same class and measures the fraction of pixels that
match (mean absolute RGB difference <= --tol at 64 x 64). Unrelated pairs match on a few per cent
of pixels; a composite built on that background matches on most of its non-leaf area.

Usage
  python scripts/trace_backgrounds.py --flagged outputs/near_duplicates_bg/flagged_pairs.csv ^
      --composites "<data>/plantvillage_cclr" "<data>/plant_village_ecclr" --out outputs/bg_trace
Outputs
  bg_trace.csv     one row per flagged background and composite set: best match fraction, number
                   of composites above --min-frac, best composite file
  pd_used.csv      the PlantDoc images whose duplicate background reached each training set
  bg_trace.jpg     background | best matching composite, for visual confirmation
Check the contact sheet: only visually confirmed matches count.
"""
from __future__ import annotations

import argparse
import csv
import ntpath
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
S = 64


def small(p: Path | str) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize((S, S), Image.BILINEAR)
    return np.asarray(im, dtype=np.int16)


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", type=Path, required=True)
    ap.add_argument("--composites", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/bg_trace"))
    ap.add_argument("--tol", type=int, default=12, help="per pixel mean |RGB difference|")
    ap.add_argument("--min-frac", type=float, default=0.25, help="match fraction counted as used")
    ap.add_argument("--all-classes", action="store_true", help="search composites of every class")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    bg_to_pd: dict[str, set[str]] = defaultdict(set)
    with open(a.flagged, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            bg_to_pd[r["query"]].add(r["plantdoc"])
    bgs = sorted(bg_to_pd)
    print(f"{len(bgs)} flagged backgrounds")
    bg_arr = {b: small(b) for b in tqdm(bgs, desc="Backgrounds")}

    rows, pd_used, sheet = [], defaultdict(set), []
    for root in a.composites:
        files, arr = [], {}
        for f in tqdm(images(root), desc=f"Composites {root.name}"):
            try:
                arr[f] = small(f)
                files.append(f)
            except Exception:  # unreadable file: skip
                pass
        by_cls: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            by_cls[f.relative_to(root).parts[0]].append(f)
        for b in bgs:
            cls = ntpath.basename(ntpath.dirname(b))
            cands = files if a.all_classes or cls not in by_cls else by_cls[cls]
            if not cands:
                continue
            stack = np.stack([arr[f] for f in cands])
            frac = (np.abs(stack - bg_arr[b]).mean(-1) <= a.tol).mean((1, 2))
            k = int(frac.argmax())
            n_used = int((frac >= a.min_frac).sum())
            rows.append({"composite_set": root.name, "background": b, "class": cls,
                         "best_frac": f"{frac[k]:.3f}", "n_composites_used": n_used,
                         "best_composite": str(cands[k]),
                         "plantdoc_duplicates": ";".join(sorted(bg_to_pd[b]))})
            if n_used:
                pd_used[root.name] |= bg_to_pd[b]
            sheet.append((frac[k], b, cands[k], root.name))

    with open(a.out / "bg_trace.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -float(r["best_frac"])))
    with open(a.out / "pd_used.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["composite_set", "plantdoc"])
        for s, ps in pd_used.items():
            w.writerows([s, p] for p in sorted(ps))

    sheet.sort(key=lambda t: -t[0])
    top = sheet[:60]
    if top:
        t, lab = 200, 18
        im = Image.new("RGB", (2 * t, len(top) * (t + lab)), "white")
        d = ImageDraw.Draw(im)
        for i, (fr, b, c, s) in enumerate(top):
            y = i * (t + lab)
            d.text((4, y + 2), f"{s} match {fr:.2f}  {Path(c).name[:40]}", fill="black")
            im.paste(Image.open(b).convert("RGB").resize((t, t)), (0, y + lab))
            im.paste(Image.open(c).convert("RGB").resize((t, t)), (t, y + lab))
        im.save(a.out / "bg_trace.jpg", quality=85)

    with open(a.out / "summary.txt", "w", encoding="utf-8") as fh:
        for root in a.composites:
            r = [x for x in rows if x["composite_set"] == root.name]
            used = [x for x in r if x["n_composites_used"] > 0]
            line = (f"{root.name}: {len(used)} of {len(r)} flagged backgrounds found in training "
                    f"composites; {len(pd_used[root.name])} PlantDoc images affected")
            print(line)
            fh.write(line + "\n")
        fh.write(f"tol={a.tol}, min_frac={a.min_frac}, size={S}\n")


if __name__ == "__main__":
    main()
