"""Re-download the scraped web images listed in results.csv (the DuckDuckGo search log).

results.csv has no header; columns are: class, query, title, image_url, page_url.
Each unique image URL is downloaded once, checked to be a readable image, converted to RGB JPEG
and saved as <out>/<class>/<hash>.jpg. A URL that appears under several classes is saved in
each of those class folders (so per-class pools stay complete) and flagged in the manifest.
Runs are resumable: files already saved are skipped.

Recovery options for links that fail directly:
  --referer  retry with the source page (from the CSV) as the HTTP Referer, as a browser would
             send it; recovers images behind hotlink protection (some 403 errors)
  --wayback  look up the image in the Internet Archive Wayback Machine and download the archived
             copy; recovers some expired links (404, dead sites)
The manifest records which route recovered each image (direct, referer or wayback).

Usage
  python scripts/download_scraped.py --csv "<path>/results.csv" --out "data/backgrounds"
  python scripts/download_scraped.py --csv ... --out ... --referer --wayback      (second pass)

Outputs
  <out>/<class>/*.jpg          downloaded images
  <out>/manifest.csv           one row per (class, url): status, file, query, title, multi_class
  printed summary              downloaded / failed per class

Note: many links will have expired since the original scrape, and this recovers the full search
results, not the filtered subset that was kept as backgrounds. Use for internal research only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps

HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset reconstruction)"}
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/130.0 Safari/537.36",
           "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}
WAYBACK_API = "https://archive.org/wayback/available"
WAYBACK_RAW = "https://web.archive.org/web/{ts}id_/{url}"
MIN_SIDE = 64


def fetch(url: str, timeout: float, headers: dict | None = None) -> tuple[str, bytes | None]:
    try:
        r = requests.get(url, headers=headers or HEADERS, timeout=timeout)
        if r.status_code != 200:
            return f"http_{r.status_code}", None
        img = Image.open(io.BytesIO(r.content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        if min(img.size) < MIN_SIDE:
            return "too_small", None
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        return "ok", buf.getvalue()
    except requests.RequestException as e:
        return f"error_{type(e).__name__}", None
    except Exception:  # noqa: BLE001  (not an image, truncated file, ...)
        return "not_image", None


def wayback(url: str, timeout: float) -> tuple[str, bytes | None]:
    try:
        r = requests.get(WAYBACK_API, params={"url": url}, headers=HEADERS, timeout=timeout)
        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        if not snap.get("available"):
            return "wayback_none", None
        return fetch(WAYBACK_RAW.format(ts=snap["timestamp"], url=url), timeout)
    except Exception:  # noqa: BLE001
        return "wayback_error", None


def get_image(url: str, page: str, a) -> tuple[str, str, bytes | None]:
    """Try direct, then (optionally) with Referer, then (optionally) the Wayback Machine."""
    st, data = fetch(url, a.timeout)
    if data:
        return st, "direct", data
    if a.referer and page.startswith("http"):
        st2, data = fetch(url, a.timeout, {**BROWSER, "Referer": page})
        if data:
            return st2, "referer", data
    if a.wayback:
        st3, data = wayback(url, a.timeout)
        if data:
            return st3, "wayback", data
    return st, "", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--referer", action="store_true", help="retry failures with the source page as Referer")
    ap.add_argument("--wayback", action="store_true", help="retry failures via the Wayback Machine")
    a = ap.parse_args()

    rows = defaultdict(list)                       # url -> [(class, query, title)]
    pages: dict[str, str] = {}                     # url -> source page
    with open(a.csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(f):
            if len(r) >= 4 and r[3].startswith("http"):
                rows[r[3]].append((r[0], r[1], r[2]))
                pages.setdefault(r[3], r[4] if len(r) > 4 else "")
    print(f"{sum(len(v) for v in rows.values())} results, {len(rows)} unique image URLs")

    def target(url: str, cls: str) -> Path:
        return a.out / cls / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg")

    todo = [u for u, v in rows.items() if not all(target(u, c).exists() for c, _, _ in v)]
    print(f"{len(rows) - len(todo)} already downloaded, {len(todo)} to fetch")
    status: dict[str, str] = {u: "ok" for u in rows if u not in todo}
    source: dict[str, str] = {}
    old = a.out / "manifest.csv"
    if old.is_file():
        with open(old, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    source[r["image_url"]] = r.get("source") or "direct"
    if a.wayback:
        a.workers = min(a.workers, 4)              # be gentle with the Internet Archive

    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(get_image, u, pages.get(u, ""), a): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            st, src, data = fut.result()
            status[u] = st
            if data:
                source[u] = src
            if data:
                for cls in {c for c, _, _ in rows[u]}:
                    p = target(u, cls)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}")

    a.out.mkdir(parents=True, exist_ok=True)
    per_class, reasons = Counter(), Counter()
    with open(a.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "status", "source", "file", "query", "title", "image_url", "multi_class"])
        for u, v in rows.items():
            classes = sorted({c for c, _, _ in v})
            for cls in classes:
                q, t = next((q, t) for c, q, t in v if c == cls)
                st = status.get(u, "not_run")
                w.writerow([cls, st, source.get(u, "") if st == "ok" else "",
                            target(u, cls).name if st == "ok" else "", q, t, u, len(classes) > 1])
                per_class[(cls, st == "ok")] += 1
            reasons[status.get(u, "not_run")] += 1
    for cls in sorted({c for c, _ in per_class}):
        print(f"{cls}: {per_class[(cls, True)]} downloaded, {per_class[(cls, False)]} failed")
    print("By status (unique URLs):", dict(reasons.most_common()))
    print("Recovered by route:", dict(Counter(source[u] for u in rows if status.get(u) == "ok"
                                              and u in source).most_common()))
    print(f"Manifest: {a.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
