r"""Phase 0: pull the embedded JPEG preview out of RAW files - the core speed trick.

Run:  ks previews                             every RAW in data\test_photos
      ks previews "E:\Sharma Wedding"         every RAW in a folder
      ks previews "E:\Wedding\IMG_0001.CR3"   one file

Prints, per camera: preview size, milliseconds per photo, and how many need rotating.
Saves up to 3 sample previews per camera in data\scratch\previews so you can open them,
and a full table in data\scratch\preview_report.csv.
"""
from __future__ import annotations

import csv
import io
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rawpy  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from engine.cli import path_arg  # noqa: E402
from engine.config import RAW_TYPES, SCRATCH_DIR, TEST_PHOTOS, worker_count  # noqa: E402
from engine.meta import BATCH_SIZE, open_exiftool, tag  # noqa: E402
from engine.scan import find_photos  # noqa: E402

OUT_DIR = SCRATCH_DIR / "previews"
SAMPLES_PER_CAMERA = 3
ROTATION = {0: "none", 3: "180", 5: "90 left", 6: "90 right"}


def key(p: Path | str) -> str:
    return os.path.normcase(os.path.abspath(p))


def read_preview(path: Path):
    """Return (jpeg_bytes or None, flip, kind). LibRaw reads only the bytes it needs."""
    try:
        raw = rawpy.imread(str(path))
    except rawpy.LibRawIOError:
        # Rare Windows paths LibRaw cannot open directly: read through Python instead.
        with open(path, "rb") as f:
            raw = rawpy.imread(f)
    with raw:
        flip = raw.sizes.flip
        thumb = raw.extract_thumb()
    if thumb.format == rawpy.ThumbFormat.JPEG:
        return thumb.data, flip, "jpeg"
    return None, flip, "bitmap"


def main() -> None:
    target = path_arg(TEST_PHOTOS)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(p for p in find_photos(target) if p.suffix.lower() in RAW_TYPES)
    else:
        print(f"Not found: {target}")
        sys.exit(1)
    if not files:
        print(f"No RAW files in {target}. Add RAW photos and run again.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Camera model for every file, using one ExifTool process
    models: dict[str, str] = {}
    with open_exiftool() as et:
        for i in range(0, len(files), BATCH_SIZE):
            for row in et.get_tags([str(p) for p in files[i:i + BATCH_SIZE]], tags=["Model"]):
                models[key(row.get("SourceFile", ""))] = str(tag(row, "Model") or "unknown")

    results = []
    saved: dict[str, int] = defaultdict(int)
    for path in tqdm(files, unit="photo", desc="Extracting previews"):
        camera = models.get(key(path), "unknown")
        start = time.perf_counter()
        try:
            data, flip, kind = read_preview(path)
            ms = (time.perf_counter() - start) * 1000
            width, height = Image.open(io.BytesIO(data)).size if data else (0, 0)
            error = ""
        except Exception as e:  # noqa: BLE001
            data, flip, kind, ms, width, height = None, 0, "error", 0.0, 0, 0
            error = type(e).__name__
        if data and saved[camera] < SAMPLES_PER_CAMERA:
            safe = "".join(c if c.isalnum() else "_" for c in camera)
            (OUT_DIR / f"{safe}_{path.stem}_preview.jpg").write_bytes(data)
            saved[camera] += 1
        results.append({
            "file": str(path), "camera": camera, "kind": kind, "error": error,
            "ms": round(ms, 1), "width": width, "height": height,
            "preview_mb": round(len(data) / 1e6, 2) if data else 0,
            "raw_mb": round(path.stat().st_size / 1e6, 1),
            "rotation": ROTATION.get(flip, str(flip)),
        })

    with open(SCRATCH_DIR / "preview_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    by_camera = defaultdict(list)
    for r in results:
        by_camera[r["camera"]].append(r)

    print(f"\n{'Camera':<24}{'Files':>6}  {'Preview size':<14}{'ms/photo':>9}"
          f"{'Preview MB':>11}{'RAW MB':>8}{'Rotated':>9}{'Failed':>8}")
    for camera, rs in sorted(by_camera.items(), key=lambda kv: -len(kv[1])):
        good = [r for r in rs if r["kind"] == "jpeg"]
        sizes = [f"{r['width']}x{r['height']}" for r in good]
        common = max(set(sizes), key=sizes.count) if sizes else "-"
        ms = statistics.median(r["ms"] for r in good) if good else 0
        pmb = statistics.median(r["preview_mb"] for r in good) if good else 0
        rmb = statistics.median(r["raw_mb"] for r in rs)
        rotated = sum(r["rotation"] != "none" for r in good)
        failed = len(rs) - len(good)
        print(f"{camera[:23]:<24}{len(rs):>6}  {common:<14}{ms:>9.0f}"
              f"{pmb:>11.1f}{rmb:>8.0f}{rotated:>9}{failed:>8}")

    good_all = [r for r in results if r["kind"] == "jpeg"]
    if good_all:
        median_ms = statistics.median(r["ms"] for r in good_all)
        est = median_ms * 4000 / 1000 / worker_count()
        print(f"\nMedian {median_ms:.0f} ms per RAW on one core. Preview step alone for a "
              f"4,000-photo wedding with {worker_count()} workers: about {est / 60:.1f} min on this laptop.")
    bad = [r for r in results if r["kind"] != "jpeg"]
    if bad:
        print(f"\n{len(bad)} file(s) without a JPEG preview:")
        for r in bad[:10]:
            print(f"    {r['file']}  ({r['error'] or r['kind']})")
    print(f"\nSample previews: {OUT_DIR}")
    print(f"Full table:      {SCRATCH_DIR / 'preview_report.csv'}")


if __name__ == "__main__":
    main()
