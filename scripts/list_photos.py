r"""Phase 0: list every photo in a folder with its camera and capture time.

Run:  ks list                       (uses data\test_photos)
      ks list "E:\Sharma Wedding"   (any folder or drive)
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from engine.cli import path_arg  # noqa: E402
from engine.config import TEST_PHOTOS  # noqa: E402
from engine.meta import BATCH_SIZE, error_of, open_exiftool, tag  # noqa: E402
from engine.scan import find_photos, raw_jpeg_pairs  # noqa: E402

TAGS = ["Model", "DateTimeOriginal"]


def show(title: str, counts: Counter) -> None:
    print(f"\n{title}")
    for name, n in counts.most_common():
        print(f"  {n:>6}  {name}")


def main() -> None:
    root = path_arg(TEST_PHOTOS)
    if not root.exists():
        print(f"Folder not found: {root}")
        sys.exit(1)

    start = time.perf_counter()
    photos = sorted(find_photos(root))
    scan_s = time.perf_counter() - start
    print(f"Folder: {root}")
    print(f"Found {len(photos)} photos in {scan_s:.2f} s")
    if not photos:
        print("Nothing to read yet. Add photos to the folder and run again.")
        return

    rows = []
    with open_exiftool() as et, tqdm(total=len(photos), unit="photo",
                                     desc="Reading metadata") as bar:
        for i in range(0, len(photos), BATCH_SIZE):
            batch = [str(p) for p in photos[i:i + BATCH_SIZE]]
            rows.extend(et.get_tags(batch, tags=TAGS))
            bar.update(len(batch))
    total_s = time.perf_counter() - start

    show("By folder", Counter(
        p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else "(top level)"
        for p in photos))
    show("By file type", Counter(p.suffix.lower() for p in photos))
    show("By camera", Counter(str(tag(r, "Model") or "unknown") for r in rows))

    pairs = raw_jpeg_pairs(photos)
    no_time = [r.get("SourceFile", "?") for r in rows if not tag(r, "DateTimeOriginal")]
    errors = [(r.get("SourceFile", "?"), error_of(r)) for r in rows if error_of(r)]

    print(f"\nRAW + JPEG pairs (count as one photo): {len(pairs)}")
    print(f"Missing capture time: {len(no_time)}")
    for f in no_time[:10]:
        print(f"    {f}")
    print(f"Unreadable files: {len(errors)}")
    for f, err in errors[:10]:
        print(f"    {f}: {err}")

    rate = len(rows) / max(total_s, 1e-6)
    print(f"\nRead {len(rows)} photos in {total_s:.1f} s ({rate:.0f} photos/s)")


if __name__ == "__main__":
    main()
