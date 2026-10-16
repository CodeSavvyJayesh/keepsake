r"""Match the studio's selected photos back to their originals, and save the labels.

Run:  ks match "D:\Dinesh & Ruchi (Bali)"

The wedding folder must contain two subfolders: one with "raw" in its name (every photo
shot) and one with "select" in its name (the studio's picks). The drive is only read.
The result is saved on this laptop: data\labels\<wedding>.csv, one row per original photo,
with selected = 1 or 0.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.cli import path_arg  # noqa: E402
from engine.config import DATA_DIR  # noqa: E402
from engine.labels import match_selected  # noqa: E402
from engine.meta import BATCH_SIZE, open_exiftool, tag  # noqa: E402
from engine.scan import find_photos  # noqa: E402

LABELS_DIR = DATA_DIR / "labels"


def pick_subfolder(root: Path, word: str) -> Path:
    folders = [d for d in root.iterdir() if d.is_dir()]
    hits = [d for d in folders if word in d.name.lower()]
    if len(hits) != 1:
        names = ", ".join(d.name for d in folders) or "none"
        print(f"Expected exactly one folder with '{word}' in its name inside {root}.")
        print(f"Folders found: {names}")
        sys.exit(1)
    return hits[0]


def camera_and_time(paths: list[Path]) -> dict[Path, tuple]:
    """{file: (camera, capture time, fraction of a second)}, read with one ExifTool process."""
    keyed = {os.path.normcase(os.path.abspath(p)): p for p in paths}
    found: dict[Path, tuple] = {}
    print(f"Checking camera and capture time for {len(paths)} files with shared names...")
    with open_exiftool() as et:
        for i in range(0, len(paths), BATCH_SIZE):
            batch = [str(p) for p in paths[i:i + BATCH_SIZE]]
            for row in et.get_tags(batch, tags=["Model", "DateTimeOriginal", "SubSecTimeOriginal"]):
                p = keyed.get(os.path.normcase(os.path.abspath(row.get("SourceFile", ""))))
                taken = tag(row, "DateTimeOriginal")
                if p is not None and taken:
                    found[p] = (str(tag(row, "Model")), str(taken),
                                str(tag(row, "SubSecTimeOriginal") or ""))
    return found


def event_of(path: Path, base: Path) -> str:
    """The first folder under base, e.g. '27.08.2026 Welcome & Bhajan Dinner'."""
    parts = path.relative_to(base).parts
    return parts[0] if len(parts) > 1 else ""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "wedding"


def main() -> None:
    if len(sys.argv) < 2:
        print('Give the wedding folder, for example:  ks match "D:\\Dinesh & Ruchi (Bali)"')
        sys.exit(1)
    root = path_arg(Path("."))
    if not root.is_dir():
        print(f"Folder not found: {root}")
        sys.exit(1)

    start = time.perf_counter()
    raw_dir = pick_subfolder(root, "raw")
    sel_dir = pick_subfolder(root, "select")
    originals = sorted(find_photos(raw_dir))
    selected = sorted(find_photos(sel_dir))
    print(f"Originals: {len(originals):>6}  in '{raw_dir.name}'")
    print(f"Selected:  {len(selected):>6}  in '{sel_dir.name}'")

    result = match_selected(originals, selected, meta_fn=camera_and_time)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELS_DIR / f"{slug(root.name)}.csv"
    per_event: Counter = Counter()
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "selected", "selected_copy", "raw_event", "selected_event"])
        for p in originals:
            copy = result.matches.get(p)
            event = event_of(copy, sel_dir) if copy else ""
            if copy:
                per_event[event or "(no event folder)"] += 1
            writer.writerow([str(p.relative_to(root)), 1 if copy else 0,
                             str(copy.relative_to(root)) if copy else "",
                             event_of(p, raw_dir), event])

    matched = len(result.matches)
    print(f"\nMatched {matched} of {len(selected)} selected photos")
    for how, n in result.how.most_common():
        print(f"  by {how:<28}{n:>6}")
    if result.repeats:
        print(f"Extra copies of the same original: {len(result.repeats)}")
    if result.unmatched:
        miss = LABELS_DIR / f"{slug(root.name)}_unmatched.txt"
        miss.write_text("\n".join(str(p.relative_to(root)) for p in result.unmatched) + "\n",
                        encoding="utf-8")
        print(f"Not matched: {len(result.unmatched)}  (listed in {miss})")
        for p in result.unmatched[:5]:
            print(f"    {p.relative_to(root)}")
    else:
        print("Not matched: 0")

    if originals:
        print(f"\nSelection rate: {matched / len(originals):.1%}  "
              f"({matched} selected, {len(originals) - matched} not selected)")
    print("\nSelected photos per event:")
    for event, n in sorted(per_event.items()):
        print(f"  {n:>6}  {event}")
    print(f"\nSaved: {out}")
    print(f"Done in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
