r"""Match the studio's selected photos back to their originals, and save the labels.

Run:  ks match "D:\Dinesh & Ruchi (Bali)"
  or: ks match "D:\All Raw images\Ansh & Anjali" "E:\All Selected images\Ansh & Anjali"

One folder: it must contain two subfolders, one with "raw" in its name (every photo shot) and
one with "select" in its name (the studio's picks). Two folders: the photos, then the picks.
The drives are only read. The result is saved on this laptop: data\labels\<shoot>.csv, one row
per original photo, with selected = 1 or 0.
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


def labels_file(raw_dir: Path) -> Path:
    """Named after the shoot's cache folder, so two weddings never share a labels file."""
    from engine.ingest import shoot_for
    return LABELS_DIR / f"{shoot_for(raw_dir).name}.csv"


def prefix(name: str) -> str:
    """'DSC_0042.JPG' -> 'DSC': the camera's file prefix, usually one per photographer."""
    m = re.match(r"[A-Za-z]+", name)
    return m.group(0).upper() if m else "(none)"


def main() -> int:
    args = [a for a in sys.argv[1:] if a.strip()]
    if not args:
        print('Give the wedding folder, for example:  ks match "D:\\Dinesh & Ruchi (Bali)"')
        print('or the photos and the picks:  ks match "D:\\All Raw images\\Ansh" "E:\\All Selected images\\Ansh"')
        return 1
    from engine.cli import clean_path
    if len(args) >= 2:
        raw_dir, sel_dir = clean_path(args[0]), clean_path(args[1])
        root = raw_dir.parent
    else:
        root = clean_path(args[0])
        raw_dir = sel_dir = None
    for folder in (root, raw_dir, sel_dir):
        if folder is not None and not folder.is_dir():
            print(f"Folder not found: {folder}")
            return 1

    start = time.perf_counter()
    if raw_dir is None:
        raw_dir = pick_subfolder(root, "raw")
        sel_dir = pick_subfolder(root, "select")
    originals = sorted(find_photos(raw_dir))
    selected = sorted(find_photos(sel_dir))
    print(f"Originals: {len(originals):>6}  in '{raw_dir}'")
    print(f"Selected:  {len(selected):>6}  in '{sel_dir}'")
    if not originals or not selected:
        print("Nothing to match.")
        return 1

    result = match_selected(originals, selected, meta_fn=camera_and_time)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = labels_file(raw_dir)
    per_event: Counter = Counter()
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "selected", "selected_copy", "raw_event", "selected_event"])
        for p in originals:
            copy = result.matches.get(p)
            event = event_of(copy, sel_dir) if copy else ""
            if copy:
                per_event[event or "(no event folder)"] += 1
            writer.writerow([f"{raw_dir.name}/{p.relative_to(raw_dir).as_posix()}", 1 if copy else 0,
                             str(copy) if copy else "", event_of(p, raw_dir), event])

    matched = len(result.matches)
    print(f"\nMatched {matched} of {len(selected)} selected photos")
    for how, n in result.how.most_common():
        print(f"  by {how:<28}{n:>6}")
    if result.repeats:
        print(f"Extra copies of the same original: {len(result.repeats)}")
    if result.unmatched:
        miss = out.with_name(out.stem + "_unmatched.txt")
        miss.write_text("\n".join(str(p) for p in result.unmatched) + "\n", encoding="utf-8")
        print(f"Not matched: {len(result.unmatched)}  (listed in {miss})")
        for p in result.unmatched[:5]:
            print(f"    {p.name}")
    else:
        print("Not matched: 0")

    print(f"\nSelection rate: {matched / len(originals):.1%}  "
          f"({matched} selected, {len(originals) - matched} not selected)")
    shot, picked = Counter(prefix(p.name) for p in originals), Counter(prefix(p.name) for p in result.matches)
    print("\nPicks per camera (file name prefix):")
    for name, n in shot.most_common():
        print(f"  {name:<8}{n:>7} photos  {picked[name]:>6} picked ({picked[name] / n:.0%})")
    silent = [name for name, n in shot.items() if n >= 200 and picked[name] == 0]
    if silent:
        print(f"  Check: no picks at all from {', '.join(silent)}. If the studio did use those photos,"
              " the Selected folder is incomplete and this wedding would teach the wrong taste.")
    if per_event:
        print("\nSelected photos per event:")
        for event, n in sorted(per_event.items()):
            print(f"  {n:>6}  {event}")
    print(f"\nSaved: {out}")
    print(f"Done in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
