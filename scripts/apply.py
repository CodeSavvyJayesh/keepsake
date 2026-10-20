r"""Phase 4: copy the cull into Selected and Not selected folders on this laptop.

    ks apply "E:\Sharma Wedding\Raw" ["C:\KeepSake Output\Sharma"] [--changes file.csv] [--removed list]
             [--folder "Haldi"]     only the photos of one event folder (a quick trial)

The photo drive is only read: every original is copied with a checksum check (see
engine/deliver.py), keeping its event folders and file name. --changes takes the file saved
from the review page (your moves). --removed list copies only Selected and lists the Not selected
photos in the manifest (saves space). Running it again carries on where it stopped.
Undo with:  ks undo "output folder"
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import DATA_DIR  # noqa: E402
from engine.deliver import MANIFEST, README, SELECTED, OutputError, apply, jobs_from  # noqa: E402
from engine.ingest import duration, shoot_for  # noqa: E402


def option(name: str) -> str | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def load_changes(cat: Catalog, path: Path) -> tuple[int, list[str]]:
    """Apply the review page's saved moves; returns (photos moved, names not found)."""
    ids = dict(cat.db.execute("SELECT path, id FROM photos"))
    moves: dict[str, list[int]] = {"selected": [], "removed": []}
    unknown = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            decision = (row.get("decision") or "").strip().lower()
            pid = ids.get((row.get("path") or "").strip())
            if pid is None or decision not in moves:
                unknown.append(row.get("path") or "?")
            else:
                moves[decision].append(pid)
    return cat.move(moves["selected"], "selected") + cat.move(moves["removed"], "removed"), unknown


def main() -> int:
    values = {option("--changes"), option("--removed"), option("--folder")}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in values]
    if not args:
        print('Give the shoot folder, for example:  ks apply "E:\\Sharma Wedding\\Raw" "C:\\KeepSake Output\\Sharma"')
        return 1
    root = clean_path(args[0])
    shoot = shoot_for(root)
    out = clean_path(args[1]) if len(args) > 1 else DATA_DIR / "output" / shoot.name
    removed = (option("--removed") or "copy").lower()
    if removed not in ("copy", "list"):
        print("--removed must be 'copy' (copy the Not selected photos too) or 'list' (only list them)")
        return 1
    if not shoot.db.exists():
        print(f'This shoot has not been prepared yet. Run first:  ks prepare "{root}"')
        return 1
    if not root.is_dir():
        print(f"Photo folder not found: {root}  (is the drive connected?)")
        return 1
    with Catalog(shoot.db) as cat:
        if not cat.db.execute("SELECT COUNT(*) FROM cull").fetchone()[0]:
            print(f'No cull yet. Run first:  ks cull "{root}"')
            return 1
        if option("--changes"):
            moved, unknown = load_changes(cat, clean_path(option("--changes")))
            print(f"Your changes: {moved:,} photos moved" + (f"; {len(unknown)} names not found" if unknown else ""))
        jobs = jobs_from(cat, removed)
    only = option("--folder")
    if only:
        jobs = [(j, c) for j, c in jobs if j.original.lower().startswith(only.replace("\\", "/").strip("/").lower() + "/")]
        if not jobs:
            print(f"No photos in a folder called '{only}' (give the event folder name as it is on the drive)")
            return 1
    n_sel = sum(1 for j, _ in jobs if j.folder == SELECTED)
    print(f"Copying {n_sel:,} Selected" + (f" and {len(jobs) - n_sel:,} Not selected" if removed == "copy" else
                                          f" (and listing {len(jobs) - n_sel:,} Not selected)") + f" photos to {out}")
    start = time.perf_counter()
    try:
        report = apply(root, out, jobs)
    except OutputError as e:
        print(e)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to carry on.")
        return 1
    lines = [f"KeepSake apply - {shoot.name} - {now_text()}", "",
             f"  Photo folder (only read)   {root}",
             f"  Output folder              {out}",
             f"  Copied now                 {report.copied:,} files, {report.bytes / 1e9:,.2f} GB",
             f"  Already there              {report.skipped:,}" + (f"; moved between folders {report.moved:,}" if report.moved else ""),
             f"  Not selected, listed only  {report.listed:,}" if report.listed else "",
             "  Every copy checked: its SHA-256 checksum equals the original's, and each original's",
             "  size and date were the same before and after it was read. Nothing was written to the",
             "  photo drive.",
             f"  List of every photo        {out / MANIFEST}",
             f"  Time                       {duration(time.perf_counter() - start)}",
             "", f'  Undo (removes only these copies):  ks undo "{out}"']
    text = "\n".join(line for line in lines if line != "")
    print("\n" + text)
    (out / README).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
