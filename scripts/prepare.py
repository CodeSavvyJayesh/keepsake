r"""Prepare one or many weddings in one go, unattended: every analysis step, in order.

    ks prepare "E:\Sharma Wedding" "E:\Mehta Wedding" ...
    ks prepare "D:\All Raw images" "E:\All Raw images" --selected "E:\All Selected images"

First form: a wedding folder with a Raw folder and a Selected folder inside (a past wedding, to
learn from: the studio's choices are matched first), or just the folder of photos to cull.
Second form: folders that hold one folder per wedding of photos, and (--selected) one folder
per wedding of the studio's picks; they are paired by the couple's names ('Ira & Kian
(Dubai)' pairs with 'ira n kian'). A wedding already prepared under another folder name
is skipped, so it is never learned from twice.

For each wedding:  ingest -> match (when the picks are known) -> quality -> faces -> clip -> groups
Every step carries on where it stopped, so after a stop or a disconnected drive the same
command can simply be run again. If one wedding fails, the others still run. A log of
everything is saved in data\reports.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import CACHE_DIR, DATA_DIR, script_command  # noqa: E402
from engine.ingest import duration, shoot_for  # noqa: E402

STEPS = ["ingest", "quality", "faces", "clip", "groups"]
NOT_NAMES = {"n", "and", "weds", "wedding", "raw", "image", "images", "s", "selected", "select", "photos"}


def plan(folder: Path) -> tuple[Path | None, Path]:
    """(wedding folder to match, or None; folder of all photos)."""
    subs = [d for d in folder.iterdir() if d.is_dir()]
    raw = [d for d in subs if "raw" in d.name.lower()]
    sel = [d for d in subs if "select" in d.name.lower()]
    if len(raw) == 1 and len(sel) == 1:
        return folder, raw[0]
    return None, folder


def couple_key(name: str) -> tuple[str, ...]:
    """The first two names in a folder name: 'Om & Zoya { Jaipur }' -> ('om', 'zoya')."""
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in NOT_NAMES]
    return tuple(words[:2])


def pair_weddings(raw_parents: list[Path], selected_parent: Path | None) -> list[tuple[Path, Path | None]]:
    """(photos folder, picks folder or None) per wedding, one entry per couple."""
    picks = {}
    if selected_parent is not None:
        for d in sorted(selected_parent.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                picks.setdefault(couple_key(d.name), d)
    out, seen = [], set()
    for parent in raw_parents:
        for d in sorted(parent.iterdir()):
            key = couple_key(d.name)
            if not d.is_dir() or d.name.startswith(".") or not key or key in seen:
                continue
            seen.add(key)
            out.append((d, picks.get(key)))
    return out


def prepared_elsewhere(photos: Path, cache_dir: Path = CACHE_DIR) -> str | None:
    """The cache name of this couple's wedding if it was already prepared from another folder."""
    key = couple_key(photos.name)
    mine = shoot_for(photos).name
    if len(key) < 2 or not cache_dir.is_dir():
        return None
    for d in sorted(cache_dir.iterdir()):
        words = set(d.name.split("_"))
        if d.name != mine and set(key) <= words and (d / "catalog.db").exists():
            return d.name
    return None


def run(script: str, args: list[Path], log) -> int:
    cmd = script_command(script, *args)
    log.write("\n$ ks " + script + " " + " ".join(f'"{a}"' for a in args) + "\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    for line in proc.stdout:
        print("    " + line.rstrip())
        log.write(line)
    proc.wait()
    log.flush()
    return proc.returncode


def jobs_from_args(argv: list[str]) -> tuple[list[tuple[str, Path, list[tuple[str, list[Path]]]]], list[str]]:
    """(label, folder, [(script, args)...]) per wedding, and notes for weddings skipped."""
    selected = None
    if "--selected" in argv:
        i = argv.index("--selected")
        selected = clean_path(argv[i + 1]) if i + 1 < len(argv) else None
        argv = argv[:i] + argv[i + 2:]
    folders = [clean_path(a) for a in argv if not a.startswith("--")]
    jobs, notes = [], []
    if selected is not None:
        for parent in [selected] + folders:
            if not parent.is_dir():
                notes.append(f"{parent}: folder not found (is the drive connected?)")
                return [], notes
        for photos, picks in pair_weddings(folders, selected):
            done_as = prepared_elsewhere(photos)
            if done_as:
                notes.append(f"{photos.name}: skipped, already prepared as {done_as}")
                continue
            steps = [("ingest", [photos])] + ([("match_selected", [photos, picks])] if picks else []) \
                + [(s, [photos]) for s in STEPS[1:]]
            label = f"{photos}" + (f"  (studio picks: {picks})" if picks else "  (no studio picks found)")
            jobs.append((label, photos, steps))
        return jobs, notes
    for folder in folders:
        if not folder.is_dir():
            notes.append(f"{folder}: folder not found (is the drive connected?)")
            continue
        wedding, photos = plan(folder)
        steps = [("ingest", [photos])] + ([("match_selected", [wedding])] if wedding else []) \
            + [(s, [photos]) for s in STEPS[1:]]
        jobs.append((f"{folder}" + ("  (studio picks: matched first)" if wedding else ""), photos, steps))
    return jobs, notes


def main() -> int:
    if len(sys.argv) < 2:
        print('Give one or more wedding folders, for example:  ks prepare "E:\\Sharma Wedding" "E:\\Mehta Wedding"')
        print('or folders of weddings and their picks:  ks prepare "D:\\All Raw images" --selected "E:\\All Selected images"')
        return 1
    jobs, notes = jobs_from_args(sys.argv[1:])
    for note in notes:
        print(note)
    if not jobs:
        print("Nothing to prepare.")
        return 1
    print(f"Preparing {len(jobs)} wedding(s):")
    for label, _, _ in jobs:
        print(f"  {label}")
    reports = DATA_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / f"prepare_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    results = []
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"KeepSake prepare - {now_text()}\n" + "".join(n + "\n" for n in notes))
        for n, (label, photos, steps) in enumerate(jobs, 1):
            t = time.perf_counter()
            print(f"\n[{n}/{len(jobs)}] {label}")
            status = "done"
            for script, args in steps:
                print(f"  - {'match' if script == 'match_selected' else script}")
                code = run(script, args, log)
                if code != 0:
                    status = f"stopped at {script} (code {code}); run the same command again to carry on"
                    break
            results.append((photos, f"{status}, {duration(time.perf_counter() - t)}"))
        summary = ["", f"KeepSake prepare - {len(jobs)} wedding(s) - {duration(time.perf_counter() - start)}"]
        summary += [f"  {folder}: {status}" for folder, status in results] + [f"  {n}" for n in notes]
        log.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nFull log: {log_path}")
    return 0 if all(s.startswith("done") for _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
