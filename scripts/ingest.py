r"""Phase 1: read a whole shoot once and build its catalog, analysis copies and thumbnails.

    ks ingest "E:\Sharma Wedding\Raw"            read every photo
    ks ingest "E:\Sharma Wedding\Raw" --bench    speed test: best worker count

Options:
    --workers N   use N worker processes (default: the speed-test result, else cores - 1)
    --limit N     only the next N photos (a quick try)
    --redo        make every copy again

Stop at any time with Ctrl+C; the same command continues where it stopped. The drive is
only read. Everything is written on this laptop, in data\cache\<shoot>.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import CACHE_DIR, DOCS_DIR, worker_count  # noqa: E402
from engine.imaging import DECODED, EMBEDDED, RAW_PREVIEW, RAW_RENDER  # noqa: E402
from engine.ingest import (IngestResult, Shoot, bench, default_workers, duration,  # noqa: E402
                           ingest, is_inside, reread_missing_copies, save_settings, shoot_files,
                           shoot_for)
from engine.scan import scan_photos  # noqa: E402

CACHE_MB_PER_PHOTO = 0.6     # analysis copy + thumbnail, with a margin
SOURCE_NAMES = {EMBEDDED: "built-in preview", DECODED: "decoded photo",
                RAW_PREVIEW: "RAW preview", RAW_RENDER: "RAW render"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ks ingest",
                                 description="Read a whole shoot once (the drive is only read).")
    ap.add_argument("folder", nargs="?", help="the shoot folder, in quotes")
    ap.add_argument("--bench", action="store_true", help="speed test: find the best worker count")
    ap.add_argument("--workers", type=int, help="number of worker processes")
    ap.add_argument("--limit", type=int, help="only the next N photos")
    ap.add_argument("--redo", action="store_true", help="make every copy again")
    ap.add_argument("--full-decode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--bench-workers", help=argparse.SUPPRESS)     # e.g. 1,2 (tests)
    ap.add_argument("--bench-size", type=int, default=150, help=argparse.SUPPRESS)
    ap.add_argument("--selfcheck", action="store_true",
                    help="check that every image library loads (used by the Mac build)")
    args = ap.parse_args()
    if not args.folder and not args.selfcheck:
        ap.error("give the shoot folder, for example: ks ingest \"E:\\Sharma Wedding\\Raw\"")
    return args


def selfcheck() -> int:
    """Load every image library the engine needs and print its version. 0 = all good."""
    import platform

    from PIL import Image, features

    from engine.config import FROZEN, DATA_DIR
    print(f"KeepSake engine on {platform.system()} {platform.machine()}, Python "
          f"{platform.python_version()}{', packaged app' if FROZEN else ''}")
    print(f"  Data folder        {DATA_DIR}")
    problems = 0

    def line(name: str, value: str, ok: bool = True) -> None:
        nonlocal problems
        problems += not ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<18} {value}")

    line("Pillow", Image.__version__)
    line("libjpeg-turbo", "yes" if features.check_feature("libjpeg_turbo") else "no",
         bool(features.check_feature("libjpeg_turbo")))
    for name, load in (("rawpy / LibRaw", lambda: __import__("rawpy")),
                       ("pillow-heif", lambda: __import__("pillow_heif")),
                       ("NumPy", lambda: __import__("numpy")),
                       ("PyExifTool", lambda: __import__("exiftool"))):
        try:
            module = load()
            version = getattr(module, "__version__", "?")
            if name.startswith("rawpy"):
                version += f" / LibRaw {module.libraw_version}"
            if name == "pillow-heif":
                libheif = module.libheif_version
                version += f" / libheif {libheif() if callable(libheif) else libheif}"
            line(name, version)
        except Exception as e:  # report every missing library, not just the first
            line(name, f"{type(e).__name__}: {e}", ok=False)
    return 1 if problems else 0


def check_folder(root: Path) -> None:
    if not root.is_dir():
        print(f"Folder not found: {root}")
        print("Is the drive plugged in? Put folder names with spaces in quotes.")
        sys.exit(1)
    if is_inside(CACHE_DIR, root) or is_inside(root, CACHE_DIR):
        print("That folder contains KeepSake's own cache. Point ingest at the shoot folder.")
        sys.exit(1)
    subfolders = [d.name for d in root.iterdir() if d.is_dir()]
    selected = [n for n in subfolders if "select" in n.lower()]
    if selected:
        print(f"This folder also holds the studio's selected copies ('{selected[0]}').")
        print("Those are duplicates of the originals, so ingest only the originals folder:")
        raw = [n for n in subfolders if "raw" in n.lower()]
        print(f'  ks ingest "{root / raw[0]}"' if len(raw) == 1 else "  ks ingest \"<originals>\"")
        sys.exit(1)


def folder_mb(folder: Path) -> float:
    total = 0
    for dirpath, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total / 2**20


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def run_bench(shoot: Shoot, found, options: list[int] | None = None, size: int = 150) -> None:
    cores = worker_count()
    logical = os.cpu_count() or cores
    options = options or sorted({1, 2, 4, cores, min(cores * 2, max(cores, logical - 1))})
    print(f"\nSpeed test: worker counts {', '.join(map(str, options))}, then the old way.")
    print("Each run reads its own photos, so the memory cache cannot flatter the result.\n")
    report = bench(shoot, found, options, per_run=size)
    best = next(r for r in report.rows if r.workers == report.best)
    total = len(found)
    lines = [f"Best: {plural(report.best, 'worker')}, {best.rate:.1f} photos/s "
             f"({best.mb_per_photo:.2f} MB read per photo)."]
    if report.full and report.full.rate:
        lines.append(f"Built-in previews are {best.rate / report.full.rate:.1f}x faster than "
                     f"decoding every photo ({report.full.rate:.1f} photos/s, "
                     f"{report.full.mb_per_photo:.1f} MB per photo).")
    if best.rate:
        lines.append(f"This shoot ({total:,} files) should take about "
                     f"{duration(total / best.rate)}.")
    print("\n" + "\n".join(lines))
    save_settings(shoot, {"workers": report.best, "photos_per_s": round(best.rate, 1),
                          "measured": now_text()})
    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "phase1_speed.txt"
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"Speed test {now_text()}  ({total:,} files, {report.rows[0].photos} per run)\n")
        for r in report.rows + ([report.full] if report.full else []):
            f.write(f"  {r.label:<34}{r.rate:8.1f} photos/s {r.mb_s:8.1f} MB/s "
                    f"{r.mb_per_photo:7.2f} MB/photo  ({r.photos} photos, {r.failed} failed)\n")
        f.write("  " + "\n  ".join(lines) + "\n\n")
    print(f"\nks ingest will now use {plural(report.best, 'worker')} for this shoot.")
    print(f"Saved: {out}")


def report_text(shoot: Shoot, cat: Catalog, run: IngestResult | None, workers: int) -> str:
    counts = cat.counts()
    total = cat.total()
    lines = [f"KeepSake shoot report - {shoot.name} - {now_text()}",
             f"Shoot folder: {shoot.root}", ""]
    parts = ", ".join(f"{k} {counts[k]:,}" for k in ("done", "failed", "empty", "missing", "new")
                      if counts.get(k))
    lines.append(f"  Photos in catalog   {total:,}   ({parts})")
    if run and run.processed:
        lines.append(f"  This run            {run.processed:,} photos in {duration(run.seconds)}"
                     f" = {run.rate:.1f} photos/s with {plural(workers, 'worker')}")
    read, held = cat.read_totals()
    if held:
        lines.append(f"  Read from drive     {gb(read)} of {gb(held)} in those files "
                     f"({100 * read / held:.1f}%)")
    sources = cat.preview_sources()
    if sources:
        lines.append("  Analysis copy from  " + " | ".join(
            f"{SOURCE_NAMES.get(k, k)} {v:,}" for k, v in sources.most_common()))
    lines.append(f"  No capture time     {cat.no_time():,}")
    lines.append(f"  Cache on laptop     {gb(folder_mb(shoot.cache))}  ({shoot.cache})")
    lines += ["", f"  {'Camera body':<40}{'photos':>7}   {'first capture':<20}  {'last capture':<20}"]
    for b in cat.bodies():
        flag = "  CHECK CLOCK" if b.clock_off else ""
        first, last = (b.first or "-")[:19], (b.last or "-")[:19]
        lines.append(f"  {b.name[:39]:<40}{b.photos:>7,}   {first:<20}  {last:<20}{flag}")
    if any(b.clock_off for b in cat.bodies()):
        lines.append("  CHECK CLOCK: this camera's dates do not overlap the rest of the shoot; its clock")
        lines.append("  was probably not set. Phase 3 lines the cameras up; nothing to do now.")
    problems = cat.failures()
    if problems:
        out = shoot.cache / "problems.txt"
        out.write_text("".join(f"{p}\t{e}\n" for p, e in problems), encoding="utf-8")
        lines += ["", f"  Files that could not be read ({len(problems)}), listed in {out}:"]
        lines += [f"    {p}   {e[:90]}" for p, e in problems[:5]]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        return selfcheck()
    root = clean_path(args.folder)
    check_folder(root)
    shoot = shoot_for(root)
    print(f"Shoot folder: {root}   (only read, never changed)")
    print(f"Saving to:    {shoot.cache}")

    t = time.perf_counter()
    print("Finding photos...", end=" ", flush=True)
    found = scan_photos(root)
    print(f"{len(found):,} files in {time.perf_counter() - t:.1f} s")
    if not found:
        print("No photos in this folder.")
        return 1
    if args.bench:
        options = [int(n) for n in args.bench_workers.split(",")] if args.bench_workers else None
        run_bench(shoot, found, options, args.bench_size)
        return 0

    workers, why = (args.workers, "you chose") if args.workers else default_workers(shoot)
    workers = max(1, workers)
    run = None
    with Catalog(shoot.db) as cat:
        cat.set("root", root)
        sync = cat.sync(shoot_files(root, found))
        print(f"Catalog: {sync.new:,} new, {sync.changed:,} changed, {sync.unchanged:,} already "
              f"read, {sync.missing:,} no longer on the drive"
              + (f", {sync.back:,} back again" if sync.back else ""))
        lost = reread_missing_copies(shoot, cat)
        if lost:
            print(f"{lost:,} photos were read on another computer (their copies are not on this one): "
                  f"reading them again. What was measured stays the same.")
        if args.redo:
            cat.redo_all()
        todo = len(cat.pending(args.limit))
        if todo:
            need = todo * CACHE_MB_PER_PHOTO
            free = shutil.disk_usage(CACHE_DIR.parent).free / 2**20
            if free < need + 2048:
                print(f"Not enough free space on this laptop: need about {gb(need + 2048)}, "
                      f"{gb(free)} free.")
                return 1
            print(f"\nReading {todo:,} photos with {plural(workers, 'worker')} ({why}).")
            print("Stop any time with Ctrl+C; run the same command again to continue.\n")
            run = ingest(shoot, cat, workers, args.limit, prefer_embedded=not args.full_decode)
        else:
            print("Nothing new to read: every photo is already in the catalog.")
        text = report_text(shoot, cat, run, workers)
    print("\n" + text)
    (shoot.cache / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved: {shoot.cache / 'report.txt'}")
    if run and run.cache_error:
        print(f"\nStopped: {run.cache_error}")
        print("Free some space on this laptop, then run the same command again to continue.")
        return 2
    if run and run.drive_gone:
        print("\nThe drive was disconnected. Plug it back in and run the same command to continue.")
        return 2
    if run and run.stopped:
        print("\nStopped. Run the same command again to continue where it stopped.")
        return 2
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()     # needed by the packaged app later (PyInstaller)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(2)
