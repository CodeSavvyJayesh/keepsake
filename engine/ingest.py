"""Phase 1: ingest a shoot.

Reads every photo once, in parallel, straight from the studio's drive (read only), and makes:
  * an upright analysis copy (long side 1620 px) that every later phase works from,
  * a 400 px thumbnail for the photo grid,
  * a catalog row with camera, capture time and exposure.
Everything is written on this laptop, in data/cache/<shoot>/, never on the shoot drive.

The work is split across worker processes (one per physical core minus one by default).
Photos are handed out in folder order with only a few in flight per worker, which keeps
reads on a USB hard disk close together on the platter.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import random
import re
import shutil
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from PIL import UnidentifiedImageError

from .catalog import Catalog, ShootFile, now_text
from .config import ANALYSIS_QUALITY, CACHE_DIR, THUMB_PX, THUMB_QUALITY, worker_count
from .imaging import fit, load_photo
from .meta import FIELDS, body_of, exiftool_fields
from .scan import FoundFile, group_photos, kind_of

Task = tuple  # (photo id, main file, companion file or None)


# --- Where a shoot's files go ----------------------------------------------------------

def slug(text: str) -> str:
    text = text.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


@dataclass
class Shoot:
    root: Path               # the shoot folder on the drive (only ever read)
    name: str                # folder name used for the cache, e.g. 'neel_tara_bali_raw_images'
    cache: Path              # data/cache/<name> on this laptop

    @property
    def db(self) -> Path:
        return self.cache / "catalog.db"

    @property
    def settings_file(self) -> Path:
        return self.cache / "settings.json"

    def analysis_path(self, photo_id: int) -> Path:
        return cache_path(self.cache, "analysis", photo_id)

    def thumb_path(self, photo_id: int) -> Path:
        return cache_path(self.cache, "thumbs", photo_id)


def shoot_for(root: Path, cache_dir: Path = CACHE_DIR) -> Shoot:
    """The cache folder is named after the last two folder names (not the drive letter), so
    the same shoot is recognised when the drive comes back as E: instead of D:."""
    parts = [p for p in Path(os.path.abspath(root)).parts[1:] if slug(p)]
    name = "_".join(slug(p) for p in parts[-2:]) or "drive_" + slug(str(root)[:1] or "x")
    return Shoot(Path(os.path.abspath(root)), name[:80], cache_dir / name[:80])


def cache_path(cache: Path, kind: str, photo_id: int) -> Path:
    """data/cache/<shoot>/<kind>/012/012345.jpg - 1,000 files per folder keeps Explorer fast."""
    return cache / kind / f"{photo_id // 1000:03d}" / f"{photo_id:06d}.jpg"


def copies_here(cache: Path, kind: str) -> set[int]:
    """Ids of the photos whose copy of this kind ('analysis' or 'thumbs') is on this computer."""
    out: set[int] = set()
    try:
        subs = [d.path for d in os.scandir(cache / kind) if d.is_dir()]
    except OSError:
        return out
    for sub in subs:
        try:
            for f in os.scandir(sub):
                stem, _, ext = f.name.partition(".")
                if ext.lower() == "jpg" and stem.isdigit():
                    out.add(int(stem))
        except OSError:
            continue
    return out


def reread_missing_copies(shoot: Shoot, catalog) -> int:
    """Photos read before whose copies are not on this computer (a catalog brought over from
    another laptop comes without them) are read again, so they can be shown and measured.
    What was measured stays as it is: only the copies are made again, the choices do not change."""
    done = [r[0] for r in catalog.db.execute("SELECT id FROM photos WHERE status = 'done'")]
    if not done:
        return 0
    have = copies_here(shoot.cache, "analysis") & copies_here(shoot.cache, "thumbs")
    lost = [(pid,) for pid in done if pid not in have]
    if lost:
        with catalog.db:
            catalog.db.executemany("UPDATE photos SET status = 'new' WHERE id = ? AND status = 'done'", lost)
    return len(lost)


def is_inside(path: Path, folder: Path) -> bool:
    a = os.path.normcase(os.path.abspath(path))
    b = os.path.normcase(os.path.abspath(folder))
    return a == b or a.startswith(b.rstrip("\\/") + os.sep)


def shoot_files(root: Path, found: list[FoundFile]) -> list[ShootFile]:
    """Turn the files found on the drive into catalog entries (RAW+JPEG pairs become one)."""
    by_path = {f.path: f for f in found}
    files = []
    for main, companion in group_photos(by_path):
        a = by_path[main]
        b = by_path.get(companion) if companion else None
        folder = main.parent.relative_to(root).as_posix()
        files.append(ShootFile(
            path=main.relative_to(root).as_posix(),
            companion=companion.relative_to(root).as_posix() if companion else None,
            folder="" if folder == "." else folder,
            kind=kind_of(main),
            bytes=a.size + (b.size if b else 0),
            mtime_ms=max(a.mtime_ms, b.mtime_ms if b else 0),
        ))
    return files


# --- Worker side (runs in each worker process) -----------------------------------------------

_cfg: dict = {}


def init_worker(cache: str, prefer_embedded: bool = True, ignore_ctrl_c: bool = True) -> None:
    """Runs once in each worker process."""
    if ignore_ctrl_c:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C is handled by the main process
    _cfg.update(cache=Path(cache), prefer_embedded=prefer_embedded, made_dirs=set())


def _ping(_=None) -> int:
    return os.getpid()


def _save_jpeg(img, path: Path, quality: int) -> None:
    folder = path.parent
    if folder not in _cfg["made_dirs"]:
        folder.mkdir(parents=True, exist_ok=True)
        _cfg["made_dirs"].add(folder)
    img.save(path, "JPEG", quality=quality, icc_profile=img.info.get("icc_profile"))


def process_one(task: Task) -> dict:
    """Read one photo and write its analysis copy and thumbnail. Never raises."""
    photo_id, main, companion = task
    start = time.perf_counter()
    result: dict = {"id": photo_id}
    # For a RAW+JPEG pair, the JPEG is read first: its built-in preview is the cheapest.
    sources = [Path(companion), Path(main)] if companion else [Path(main)]
    loaded, used, errors = None, None, []
    for src in sources:
        try:
            if os.path.getsize(src) == 0:
                errors.append(f"{src.name} is empty (0 bytes)")
                continue
            loaded, used = load_photo(src, prefer_embedded=_cfg["prefer_embedded"]), src
            break
        except Exception as e:  # any damaged file: record it and carry on with the next
            errors.append(describe(e))
    if loaded is None:
        empty = all("empty (0 bytes)" in e for e in errors)
        result.update(status="empty" if empty else "failed", error="; ".join(errors)[:500],
                      ms=round((time.perf_counter() - start) * 1000, 1))
        return result

    meta = {name: loaded.meta.get(name) for name in FIELDS}
    if not meta["taken"] or not meta["camera"]:
        try:                       # CR3, RAF, HEIF and files without EXIF: ask ExifTool
            extra = exiftool_fields(used)
        except Exception:          # no ExifTool, or it could not read the file either
            extra = {}
        for key, value in extra.items():
            # orientation is not taken: the copy was already turned upright from the file itself
            if key != "orientation" and meta.get(key) is None:
                meta[key] = value
    try:
        _save_jpeg(loaded.image, cache_path(_cfg["cache"], "analysis", photo_id), ANALYSIS_QUALITY)
        _save_jpeg(fit(loaded.image, THUMB_PX), cache_path(_cfg["cache"], "thumbs", photo_id),
                   THUMB_QUALITY)
    except OSError as e:           # this laptop's disk is full or not writable: a real stop
        result.update(status="failed", error=f"could not write cache: {e}"[:500])
        return result

    result.update(meta)
    result.update(
        status="done", error=None,
        body=body_of(meta["camera"], meta["serial"]),
        width=loaded.width, height=loaded.height,
        preview=loaded.source, preview_w=loaded.image.width, preview_h=loaded.image.height,
        read_mb=None if loaded.bytes_read is None else round(loaded.bytes_read / 2**20, 3),
        ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return result


def describe(error: Exception) -> str:
    """The reason a photo could not be read, in plain words where we can."""
    message = str(error)
    if isinstance(error, UnidentifiedImageError):
        return "not a readable photo (damaged, or another kind of file)"
    if "truncated" in message or "premature end" in message.lower():
        return "file is cut short (damaged, or only partly copied)"
    if isinstance(error, FileNotFoundError):
        return "file disappeared while reading"
    return f"{type(error).__name__}: {message}"


# --- Main side ------------------------------------------------------------------------------------

class Runner:
    """Runs process_one over many photos in worker processes.

    `workers=0` runs everything in this process (used by the tests). A worker process that
    crashes (a damaged RAW can crash LibRaw) is replaced; the photos it had are retried one at a
    time, and the one that crashes again is marked failed, so one bad file cannot stop a shoot.
    """

    def __init__(self, workers: int, cache: Path, prefer_embedded: bool = True,
                 fn: Callable[[Task], dict] | None = None):
        self.workers = workers
        self.fn = fn or process_one
        self.args = (str(cache), prefer_embedded)
        self.pool: ProcessPoolExecutor | None = None
        self.window = max(2, workers * 3)

    def __enter__(self) -> "Runner":
        if self.workers == 0:
            init_worker(*self.args, ignore_ctrl_c=False)
        else:
            self._start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _start(self) -> None:
        self.pool = ProcessPoolExecutor(
            max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker, initargs=self.args)
        list(self.pool.map(_ping, range(self.workers * 2)))   # start them all before timing

    def close(self, cancel: bool = False) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=cancel)
            self.pool = None

    def run(self, tasks: Iterable[Task], on_result: Callable[[dict], None]) -> None:
        if self.workers == 0:
            for task in tasks:
                on_result(self.fn(task))
            return
        todo = iter(tasks)
        inflight: dict = {}
        while True:
            try:
                while len(inflight) < self.window:
                    task = next(todo, None)
                    if task is None:
                        break
                    inflight[self.pool.submit(self.fn, task)] = task
                if not inflight:
                    return
                # wake up twice a second so Ctrl+C works on Windows too
                done, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                for fut in done:
                    result = fut.result()
                    del inflight[fut]
                    on_result(result)
            except BrokenProcessPool:
                suspects = list(inflight.values())
                inflight.clear()
                self._recover(suspects, on_result)

    def _recover(self, suspects: list[Task], on_result) -> None:
        self.close()
        self._start()
        for task in suspects:
            try:
                on_result(self.pool.submit(self.fn, task).result())
            except BrokenProcessPool:
                on_result({"id": task[0], "status": "failed",
                           "error": "crashed the image reader (file is probably damaged)"})
                self.close()
                self._start()


class Progress:
    """One updating line: count, percent, photos/s, MB/s from the drive and time left."""

    def __init__(self, total: int, label: str = "Reading", every: float = 0.5, out=print):
        self.total, self.label, self.every, self.out = total, label, every, out
        self.done = 0
        self.mb = 0.0
        self.start = time.perf_counter()
        self.shown = 0.0

    def add(self, mb: float | None) -> None:
        self.done += 1
        self.mb += mb or 0.0
        now = time.perf_counter()
        if now - self.shown >= self.every and self.done < self.total:   # finish() shows the last
            self.shown = now
            self.out(self.line(now), end="\r", flush=True)

    def line(self, now: float | None = None) -> str:
        secs = max(1e-6, (now or time.perf_counter()) - self.start)
        rate = self.done / secs
        left = (self.total - self.done) / rate if rate > 0 else 0
        pct = 100 * self.done / self.total if self.total else 100
        return (f"  {self.label} {self.done:>7,} / {self.total:,}  {pct:5.1f}%  "
                f"{rate:6.1f} photos/s  {self.mb / secs:6.1f} MB/s  "
                f"about {duration(left)} left   ")

    def finish(self) -> None:
        self.out(self.line())


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


class DriveGone(Exception):
    """The shoot folder disappeared while reading: the drive was unplugged."""


class CacheWriteError(Exception):
    """This laptop's disk is full or not writable: every further photo would fail too."""


@dataclass
class IngestResult:
    processed: int = 0
    done: int = 0
    failed: int = 0
    seconds: float = 0.0
    mb_read: float = 0.0
    stopped: bool = False          # Ctrl+C
    drive_gone: bool = False
    cache_error: str | None = None

    @property
    def rate(self) -> float:
        return self.processed / self.seconds if self.seconds > 0 else 0.0


def ingest(shoot: Shoot, catalog: Catalog, workers: int, limit: int | None = None,
           prefer_embedded: bool = True, progress: bool = True,
           out=print) -> IngestResult:
    """Process every photo still to do in the catalog. Safe to stop at any time."""
    tasks_rows = catalog.pending(limit)
    result = IngestResult()
    if not tasks_rows:
        return result
    root = shoot.root
    tasks = [(pid, str(root / path), str(root / comp) if comp else None)
             for pid, path, comp, _kind, _bytes in tasks_rows]
    bar = Progress(len(tasks), out=out) if progress else None
    batch: list[dict] = []
    last_save = time.perf_counter()

    def flush() -> None:
        nonlocal last_save
        catalog.save(batch)
        batch.clear()
        last_save = time.perf_counter()

    def on_result(r: dict) -> None:
        if r["status"] == "failed":     # don't record failures that are not the photo's fault
            if not os.path.isdir(root):
                raise DriveGone()
            if (r.get("error") or "").startswith("could not write cache"):
                raise CacheWriteError(r["error"])
        batch.append(r)
        result.processed += 1
        result.done += r["status"] == "done"
        result.failed += r["status"] in ("failed", "empty")
        result.mb_read += r.get("read_mb") or 0.0
        if bar:
            bar.add(r.get("read_mb"))
        if len(batch) >= 200 or time.perf_counter() - last_save > 2.0:
            flush()

    runner = Runner(workers, shoot.cache, prefer_embedded)
    start = time.perf_counter()
    try:
        with runner:
            start = time.perf_counter()        # do not count worker start-up
            try:
                runner.run(tasks, on_result)
            except (KeyboardInterrupt, DriveGone, CacheWriteError) as e:
                result.stopped = isinstance(e, KeyboardInterrupt)
                result.drive_gone = isinstance(e, DriveGone)
                result.cache_error = str(e) if isinstance(e, CacheWriteError) else None
                runner.close(cancel=True)
    finally:
        result.seconds = time.perf_counter() - start
        flush()
        if bar:
            bar.finish()
    catalog.set("last_ingest", now_text())
    catalog.set("last_workers", workers)
    if result.processed:
        catalog.set("last_rate", f"{result.rate:.1f}")
    return result


# --- Speed test -------------------------------------------------------------------------------------

@dataclass
class BenchRow:
    label: str
    workers: int
    photos: int
    seconds: float
    mb: float
    failed: int

    @property
    def rate(self) -> float:
        return self.photos / self.seconds if self.seconds else 0.0

    @property
    def mb_s(self) -> float:
        return self.mb / self.seconds if self.seconds else 0.0

    @property
    def mb_per_photo(self) -> float:
        return self.mb / self.photos if self.photos else 0.0


@dataclass
class BenchReport:
    rows: list[BenchRow] = field(default_factory=list)
    best: int = 1
    full: BenchRow | None = None


def bench(shoot: Shoot, found: list[FoundFile], options: list[int], per_run: int = 120,
          seed: int | None = None, out=print) -> BenchReport:
    """Time ingest with different worker counts, each on fresh photos (no disk cache help).

    Then times the old way (decoding every full photo) with the best count, to show what the
    built-in previews save. Writes only to a temporary folder on this laptop.
    """
    files = [(str(main), str(comp) if comp else None)
             for main, comp in group_photos([f.path for f in found])]
    runs = len(options) + 1
    per_run = max(1, min(per_run, len(files) // runs))
    # Each run gets its own block of consecutive photos (like a real ingest, which reads in
    # folder order) from its own part of the shoot, so no run re-reads another run's files.
    rng = random.Random(seed if seed is not None else time.time_ns())
    part = len(files) // runs
    blocks = []
    for n in range(runs):
        start = n * part + rng.randint(0, max(0, part - per_run))
        blocks.append(files[start:start + per_run])
    rng.shuffle(blocks)
    tmp = shoot.cache.parent / "_speed_test"     # on this laptop; deleted afterwards
    report = BenchReport()

    def run(label: str, workers: int, chunk: list, prefer_embedded: bool) -> BenchRow:
        tasks = [(i + 1, main, comp) for i, (main, comp) in enumerate(chunk)]
        mb, failed = [0.0], [0]

        def on_result(r: dict) -> None:
            mb[0] += r.get("read_mb") or 0.0
            failed[0] += r["status"] != "done"

        with Runner(workers, tmp, prefer_embedded) as runner:
            t0 = time.perf_counter()
            runner.run(tasks, on_result)
            secs = time.perf_counter() - t0
        row = BenchRow(label, workers, len(tasks), secs, mb[0], failed[0])
        out(f"  {label:<34}{row.rate:8.1f} photos/s {row.mb_s:8.1f} MB/s "
            f"{row.mb_per_photo:7.2f} MB/photo")
        return row

    try:
        for workers, chunk in zip(options, blocks):
            report.rows.append(run(f"{workers} worker{'s' if workers > 1 else ''}", workers,
                                   chunk, True))
        top = max(r.rate for r in report.rows)
        # fewest workers within 5% of the fastest: same speed, lighter on the laptop
        report.best = min(r.workers for r in report.rows if r.rate >= 0.95 * top)
        report.full = run(f"old way: full decode, {report.best} workers", report.best,
                          blocks[-1], False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return report


def load_settings(shoot: Shoot) -> dict:
    try:
        return json.loads(shoot.settings_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(shoot: Shoot, settings: dict) -> None:
    shoot.cache.mkdir(parents=True, exist_ok=True)
    shoot.settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def default_workers(shoot: Shoot) -> tuple[int, str]:
    """(workers, why): the speed-test result for this shoot if there is one."""
    best = load_settings(shoot).get("workers")
    if isinstance(best, int) and best > 0:
        return best, "measured best for this drive (ks ingest --bench)"
    return worker_count(), "physical cores minus one"
