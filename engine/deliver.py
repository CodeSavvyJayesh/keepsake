"""Phase 4: delivering the cull - Selected and Removed folders on this laptop, and undo.

The photo drive is only ever read. Every original is copied into the output folder
(Selected/... or Removed/..., keeping its event folders and file name) like this:
  1. note the original's size and date,
  2. read it once, writing the copy and computing its SHA-256 checksum at the same time,
  3. read the copy back and compute its checksum again: both must be equal,
  4. check the original's size and date again: they must not have changed.
Only then is the copy given its real name. Every copy is listed in the output folder's
'KeepSake manifest.csv' with its checksum, so a stopped run carries on where it left off and
'ks undo' can remove exactly what was copied (and nothing a person has changed since).
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Catalog
from .config import DATA_DIR

MANIFEST = "KeepSake manifest.csv"
README = "KeepSake read me.txt"
PART = ".keepsake-part"
CHUNK = 8 * 1024 * 1024
SPARE_BYTES = 1024 ** 3          # leave 1 GB free on the laptop
FIELDS = ["folder", "unsure", "reason", "original", "copy", "bytes", "sha256"]


class OutputError(Exception):
    """The output folder cannot be used; the message says why, in plain words."""


class SourceChanged(Exception):
    pass


@dataclass
class Job:
    photo_id: int
    folder: str               # 'Selected' / 'Removed'
    unsure: int
    reason: str
    original: str             # relative to the shoot folder
    copy: bool                # False: listed only (--removed list)


@dataclass
class ApplyReport:
    copied: int = 0
    skipped: int = 0          # already there from an earlier run
    moved: int = 0            # copied earlier, moved since between Selected and Removed
    listed: int = 0
    bytes: int = 0
    seconds: float = 0.0
    folders: dict = field(default_factory=dict)


def _device(path: Path) -> int | None:
    p = Path(os.path.abspath(path))
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return os.stat(p).st_dev
    except OSError:
        return None


def check_output(root: Path, out: Path) -> None:
    """Refuse output folders that would write to the photo drive or mix with other files."""
    root, out = Path(os.path.abspath(root)), Path(os.path.abspath(out))
    if out == root or root in out.parents or out in root.parents:
        raise OutputError("The output folder must not be inside the photo folder (the client's "
                          "drive is never written to). Choose a folder on this laptop, for example "
                          "C:\\KeepSake Output\\<wedding name>.")
    drive = _device(root)
    if drive is not None and drive == _device(out) and drive != _device(DATA_DIR):
        raise OutputError("The output folder is on the same drive as the photos. KeepSake never "
                          "writes to the client's drive: choose a folder on this laptop.")
    if out.exists() and any(out.iterdir()) and not (out / MANIFEST).exists():
        raise OutputError(f"{out} already has other files in it. Choose a new, empty folder.")


def jobs_from(cat: Catalog, removed: str = "copy") -> list[tuple[Job, str | None]]:
    """(job, companion file) per photo, from the saved cull."""
    out = []
    for pid, path, companion, decision, unsure, reason in cat.db.execute(
            "SELECT p.id, p.path, p.companion, c.decision, c.unsure, COALESCE(c.reason, '') "
            "FROM cull c JOIN photos p ON p.id = c.id ORDER BY p.path"):
        folder = "Selected" if decision == "selected" else "Removed"
        copy = folder == "Selected" or removed == "copy"
        out.append((Job(pid, folder, unsure, reason, path, copy), companion))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path) -> tuple[int, str]:
    """Copy one original; returns (bytes, checksum). The original is opened read-only."""
    before = os.stat(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + PART)
    h = hashlib.sha256()
    size = 0
    try:
        with open(src, "rb") as f, open(tmp, "wb") as g:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
                g.write(block)
                size += len(block)
        after = os.stat(src)
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or size != before.st_size:
            raise SourceChanged(f"{src.name} changed while it was being copied")
        digest = h.hexdigest()
        if sha256_of(tmp) != digest:
            raise OSError(f"the copy of {src.name} does not match the original (disk error?)")
        os.utime(tmp, ns=(before.st_atime_ns, before.st_mtime_ns))     # keep the original's date
        os.replace(tmp, dst)
        return size, digest
    finally:
        if tmp.exists():
            tmp.unlink()


def read_manifest(out: Path) -> dict[str, dict]:
    path = out / MANIFEST
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["original"]: row for row in csv.DictReader(f)}


def apply(root: Path, out: Path, jobs: list[tuple[Job, str | None]], progress=print) -> ApplyReport:
    """Copy (or list) every photo into out/Selected and out/Removed. Safe to run again."""
    check_output(root, out)
    done = read_manifest(out)
    files: list[tuple[Job, str]] = []
    for job, companion in jobs:
        files.append((job, job.original))
        if companion:
            files.append((job, companion))
    todo = [(j, f) for j, f in files if j.copy and not _already(out, j, f, done)
            and not (done.get(f, {}).get("copy") and (out / done[f]["copy"]).exists())]
    need = 0
    for _, f in todo:
        try:
            need += os.stat(root / f).st_size
        except OSError:
            pass
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if need + SPARE_BYTES > free:
        raise OutputError(f"Not enough space on this laptop: the copies need {need / 1e9:,.1f} GB, "
                          f"{free / 1e9:,.1f} GB is free. Free some space, choose another folder, or "
                          "copy only Selected and list the Removed photos:  --removed list")
    report = ApplyReport()
    new_file = not (out / MANIFEST).exists()
    start = last = time.perf_counter()
    with open(out / MANIFEST, "a", newline="", encoding="utf-8") as m:
        writer = csv.DictWriter(m, FIELDS)
        if new_file:
            writer.writeheader()
        for k, (job, rel) in enumerate(files, 1):
            report.folders[job.folder] = report.folders.get(job.folder, 0) + 1
            row = {"folder": job.folder, "unsure": job.unsure, "reason": job.reason, "original": rel}
            earlier = done.get(rel)
            if not job.copy and earlier and earlier.get("copy") and earlier["folder"] != job.folder \
                    and (out / earlier["copy"]).exists():
                job = Job(job.photo_id, job.folder, job.unsure, job.reason, job.original, True)   # move it
            if not job.copy:
                if rel not in done:
                    writer.writerow({**row, "copy": "", "bytes": "", "sha256": ""})
                report.listed += 1
                continue
            if _already(out, job, rel, done):
                report.skipped += 1
                continue
            if earlier and earlier.get("copy") and (out / earlier["copy"]).exists():
                # moved between Selected and Removed since the last run: move the copy too
                dst = out / job.folder / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(out / earlier["copy"], dst)
                writer.writerow({**row, "copy": f"{job.folder}/{rel}", "bytes": earlier["bytes"],
                                 "sha256": earlier["sha256"]})
                report.moved += 1
                continue
            src = root / rel
            if not root.is_dir():
                raise OutputError("The photo drive is not connected any more. Plug it back in and run "
                                  "the same command again: it carries on where it stopped.")
            dst = out / job.folder / rel
            size, digest = copy_verified(src, dst)
            writer.writerow({**row, "copy": f"{job.folder}/{rel}", "bytes": size, "sha256": digest})
            m.flush()
            report.copied += 1
            report.bytes += size
            now = time.perf_counter()
            if now - last >= 2:
                last = now
                rate = report.bytes / max(now - start, 1e-6)
                left = sum(1 for j, _ in files[k:] if j.copy)
                progress(f"  copied {report.copied:,} files ({report.bytes / 1e9:,.1f} GB, "
                         f"{rate / 1e6:,.0f} MB/s), {left:,} to go")
    report.seconds = time.perf_counter() - start
    return report


def _already(out: Path, job: Job, rel: str, done: dict[str, dict]) -> bool:
    row = done.get(rel)
    if not row or not row.get("copy") or row["folder"] != job.folder:
        return False
    path = out / row["copy"]
    return path.exists() and str(path.stat().st_size) == row["bytes"]


@dataclass
class UndoReport:
    removed: int = 0
    kept_changed: list[str] = field(default_factory=list)
    missing: int = 0


def undo(out: Path) -> UndoReport:
    """Remove what ks apply copied into `out` (only files whose checksum still matches)."""
    rows = read_manifest(out)
    if not rows:
        raise OutputError(f"No KeepSake manifest in {out}: nothing to undo.")
    report = UndoReport()
    for row in rows.values():
        if not row.get("copy"):
            continue
        path = out / row["copy"]
        if not path.exists():
            report.missing += 1
        elif sha256_of(path) == row["sha256"]:
            path.unlink()
            report.removed += 1
        else:
            report.kept_changed.append(row["copy"])
    for folder in ("Selected", "Removed"):
        base = out / folder
        if base.is_dir():
            for d in sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
                if not any(d.iterdir()):
                    d.rmdir()
            if not any(base.iterdir()):
                base.rmdir()
    if not report.kept_changed:
        (out / MANIFEST).unlink()
        if (out / README).exists():
            (out / README).unlink()
        if out.is_dir() and not any(out.iterdir()):
            out.rmdir()
    return report
