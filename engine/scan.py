"""Find photos in a folder tree."""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .config import HEIF_TYPES, JPEG_TYPES, PHOTO_TYPES, RAW_TYPES, SKIP_DIRS


def _photo_entries(root: str | Path) -> Iterator[os.DirEntry]:
    """Every photo file under root, as os.DirEntry.

    Skips hidden files (including macOS '._' files), videos, and system or cache folders.
    Uses os.scandir, which is much faster than Path.rglob on big drives because it gets the
    file type (and on Windows the size and date) without an extra disk call per file.
    """
    stack = [os.fspath(root)]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as it:
                entries = list(it)
        except OSError:          # no permission, drive unplugged, etc.
            continue
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    if name.lower() not in SKIP_DIRS:
                        stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    if os.path.splitext(name)[1].lower() in PHOTO_TYPES:
                        yield entry
            except OSError:
                continue


def find_photos(root: str | Path) -> Iterator[Path]:
    """Yield every photo under root."""
    for entry in _photo_entries(root):
        yield Path(entry.path)


@dataclass(frozen=True)
class FoundFile:
    path: Path
    size: int        # bytes
    mtime_ms: int    # last modified, milliseconds: a changed file gets read again


def scan_photos(root: str | Path) -> list[FoundFile]:
    """Every photo under root with its size and date, sorted by path (roughly disk order)."""
    found = []
    for entry in _photo_entries(root):
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        found.append(FoundFile(Path(entry.path), st.st_size, st.st_mtime_ns // 1_000_000))
    found.sort(key=lambda f: str(f.path).lower())
    return found


def kind_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in RAW_TYPES:
        return "raw"
    if ext in HEIF_TYPES:
        return "heif"
    return "jpeg"


def group_photos(paths: Iterable[Path]) -> list[tuple[Path, Path | None]]:
    """One entry per photo: (main file, companion or None).

    A RAW and a JPEG/HEIF with the same name in the same folder were saved together by the
    camera: they are one photo, so they become (RAW, JPEG) and are kept or removed together.
    """
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in paths:
        groups[(os.path.normcase(str(p.parent)), p.stem.lower())].append(p)

    photos: list[tuple[Path, Path | None]] = []
    for files in groups.values():
        raws = [f for f in files if f.suffix.lower() in RAW_TYPES]
        others = [f for f in files if f.suffix.lower() not in RAW_TYPES]
        if raws and others:
            photos.append((raws[0], others[0]))
            photos.extend((f, None) for f in raws[1:] + others[1:])
        else:
            photos.extend((f, None) for f in files)
    photos.sort(key=lambda t: str(t[0]).lower())
    return photos


def raw_jpeg_pairs(paths: Iterable[Path]) -> list[tuple[Path, Path]]:
    """Just the RAW + JPEG/HEIF pairs (same folder, same name) that a camera saved together."""
    pairs = []
    for main, companion in group_photos(paths):
        if companion is not None and companion.suffix.lower() in JPEG_TYPES | HEIF_TYPES:
            pairs.append((main, companion))
    return pairs
