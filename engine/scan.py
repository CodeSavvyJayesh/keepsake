"""Find photos in a folder tree. Phase 1 builds the full scanner on top of this."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

from .config import HEIF_TYPES, JPEG_TYPES, PHOTO_TYPES, RAW_TYPES, SKIP_DIRS


def find_photos(root: str | Path) -> Iterator[Path]:
    """Yield every photo under root.

    Skips hidden files (including macOS '._' files), videos, and system or cache folders.
    Uses os.scandir, which is much faster than Path.rglob on big drives because it
    gets the file type without an extra disk call per file.
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
                        yield Path(entry.path)
            except OSError:
                continue


def raw_jpeg_pairs(paths: Iterable[Path]) -> list[tuple[Path, Path]]:
    """Find RAW + JPEG/HEIF pairs (same folder, same name) that a camera saved together.

    They are one photo, so later phases must treat them as one.
    """
    groups: dict[tuple[str, str], list[Path]] = {}
    for p in paths:
        key = (os.path.normcase(str(p.parent)), p.stem.lower())
        groups.setdefault(key, []).append(p)

    pairs = []
    for files in groups.values():
        raws = [f for f in files if f.suffix.lower() in RAW_TYPES]
        others = [f for f in files if f.suffix.lower() in JPEG_TYPES | HEIF_TYPES]
        if raws and others:
            pairs.append((raws[0], others[0]))
    return pairs
