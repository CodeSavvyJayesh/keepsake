"""Match a studio's selected photos back to the original files they came from.

Studios usually keep two folders per wedding: every photo shot, and a separate folder of
copies of the ones they chose. Matching each copy to its original gives us the labels the
taste model learns from: selected (1) or not selected (0) for every original.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# Given some files, return {file: (camera, capture time, fraction of a second)}
MetaFn = Callable[[list[Path]], dict[Path, tuple]]


@dataclass
class MatchResult:
    matches: dict[Path, Path] = field(default_factory=dict)  # original -> selected copy
    how: Counter = field(default_factory=Counter)            # matching method -> count
    unmatched: list[Path] = field(default_factory=list)      # selected copies with no original
    repeats: list[Path] = field(default_factory=list)        # extra copies of an already-matched original


def match_selected(originals: Iterable[Path], selected: Iterable[Path],
                   meta_fn: MetaFn | None = None,
                   size_fn: Callable[[Path], int] = os.path.getsize) -> MatchResult:
    """Match each selected copy to exactly one original.

    Tried in order, stopping at the first that gives a single answer:
      1. same file name (ignoring upper/lower case)
      2. same name without the extension, for edited exports (IMG_1043.CR3 -> IMG_1043.jpg)
      3. same name up to the first dot, for typos made while copying (DSC_0342.JPGG.JPG)
      4. if several originals share the name: the one with the same file size
      5. still several: the one with the same camera and capture time (needs meta_fn)
    Anything that stays ambiguous is reported as unmatched rather than guessed.
    """
    result = MatchResult()
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    by_base: dict[str, list[Path]] = defaultdict(list)
    for p in originals:
        by_name[p.name.lower()].append(p)
        by_stem[p.stem.lower()].append(p)
        by_base[_base(p)].append(p)

    ambiguous: list[tuple[Path, list[Path]]] = []
    for s in selected:
        candidates = by_name.get(s.name.lower()) or by_stem.get(s.stem.lower()) or []
        how = "file name"
        if not candidates:
            # a typo in the extension: only trust it when the copy is byte-for-byte the same size
            candidates = [c for c in by_base.get(_base(s), []) if size_fn(c) == size_fn(s)]
            how = "file name with typo + size"
        if not candidates:
            result.unmatched.append(s)
        elif len(candidates) == 1:
            _add(result, candidates[0], s, how)
        elif how != "file name":
            ambiguous.append((s, candidates))
        else:
            same_size = [c for c in candidates if size_fn(c) == size_fn(s)]
            if len(same_size) == 1:
                _add(result, same_size[0], s, "file name + size")
            else:
                ambiguous.append((s, same_size or candidates))

    if ambiguous and meta_fn is not None:
        files = sorted({s for s, _ in ambiguous} | {c for _, cs in ambiguous for c in cs})
        meta = meta_fn(files)
        for s, candidates in ambiguous:
            want = meta.get(s)
            hits = [c for c in candidates if want is not None and meta.get(c) == want]
            if len(hits) == 1:
                _add(result, hits[0], s, "file name + camera + time")
            else:
                result.unmatched.append(s)
    else:
        result.unmatched.extend(s for s, _ in ambiguous)
    return result


def load_label_csv(csv_path: Path, shoot_root: Path) -> dict[str, int]:
    """The studio's choices from a ks match file, keyed like the catalog (relative to the shoot).

    The file lists originals relative to the wedding folder ('Raw image's\\Day 1\\DSC_1.JPG');
    only those under the shoot folder are kept, with that first part removed.
    """
    import csv
    wanted = shoot_root.name.lower()
    out: dict[str, int] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parts = row["original"].replace("\\", "/").split("/", 1)
            if len(parts) == 2 and parts[0].lower() == wanted:
                out[parts[1]] = int(row["selected"])
    return out


def attach_labels(catalog, shoot_root: Path) -> tuple[int, str] | None:
    """Load the wedding's ks match file (if there is one) into the shoot's catalog.

    Returns (photos labelled, file name), or None when the studio's choices are not known.
    """
    from .config import DATA_DIR
    from .ingest import shoot_for, slug
    label_file = DATA_DIR / "labels" / f"{shoot_for(shoot_root).name}.csv"      # ks match, Raw + Selected given
    if not label_file.exists():
        label_file = DATA_DIR / "labels" / f"{slug(shoot_root.parent.name)}.csv"   # ks match, wedding folder
    if not label_file.exists():
        return None
    return catalog.save_labels(load_label_csv(label_file, shoot_root), label_file.name), label_file.name


def _base(path: Path) -> str:
    """The name up to the first dot: 'DSC_0342.JPGG.JPG' -> 'dsc_0342'."""
    return path.name.split(".", 1)[0].lower()


def _add(result: MatchResult, original: Path, copy: Path, how: str) -> None:
    if original in result.matches:
        result.repeats.append(copy)
    else:
        result.matches[original] = copy
        result.how[how] += 1
