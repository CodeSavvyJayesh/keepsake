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


Capture = tuple   # (body serial or None, camera name or None, capture time in seconds or None)


def _norm(text) -> str | None:
    return str(text).strip().lower() or None if text else None


def match_by_capture(selected: dict, originals: dict, min_support: int = 5) -> tuple[dict, Counter]:
    """Match renamed copies to originals by camera and capture time, which edited exports keep.

    selected / originals: {key: (serial, camera, taken_s)}; an original may add its size in bytes
    as a 4th value, so that two byte-identical copies of one photo in the Raw folder (the same
    frame twice) are both marked, instead of left unsure. Studios often rename their picks on
    export (RBP_1215.jpg is CM2_1758.JPG), but the camera's serial number and the capture time,
    to the hundredth of a second, travel with the file. Tried in order:
      1. same camera body (serial, else camera name) and the same time to 1/100 s
      2. the same, after a whole-camera clock shift (the studio corrected a camera's clock in
         Lightroom): the shift is the one most copies agree on, at least `min_support` of them
      3. copies without the fraction of a second: same body and second, if only one original fits
    Returns ({selected key: original key, or a tuple of identical copies}, {method: count});
    anything unsure stays unmatched.
    """
    size = {o: (v[3] if len(v) > 3 else None) for o, v in originals.items()}

    def one(hits):
        """The single original these hits point to (a tuple when they are identical copies)."""
        hits = list(dict.fromkeys(hits))
        if len(hits) == 1:
            return hits[0]
        sizes = {size[h] for h in hits}
        return tuple(hits) if len(sizes) == 1 and None not in sizes else None

    by_body: dict[str, list] = defaultdict(list)
    exact: dict[tuple, list] = defaultdict(list)
    second: dict[tuple, list] = defaultdict(list)
    for o, (serial, camera, t, *_) in originals.items():
        if t is None:
            continue
        for body in {_norm(serial), _norm(camera)} - {None}:
            by_body[body].append((t, o))
            exact[(body, round(t * 100))].append(o)
            second[(body, int(t))].append(o)

    def body_of(serial, camera):
        for body in (_norm(serial), _norm(camera)):
            if body in by_body:
                return body
        return None

    out, how = {}, Counter()
    rest = []
    for s, (serial, camera, t) in selected.items():
        body = body_of(serial, camera)
        hits = exact.get((body, round(t * 100)), []) if body and t is not None else []
        if hits and one(hits) is not None:
            out[s] = one(hits)
            how["camera + capture time"] += 1
        elif body and t is not None:
            rest.append((s, body, t))

    # 2. a camera's clock corrected after the shoot: one shift for all its photos
    shifts: dict[str, float] = {}
    by_rest_body = defaultdict(list)
    for s, body, t in rest:
        by_rest_body[body].append(t)
    for body, times in by_rest_body.items():
        if len(times) < min_support:
            continue
        votes = Counter()
        for t in times:
            frac = round(t % 1, 2)
            for to, _ in by_body[body]:
                if round(to % 1, 2) == frac and frac and abs(t - to) < 2 * 86400:
                    votes[round(t - to, 2)] += 1
        if votes:
            (best, n), *others = votes.most_common(2) + [(None, 0)]
            if n >= min_support and n >= 3 * others[0][1]:
                shifts[body] = best
    still = []
    for s, body, t in rest:
        if body in shifts:
            hits = exact.get((body, round((t - shifts[body]) * 100)), [])
            if hits and one(hits) is not None:
                out[s] = one(hits)
                how["camera + capture time (clock corrected)"] += 1
                continue
        still.append((s, body, t))

    # 3. no fraction of a second in the copy: the second alone, when only one original fits
    for s, body, t in still:
        if round(t % 1, 3) == 0:
            hits = second.get((body, int(t - shifts.get(body, 0))), [])
            if hits and one(hits) is not None:
                out[s] = one(hits)
                how["camera + capture second"] += 1
    return out, how


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
