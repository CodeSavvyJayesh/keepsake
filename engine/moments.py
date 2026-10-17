"""Phase 3: moments - photos of the same instant, from every camera, as one group.

A studio keeps roughly one frame per moment, so the moment (not the single photo) is the unit
the culling decision is made in.

1. Runs per camera. Each camera's photos, in time order, are cut into runs: a new run starts
   after a pause (SEG_GAP), when the picture changes (CLIP similarity to the run so far below
   SEG_SIM) or when a run gets long (SEG_SPAN). A burst is one run.
2. Camera clocks. Each camera body's clock can be off: the wrong time zone, or never set (a Leica
   here says 2015). Runs of a camera are matched (CLIP) with runs of the main camera in the same
   event folder; the time differences of the matching runs pile up at one value, the clock
   offset. Bodies without enough matches keep their own clock and are not joined with others.
3. Moments. Runs of different cameras are joined when close on the lined-up clocks and similar
   in CLIP (two photographers on the same moment).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Runs and moments
SEG_GAP = 5.0              # same camera: a pause this long (s) starts a new run...
SEG_SIM = 0.85             # ...so does a picture this different from the run so far...
SEG_SPAN = 60.0            # ...or a run longer than this (s)
CROSS_SIM = 0.88           # runs of two cameras: one moment if this similar...
CROSS_GAP = 15.0           # ...and this close (s) on the lined-up clocks...
MOMENT_SPAN = 60.0         # ...as long as the joined moment stays this short (s): no chains

# Clock alignment
MATCH_SIM = 0.88           # runs this similar are taken to be the same moment
OFFSET_WINDOW = 120.0      # seconds: matches agreeing within this support the same offset
MIN_SUPPORT = 4            # matches needed to trust an offset
MAX_RUNS = 600             # runs per camera used to find its offset


@dataclass
class Photo:
    id: int
    body: str
    taken: float | None
    event: str


@dataclass
class Run:
    body: str
    event: str
    members: list[int]          # indexes into the photo list, in time order
    t0: float | None
    t1: float | None
    centroid: np.ndarray


@dataclass
class Alignment:
    offsets: dict[str, float] = field(default_factory=dict)   # body -> seconds to add
    support: dict[str, int] = field(default_factory=dict)     # body -> matching runs found
    reference: str | None = None


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def camera_runs(photos: list[Photo], vectors: np.ndarray) -> list[Run]:
    """Each camera's photos cut into runs (bursts and continuous shooting of one thing)."""
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    runs: list[Run] = []
    for i, p in enumerate(photos):
        if p.taken is None:
            runs.append(Run(p.body, p.event, [i], None, None, vectors[i]))
        else:
            by_key[(p.body, p.event)].append(i)
    for (body, event), idx in by_key.items():
        idx.sort(key=lambda i: photos[i].taken)
        current: list[int] = []
        total = None
        for i in idx:
            t = photos[i].taken
            if current:
                gap = t - photos[current[-1]].taken
                span = t - photos[current[0]].taken
                similar = float(vectors[i] @ _unit(total)) >= SEG_SIM
                if gap > SEG_GAP or span > SEG_SPAN or not similar:
                    runs.append(Run(body, event, current, photos[current[0]].taken,
                                    photos[current[-1]].taken, _unit(total)))
                    current, total = [], None
            current.append(i)
            total = vectors[i].copy() if total is None else total + vectors[i]
        if current:
            runs.append(Run(body, event, current, photos[current[0]].taken,
                            photos[current[-1]].taken, _unit(total)))
    return runs


def densest(values: np.ndarray, window: float) -> tuple[float, int]:
    """(median of the densest run of values within `window` of each other, its size)."""
    v = np.sort(values)
    if not len(v):
        return 0.0, 0
    ends = np.searchsorted(v, v + window, side="right")
    i = int(np.argmax(ends - np.arange(len(v))))
    run = v[i:ends[i]]
    return float(np.median(run)), len(run)


def align_clocks(photos: list[Photo], vectors: np.ndarray, runs: list[Run] | None = None,
                 seed: int = 0) -> Alignment:
    """Clock offset per camera body, relative to the body with the most photos."""
    runs = runs if runs is not None else camera_runs(photos, vectors)
    counts: dict[str, int] = defaultdict(int)
    for p in photos:
        if p.taken is not None:
            counts[p.body] += 1
    out = Alignment()
    if not counts:
        return out
    reference = max(counts, key=counts.get)
    out.reference = reference
    out.offsets[reference] = 0.0
    out.support[reference] = counts[reference]
    timed = [r for r in runs if r.t0 is not None]
    ref_by_event: dict[str, list[Run]] = defaultdict(list)
    for r in timed:
        if r.body == reference:
            ref_by_event[r.event].append(r)
    ref_mats = {e: np.stack([r.centroid for r in rs]) for e, rs in ref_by_event.items()}
    rng = np.random.default_rng(seed)
    for body in counts:
        if body == reference:
            continue
        mine = [r for r in timed if r.body == body and r.event in ref_mats]
        if len(mine) > MAX_RUNS:
            mine = [mine[i] for i in rng.choice(len(mine), MAX_RUNS, replace=False)]
        deltas = []
        for r in mine:
            sims = ref_mats[r.event] @ r.centroid
            best = int(np.argmax(sims))
            if sims[best] >= MATCH_SIM:
                ref = ref_by_event[r.event][best]
                deltas.append((ref.t0 + ref.t1) / 2 - (r.t0 + r.t1) / 2)
        offset, n = densest(np.array(deltas), OFFSET_WINDOW) if deltas else (0.0, 0)
        out.support[body] = n
        if n >= MIN_SUPPORT:
            out.offsets[body] = offset
    return out


def group_moments(photos: list[Photo], vectors: np.ndarray, alignment: Alignment,
                  runs: list[Run] | None = None) -> list[list[int]]:
    """Groups of photo ids (each group = one moment), in time order. Singles included."""
    runs = runs if runs is not None else camera_runs(photos, vectors)
    parent = list(range(len(runs)))
    lo: dict[int, float] = {}
    hi: dict[int, float] = {}

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    shifted = []
    for k, r in enumerate(runs):
        offset = alignment.offsets.get(r.body)
        if r.t0 is not None and offset is not None:
            shifted.append((r.event, r.t0 + offset, r.t1 + offset, k))
            lo[k], hi[k] = r.t0 + offset, r.t1 + offset
    by_event: dict[str, list] = defaultdict(list)
    for item in shifted:
        by_event[item[0]].append(item)
    for items in by_event.values():
        items.sort(key=lambda x: x[1])
        starts = np.array([x[1] for x in items])
        for a, (_, t0, t1, k) in enumerate(items):
            end = int(np.searchsorted(starts, t1 + CROSS_GAP, side="right"))
            for _, u0, u1, j in items[a + 1:end]:
                if runs[j].body != runs[k].body and float(runs[j].centroid @ runs[k].centroid) >= CROSS_SIM:
                    ra, rb = root(k), root(j)
                    if ra != rb and max(hi[ra], hi[rb]) - min(lo[ra], lo[rb]) <= MOMENT_SPAN:
                        keep, gone = min(ra, rb), max(ra, rb)
                        parent[gone] = keep
                        lo[keep], hi[keep] = min(lo[ra], lo[rb]), max(hi[ra], hi[rb])
    groups: dict[int, list[int]] = defaultdict(list)
    for k, r in enumerate(runs):
        groups[root(k)].extend(r.members)

    def when(i: int) -> float:
        p = photos[i]
        offset = alignment.offsets.get(p.body, 0.0)
        return (p.taken + offset) if p.taken is not None else float("inf")

    out = []
    for members in groups.values():
        members.sort(key=lambda i: (when(i), photos[i].id))
        out.append([photos[i].id for i in members])
    out.sort(key=lambda g: g[0])
    return out


def group_quality(groups: list[list[int]], selected: dict[int, int]) -> dict:
    """How the groups line up with the studio's picks (a good grouping: about one pick each)."""
    multi = [g for g in groups if len(g) > 1]
    picks_per = [sum(selected.get(i, 0) for i in g) for g in groups]
    total_picks = sum(picks_per)
    return {
        "groups": len(groups),
        "photos": sum(len(g) for g in groups),
        "in_multi": sum(len(g) for g in multi),
        "multi_groups": len(multi),
        "largest": max((len(g) for g in groups), default=0),
        "groups_with_pick": sum(1 for p in picks_per if p),
        "picks": total_picks,
        # if only the single best photo of each group were kept, at most this many picks survive
        "picks_if_one_per_group": sum(1 for p in picks_per if p),
    }
