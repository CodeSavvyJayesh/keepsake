"""Bursts: runs of photos one camera took within moments of each other.

A burst is the same moment shot several times; the studio usually keeps one frame of it.
Phase 3 builds on this (duplicates across cameras, near-identical frames); here it is the
simple version used to test Phase 2's measurements: same camera body, each frame within
`gap` seconds of the previous one.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

BURST_GAP_S = 2.0


def group_bursts(photos: Iterable[tuple[int, str, float]], gap: float = BURST_GAP_S) -> list[list[int]]:
    """[(id, body, taken_s), ...] -> bursts as lists of ids, in time order. Singles included."""
    by_body: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for photo_id, body, taken in photos:
        if taken is not None:
            by_body[body or ""].append((taken, photo_id))
    bursts: list[list[int]] = []
    for frames in by_body.values():
        frames.sort()
        current: list[int] = []
        last = None
        for taken, photo_id in frames:
            if current and taken - last > gap:
                bursts.append(current)
                current = []
            current.append(photo_id)
            last = taken
        if current:
            bursts.append(current)
    return bursts
