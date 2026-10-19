"""Phase 6: one photo per shot, and the best frame of each shot.

A "shot" is a set of near-identical frames: the same moment (engine/moments.py) and almost the
same picture (CLIP similarity >= SHOT_SIM). Studios keep about one photo per shot: in the 400-photo
re-pick test the studio's 40 picks were 39 different shots. So KeepSake chooses shots, and from
each chosen shot only its best frame; the other frames are near-duplicates.

The best frame is chosen by a frame ranker learned from the studio's choices inside shots (which
frame of a burst they kept), minus penalties for what an editor never wants: closed or
half-closed eyes and turned heads among the main people.
Measured on 9 weddings (each decided by a model trained on the other 8): with the same number of
photos kept, duplicates went from 1,509 to 0 and the studio's shots covered from 26% to 32%.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .learn import Logistic

SHOT_SIM = 0.90


def group_shots(moment: np.ndarray, clip: np.ndarray, groups: list | None = None) -> np.ndarray:
    """Shot number per row: frames of one moment whose CLIP vectors are this similar to the
    shot's first frame. `groups` (e.g. the wedding of each row) keeps weddings apart."""
    v = clip.astype(np.float32)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    by = defaultdict(list)
    for i, m in enumerate(moment):
        by[(groups[i] if groups is not None else 0, int(m) if m else -i - 1)].append(i)
    shot = np.full(len(moment), -1)
    g = 0
    for members in by.values():
        sims = v[members] @ v[members].T
        for a in range(len(members)):
            if shot[members[a]] < 0:
                shot[members[a]] = g
                for b in range(a + 1, len(members)):
                    if shot[members[b]] < 0 and sims[a, b] >= SHOT_SIM:
                        shot[members[b]] = g
                g += 1
    return shot


def relative_to_shot(X: np.ndarray, shot: np.ndarray) -> np.ndarray:
    """Each feature minus its shot's average: how this frame differs from its siblings."""
    R = np.zeros_like(X, dtype=np.float64)
    by = defaultdict(list)
    for i, s in enumerate(shot):
        by[s].append(i)
    for members in by.values():
        R[members] = X[members] - X[members].mean(axis=0)
    return R


def frame_penalty(X: np.ndarray, names: list[str]) -> np.ndarray:
    """What an editor never wants in the chosen frame: eyes not fully open (closed or half closed)
    and turned heads among the main people."""
    col = {n: i for i, n in enumerate(names)}
    eyes_open = X[:, col["eyes_open"]]
    closed = X[:, col["eyes_closed_any"]]
    turned = X[:, col["turned"]]
    has_people = X[:, col["subjects"]] > 0
    return 2.0 * (1 - eyes_open) * has_people + 2.0 * closed + np.clip(turned - 25, 0, None) / 25


def train_frame_ranker(R: np.ndarray, y: np.ndarray, shot: np.ndarray, rows: np.ndarray):
    """Learns which frame of a shot the studio keeps, from shots of 2+ frames with exactly one
    pick among `rows`. Returns a model, or None when there is too little to learn from."""
    by = defaultdict(list)
    for i in rows:
        by[shot[i]].append(i)
    use = [i for members in by.values() if len(members) >= 2 and y[members].sum() == 1 for i in members]
    if len(use) < 50:
        return None
    use = np.array(use)
    return Logistic().fit(R[use], y[use])


def one_per_shot(rows: np.ndarray, score: np.ndarray, frame_score: np.ndarray, shot: np.ndarray,
                 n: int) -> tuple[np.ndarray, dict[int, int]]:
    """The best frame of each of the n best shots (a shot's worth = its best photo's score).
    Returns (chosen rows, {row: chosen row of its shot} for every row of a chosen shot)."""
    by = defaultdict(list)
    for i in rows:
        by[shot[i]].append(i)
    ranked = sorted(by.values(), key=lambda members: -max(score[i] for i in members))
    chosen, kept_in = [], {}
    for members in ranked[:n]:
        best = max(members, key=lambda i: frame_score[i])
        chosen.append(best)
        for i in members:
            kept_in[i] = best
    return np.array(chosen, dtype=int), kept_in
