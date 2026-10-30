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

Upgrade (20 Oct): the frame ranker also sees each main face's own measurements (FACE_DETAILS: how
shut the eyes are, head turn, face sharpness, smile, open mouth), and which shots are kept also
uses the photo's full CLIP content (clip_ridge, CLIP_WEIGHT of the score). Blind on 9 weddings:
the studio's exact photo 1,253 -> 1,327 of 6,229 choices, the studio's moment 1,850 -> 1,933;
the studio's own frame of a burst 43% -> 45% (chance 36%).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .learn import Logistic

SHOT_SIM = 0.90
CLIP_WEIGHT = 0.3          # share of the photo score from its full CLIP content
CLIP_RIDGE = 1.0           # how strongly the CLIP content model is held back (regularisation)

FACE_DETAILS = ["n_read", "blink_worst", "blink_mean", "blink_worst_open", "halfblink_n", "closed_n",
                "unsure_n", "down_n", "side_n", "yaw_max", "yaw_mean", "pitch_max", "sharp_min",
                "sharp_mean", "smile_min", "smile_mean2", "jaw_max", "squint_max", "lookdown_max", "relw_max"]


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


def face_details(cat, ids: list[int]) -> np.ndarray:
    """Per photo (rows in the order of `ids`): the main faces' own measurements (FACE_DETAILS),
    zeros for photos without read faces."""
    by: dict[int, list] = defaultdict(list)
    for pid, rel_w, sharp, bl, br, squint, down, smile, jaw, yaw, pitch, eyes in cat.db.execute(
            "SELECT photo_id, rel_w, sharp, blink_l, blink_r, squint, look_down, smile, jaw_open, yaw, pitch, "
            "eyes FROM faces WHERE subject = 1 AND read = 1"):
        by[pid].append((rel_w or 0, sharp or 0, bl or 0, br or 0, squint or 0, down or 0, smile or 0, jaw or 0,
                        abs(yaw or 0), abs(pitch or 0), eyes or ""))
    out = np.zeros((len(ids), len(FACE_DETAILS)), np.float32)
    for row, pid in enumerate(ids):
        fs = by.get(pid)
        if not fs:
            continue
        a = np.array([f[:10] for f in fs], dtype=float)
        blink = (a[:, 2] + a[:, 3]) / 2
        eyes = [f[10] for f in fs]
        plain = np.array([e in ("open", "unsure", "closed") for e in eyes])   # not smiling, down or side
        out[row] = [len(fs), blink.max(), blink.mean(), blink[plain].max() if plain.any() else 0,
                    int(((blink > 0.3) & (blink < 0.55) & plain).sum()), eyes.count("closed"), eyes.count("unsure"),
                    eyes.count("down"), eyes.count("side"), a[:, 8].max(), a[:, 8].mean(), a[:, 9].max(),
                    np.log1p(a[:, 1].min()), np.log1p(a[:, 1].mean()), a[:, 6].min(), a[:, 6].mean(), a[:, 7].max(),
                    a[:, 4].max(), a[:, 5].max(), a[:, 0].max()]
    return out


# The editor's rule for Selected: no closed or half-closed eyes and no faces turned to profile
# among the photo's main people. Eyes lowered (looking down) count as closed: they look closed in
# the photo. A laugh that narrows the eyes is fine. Measured blind on 9 weddings (29 Oct 2026):
# 6.6% of Selected broke the rule before, 0.1% with these weights, for 0.2 points fewer of the
# studio's own picks found. A moment whose every frame breaks it gives its place to another one.
FAULT_SHUT = 0.45            # both eyes at least this shut (MediaPipe blink score, 0 open - 1 shut)
FAULT_HALF = (0.55, 0.30)    # one eye at least 0.55 shut and the other at least 0.30: half closed
FAULT_PROFILE = 50           # head turned at least this many degrees
FAULT_LAUGH = (0.5, 0.45)    # smile and squint at least this: laughing eyes, not a fault
FAULT_SHOT_W, FAULT_FRAME_W = 1.0, 3.0
EYES_SHUT, EYES_HALF, PROFILE = 1, 2, 4


def face_faults(cat, ids: list[int]) -> np.ndarray:
    """Per photo (order of `ids`): 0, or EYES_SHUT / EYES_HALF / PROFILE flags for its main people."""
    pos = {pid: k for k, pid in enumerate(ids)}
    out = np.zeros(len(ids), np.int8)
    for pid, bl, br, yaw, smile, squint in cat.db.execute(
            "SELECT photo_id, blink_l, blink_r, yaw, smile, squint FROM faces WHERE subject = 1 AND read = 1"):
        k = pos.get(pid)
        if k is None:
            continue
        lo, hi = sorted((bl or 0.0, br or 0.0))
        laughing = (smile or 0) >= FAULT_LAUGH[0] and (squint or 0) >= FAULT_LAUGH[1]
        if not laughing and lo >= FAULT_SHUT:
            out[k] |= EYES_SHUT
        elif not laughing and hi >= FAULT_HALF[0] and lo >= FAULT_HALF[1]:
            out[k] |= EYES_HALF
        if abs(yaw or 0) >= FAULT_PROFILE:
            out[k] |= PROFILE
    return out


def fault_words(flags: int) -> str:
    if flags & EYES_SHUT:
        return "eyes closed (or lowered) on a main person"
    if flags & EYES_HALF:
        return "eyes half closed on a main person"
    if flags & PROFILE:
        return "a main person's face is turned to the side"
    return ""


def centred_clip(clip: np.ndarray, groups) -> np.ndarray:
    """Unit CLIP vectors minus each wedding's average (removes a wedding's overall look)."""
    v = clip.astype(np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    groups = np.asarray(groups)
    for g in set(groups.tolist()):
        v[groups == g] -= v[groups == g].mean(axis=0)
    return v


def clip_ridge(C: np.ndarray, y: np.ndarray, strength: float = CLIP_RIDGE) -> np.ndarray:
    """Weights for 'how much this content looks like the studio's picks': class-balanced ridge
    regression on centred CLIP vectors (closed form, NumPy only). Score = C @ weights."""
    y = y.astype(np.float64)
    w = np.where(y == 1, (len(y) - y.sum()) / max(y.sum(), 1), 1.0)
    lam = strength * w.sum() / C.shape[1]
    target = y - np.average(y, weights=w)
    return np.linalg.solve(C.T @ (C * w[:, None]) + lam * np.eye(C.shape[1]), C.T @ (w * target))


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


# Shortlist: keep the moments the studio is most likely to pick from, until they hold the target
# share of the studio's picks (expected), with the best frame of each moment in Selected and its
# other frames one click away (Swap frame). How many moments that takes is decided per wedding.
# Measured blind on 10 weddings (30 Oct 2026, each judged by a model that never saw that couple):
# target 0.94 with the face rule at weight 3 kept 29-48% of the photos (38% on average) and held
# 89% of the studio's picked moments (every wedding at least 86%) and 57% of their exact photos,
# with 0.03% of Selected breaking the face rule. SHORTLIST_CAL turns a photo's score (z within its
# wedding) into the chance that the studio picks it, fitted on those blind scores (about the same
# on any 9 of the 10 weddings: a 0.70-0.83, b -2.39 to -2.26).
SHORTLIST_TARGET = 0.94
SHORTLIST_CAL = (0.81, -2.33)
SHORTLIST_FAULT_W = 3.0      # a moment whose best frame breaks the face rule counts for much less


def shortlist(rows: np.ndarray, score: np.ndarray, frame_score: np.ndarray, shot: np.ndarray,
              target: float = SHORTLIST_TARGET) -> tuple[np.ndarray, dict[int, int]]:
    """The best frame of each moment, best moments first, until the moments kept are expected to hold
    `target` of the studio's picks. Moments are ranked by how many picks they are expected to hold
    (the sum of their frames' chances), so a long burst of a key moment counts for more than one
    frame. Returns the same as one_per_shot."""
    a, b = SHORTLIST_CAL
    by = defaultdict(list)
    for i in rows:
        by[shot[i]].append(i)
    groups = list(by.values())
    expected = np.array([float(np.sum(1 / (1 + np.exp(-(a * score[m] + b))))) for m in (np.array(g) for g in groups)])
    order = np.argsort(-expected, kind="mergesort")
    need = target * expected.sum()
    chosen, kept_in, got = [], {}, 0.0
    for k in order:
        if got >= need:
            break
        members = groups[k]
        best = max(members, key=lambda i: frame_score[i])
        chosen.append(best)
        for i in members:
            kept_in[i] = best
        got += expected[k]
    return np.array(chosen, dtype=int), kept_in


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
