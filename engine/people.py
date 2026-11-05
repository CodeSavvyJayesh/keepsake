"""Phase 3: who is in each photo - the same person found again across the whole shoot.

Every clear face has an SFace fingerprint (engine/faces.py). Fingerprints of the same person
point the same way (high cosine similarity). Measured on the Neel & Tara wedding: two
different faces in one photo reach 0.55 only 0.2% of the time, while the same face in the
next burst frame is above 0.55 96% of the time.

Simply joining every pair above a threshold chains everyone into one "person" (one wrong link
is enough). So, in three steps:
1. Anchors: faces that are large, sharp and roughly frontal, whose fingerprints are reliable.
2. Chinese whispers (Biemann 2006) on the anchors' nearest-neighbour graph: every face takes
   the person label most of its close neighbours have, repeated until stable. A single wrong
   link cannot pull two crowds together. Then people whose average fingerprints are very close
   are merged (one person split across lighting or events).
3. Every other fingerprinted face joins its closest person if similar enough.

People are ranked by the number of photos they are in; at a wedding persons 1 and 2 are the
bride and groom (checked by eye on the Neel & Tara wedding). Nothing leaves the laptop.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

ANCHOR_PX = 60             # anchor faces: at least this wide in the 1620 px copy...
ANCHOR_FOCUS = 0.05        # ...this sharp for their detail (Laplacian / (contrast^2 + 10))...
ANCHOR_FRONTAL = 0.35      # ...and the nose within 0.35 eye-widths of the eyes' middle
NEIGHBOURS = 10            # nearest neighbours per anchor in the graph
LINK_SIM = 0.55            # graph edges: fingerprints at least this similar
MERGE_SIM = 0.60           # two people whose average fingerprints are this close are one
ASSIGN_SIM = 0.55          # other faces join their closest person at this similarity
MIN_ANCHORS = 3            # a person needs this many anchor faces


@dataclass
class FaceRef:
    photo_id: int
    n: int
    w: float
    emb: np.ndarray                  # unit vector
    focus: float = 1.0
    frontal: float = 0.0             # 0 = facing the camera


def frontal_from_points(points: np.ndarray) -> float:
    """How far the nose sits from the middle of the eyes, in eye-widths (YuNet's 5 points)."""
    p = np.asarray(points, np.float32).reshape(5, 2)
    eye_mid = (p[0, 0] + p[1, 0]) / 2
    eye_w = abs(p[1, 0] - p[0, 0]) + 1e-6
    return float(abs(p[2, 0] - eye_mid) / eye_w)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def cluster_people(faces: list[FaceRef], seed: int = 0) -> dict[tuple[int, int], int]:
    """(photo id, face n) -> person number (1 = in the most photos), for faces of known people."""
    anchors = [f for f in faces if f.w >= ANCHOR_PX and f.focus >= ANCHOR_FOCUS
               and f.frontal < ANCHOR_FRONTAL]
    if len(anchors) < MIN_ANCHORS:
        return {}
    A = np.stack([f.emb for f in anchors]).astype(np.float32)
    photo = np.array([f.photo_id for f in anchors])
    k = min(NEIGHBOURS, len(anchors) - 1)
    edges: list[list[tuple[int, float]]] = [[] for _ in anchors]
    for start in range(0, len(anchors), 2000):
        S = A[start:start + 2000] @ A.T
        rows = np.arange(S.shape[0])
        S[rows, start + rows] = -1
        S[photo[start:start + 2000, None] == photo[None, :]] = -1   # two faces of one photo: never the same person
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, axis=1)
        for r in range(S.shape[0]):
            i = start + r
            for j, s in zip(idx[r], sims[r]):
                if s >= LINK_SIM:
                    edges[i].append((int(j), float(s)))
                    edges[int(j)].append((i, float(s)))

    # Chinese whispers
    label = list(range(len(anchors)))
    order = list(range(len(anchors)))
    rng = random.Random(seed)
    for _ in range(30):
        rng.shuffle(order)
        changed = 0
        for i in order:
            if not edges[i]:
                continue
            votes: dict[int, float] = defaultdict(float)
            for j, s in edges[i]:
                votes[label[j]] += s
            best = max(votes, key=votes.get)
            if best != label[i]:
                label[i] = best
                changed += 1
        if not changed:
            break
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(label):
        groups[lab].append(i)
    people = [g for g in groups.values() if len(g) >= MIN_ANCHORS]
    if not people:
        return {}

    # merge people split in two (closest pair first, averages recomputed after each merge)
    centroids = [_unit(A[g].mean(axis=0)) for g in people]
    while len(people) > 1:
        C = np.stack(centroids)
        S = C @ C.T
        np.fill_diagonal(S, -1)
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        if S[a, b] < MERGE_SIM:
            break
        a, b = min(a, b), max(a, b)
        people[a] += people[b]
        centroids[a] = _unit(A[people[a]].mean(axis=0))
        del people[b], centroids[b]

    # every fingerprinted face joins its closest person, if close enough
    C = np.stack(centroids)
    E = np.stack([f.emb for f in faces]).astype(np.float32)
    S = E @ C.T
    best = S.argmax(axis=1)
    best_sim = S[np.arange(len(faces)), best]
    photos_of: dict[int, set] = defaultdict(set)
    chosen = {}
    for f, k_, s in zip(faces, best, best_sim):
        if s >= ASSIGN_SIM:
            chosen[(f.photo_id, f.n)] = int(k_)
            photos_of[int(k_)].add(f.photo_id)
    ranked = sorted(photos_of, key=lambda p: -len(photos_of[p]))
    rank = {p: r for r, p in enumerate(ranked, 1)}
    return {key: rank[p] for key, p in chosen.items()}
