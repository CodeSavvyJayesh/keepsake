"""Phase 3: learning the studio's taste from its past choices.

Every photo becomes one row of numbers ("features") built from what Phases 1-3 measured:
picture quality, faces and smiles, who is in it (the couple), the moment it belongs to and how
it compares with the other frames of that moment, and CLIP's description of its content.
A model learns from the studio's past choices which rows they keep.

Models
  LightGBM (gradient-boosted decision trees), the main model, when installed.
  Logistic regression (written here in NumPy), a simple baseline that always works.

Honest testing: the model is always tested on a wedding (or, with one wedding, an event) it
did not learn from. Reported: AUC (0.5 = chance, 1 = perfect), the share of the studio's picks
caught in the top 20/30/50% of photos, and how big the shortlist must be to catch 90% / 95%.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .quality import auc

CLIP_COMPONENTS = 24       # CLIP's 512 numbers, condensed to the 24 that vary most (PCA)
TOP_PEOPLE = 10


@dataclass
class Dataset:
    ids: list[int]
    names: list[str]
    X: np.ndarray
    y: np.ndarray                      # 1 selected, 0 not, -1 unknown
    group: list[str]                   # wedding or event, for honest testing
    moment: np.ndarray                 # moment id per photo (0 if unknown)
    clip: np.ndarray | None = None     # N x 512, for the pooled PCA
    meta: dict = field(default_factory=dict)


def _log(v, floor=1e-3):
    return math.log(max(float(v), floor)) if v is not None else 0.0


def _num(v, default=0.0):
    return float(v) if v is not None else default


def photo_features(cat: Catalog, wedding: str) -> Dataset:
    """One feature row per read photo of one shoot (without the CLIP components yet)."""
    rows = list(cat.db.execute("""
        SELECT p.id, p.folder, p.body, p.taken_s, p.iso, p.shutter, p.aperture, p.focal, p.flash,
               q.sharp_global, q.sharp_peak, q.sharp_rel, q.peak_contrast, q.noise, q.motion,
               q.luma_mean, q.luma_rel, q.luma_p01, q.luma_p99, q.clip_low, q.clip_high,
               COALESCE(l.selected, -1), COALESCE(m.moment, 0), COALESCE(m.size, 1), COALESCE(m.pos, 0)
        FROM photos p
        LEFT JOIN quality q ON q.id = p.id AND q.ok = 1
        LEFT JOIN labels l ON l.id = p.id
        LEFT JOIN moments m ON m.id = p.id
        WHERE p.status = 'done' ORDER BY p.id"""))
    faces = defaultdict(list)
    for r in cat.db.execute("""SELECT photo_id, w, rel_w, score, sharp, contrast, subject, read, smile,
                                      yaw, eyes, person, x, y, h FROM faces"""):
        faces[r[0]].append(r[1:])
    # same-camera activity: photos within a minute either side
    times = defaultdict(list)
    for r in rows:
        if r[3] is not None:
            times[r[2]].append(r[3])
    times = {body: np.array(sorted(t)) for body, t in times.items()}   # sorted arrays: fast lookups

    names, data = None, []
    for r in rows:
        (pid, folder, body, taken, iso, shutter, aperture, focal, flash, sg, sp, srel, pc, noise,
         motion, luma, lrel, p01, p99, clow, chigh, _sel, _moment, msize, mpos) = r
        f = {
            "sharp_global": _log(sg), "sharp_peak": _log(sp), "sharp_rel": _num(srel),
            "peak_contrast": _num(pc), "noise": _log(noise), "motion": _num(motion),
            "luma": _num(luma, 128), "luma_rel": _num(lrel), "luma_p01": _num(p01), "luma_p99": _num(p99),
            "clip_low": _num(clow), "clip_high": _num(chigh),
            "iso": _log(iso, 1), "shutter": _log(shutter, 1e-5), "aperture": _log(aperture, 0.5),
            "focal": _log(focal, 1), "flash": float(bool(flash) and int(flash) & 1),
            "moment_size": math.log(msize), "moment_pos": mpos / (msize - 1) if msize > 1 else 0.5,
            "single": float(msize == 1),
        }
        if taken is not None and body in times:
            t = times[body]
            f["activity"] = math.log(np.searchsorted(t, taken + 60, "right") - np.searchsorted(t, taken - 60))
        else:
            f["activity"] = 0.0
        fl = faces.get(pid, [])
        good = [x for x in fl if (x[2] or 0) >= 0.75]
        subj = [x for x in fl if x[5]]
        read = [x for x in subj if x[6] and x[9] in ("open", "closed", "smiling", "unsure")]
        smiles = [x[7] or 0 for x in read]
        people = {x[10] for x in fl if x[10] is not None}
        couple = [x for x in fl if x[10] in (1, 2)]
        couple_read = [x for x in couple if x[6]]
        largest = max(good, key=lambda x: x[0]) if good else None
        f.update({
            "faces": len(good), "subjects": len(subj),
            "largest_face": largest[1] if largest else 0.0,
            "face_focus": _log((largest[3] or 0) / ((largest[4] or 0) ** 2 + 10)) if largest else -7.0,
            "smile_mean": float(np.mean(smiles)) if smiles else 0.0,
            "smile_max": max(smiles) if smiles else 0.0,
            "eyes_open": sum(x[9] in ("open", "smiling") for x in read) / len(read) if read else 0.5,
            "eyes_closed_any": float(any(x[9] == "closed" for x in read)),
            "turned": float(np.mean([abs(x[8] or 0) for x in read])) if read else 0.0,
            "p1": float(1 in people), "p2": float(2 in people), "couple": float({1, 2} <= people),
            "top5": float(len([p for p in people if p <= 5])),
            "top10": float(len([p for p in people if p <= TOP_PEOPLE])),
            "couple_size": max((x[1] for x in couple), default=0.0),
            "couple_smile": float(np.mean([x[7] or 0 for x in couple_read])) if couple_read else 0.0,
            "couple_eyes_open": float(any(x[9] in ("open", "smiling") for x in couple_read)),
        })
        if largest:
            cx = (largest[11] + largest[0] / 2) / 1620
            cy = (largest[12] + largest[13] / 2) / 1620
            f["face_centre"] = math.hypot(cx - 0.5, cy - 0.33)
        else:
            f["face_centre"] = 1.0
        if names is None:
            names = list(f)
        data.append([f[n] for n in names])

    ids = [r[0] for r in rows]
    X = np.array(data, dtype=np.float32) if data else np.zeros((0, 0), np.float32)
    y = np.array([r[21] for r in rows], dtype=np.int8)
    moment = np.array([r[22] for r in rows], dtype=np.int64)
    events = [f"{wedding} / {(r[1] or '').split('/')[0]}" for r in rows]
    ds = Dataset(ids, names or [], X, y, events, moment, meta={"wedding": wedding})
    ds = add_moment_relative(ds, ["smile_mean", "largest_face", "sharp_peak", "couple_smile",
                                  "couple_size", "face_focus", "eyes_open", "subjects"])
    clip_ids, clip_vecs = cat.clip_vectors()
    if len(clip_ids):
        index = {pid: i for i, pid in enumerate(clip_ids)}
        ds.clip = np.zeros((len(ids), clip_vecs.shape[1]), np.float32)
        for i, pid in enumerate(ids):
            if pid in index:
                ds.clip[i] = clip_vecs[index[pid]]
        ds = add_clip_moment_features(ds)
    return ds


# --- Features of past weddings, kept on this laptop between runs -----------------------------

FEATURES_FILE = "features.npz"


def _code_version() -> str:
    """Changes whenever the code that computes features changes (then everything is recomputed)."""
    import hashlib
    here = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for name in ("learn.py", "shots.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def _catalog_stamp(db: Path) -> str:
    """Size and time of the catalog (and its write-ahead log): any change to it changes this."""
    parts = []
    for p in (db, db.with_name(db.name + "-wal")):
        try:
            st = p.stat()
            parts.append(f"{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append("-")
    return "|".join(parts)


def past_wedding(db: Path, details) -> Dataset:
    """photo_features (+ meta['faces'] = details(cat, ids)) of a past wedding, from the copy saved
    next to its catalog when the catalog has not changed since; computed and saved otherwise.
    The saved copy holds exactly the same numbers, so the choices do not change, only the time."""
    name = db.parent.name
    key = f"{_code_version()}|{name}|{_catalog_stamp(db)}"
    cache = db.parent / FEATURES_FILE
    try:
        with np.load(cache, allow_pickle=False) as z:
            if str(z["key"]) == key:
                ds = Dataset([int(i) for i in z["ids"]], [str(n) for n in z["names"]], z["X"], z["y"],
                             [str(g) for g in z["group"]], z["moment"],
                             z["clip"].astype(np.float32) if z["clip"].size else None, meta={"wedding": name})
                ds.meta["faces"] = z["faces"]
                return ds
    except (OSError, KeyError, ValueError):
        pass
    with Catalog(db) as cat:
        ds = photo_features(cat, name)
        ds.meta["faces"] = details(cat, ds.ids)
    key = f"{_code_version()}|{name}|{_catalog_stamp(db)}"     # after reading (opening may touch the log)
    tmp = cache.with_name("features.tmp.npz")
    try:
        np.savez(tmp, key=np.array(key), ids=np.array(ds.ids, np.int64), names=np.array(ds.names), X=ds.X, y=ds.y,
                 group=np.array(ds.group), moment=ds.moment, faces=ds.meta["faces"],
                 clip=ds.clip.astype(np.float16) if ds.clip is not None else np.zeros(0, np.float16))
        os.replace(tmp, cache)
    except OSError:                       # no space or read-only: just not saved
        try:
            tmp.unlink()
        except OSError:
            pass
    return ds


def add_moment_relative(ds: Dataset, columns: list[str]) -> Dataset:
    """For each column: the photo's rank inside its moment (1 = best) and its gap to the moment's
    average. The studio chooses between frames of one moment, so 'better than its siblings'
    matters more than 'good in general'."""
    by_moment = defaultdict(list)
    for i, m in enumerate(ds.moment):
        by_moment[int(m) if m else -i - 1].append(i)
    new_cols, new_names = [], []
    for name in columns:
        if name not in ds.names:
            continue
        v = ds.X[:, ds.names.index(name)]
        rank = np.zeros(len(v), np.float32)
        gap = np.zeros(len(v), np.float32)
        for members in by_moment.values():
            vals = v[members]
            if len(members) > 1:
                order = np.argsort(-vals, kind="mergesort")
                r = np.empty(len(members), np.float32)
                r[order] = np.arange(len(members)) / (len(members) - 1)
                rank[members] = 1 - r
                gap[members] = vals - vals.mean()
            else:
                rank[members] = 1.0
        new_cols += [rank, gap]
        new_names += [f"{name}_rank_in_moment", f"{name}_vs_moment"]
    ds.X = np.hstack([ds.X, np.stack(new_cols, axis=1)]) if new_cols else ds.X
    ds.names = ds.names + new_names
    return ds


def add_clip_moment_features(ds: Dataset) -> Dataset:
    """How typical the photo is of its moment (mean CLIP similarity to its siblings) and of the
    whole shoot."""
    by_moment = defaultdict(list)
    for i, m in enumerate(ds.moment):
        if m:
            by_moment[int(m)].append(i)
    typical = np.ones(len(ds.ids), np.float32)
    for members in by_moment.values():
        if len(members) > 1:
            v = ds.clip[members]
            s = v @ v.T
            typical[members] = (s.sum(axis=1) - 1) / (len(members) - 1)
    centre = ds.clip.mean(axis=0)
    centre /= np.linalg.norm(centre) + 1e-12
    ds.X = np.hstack([ds.X, typical[:, None], (ds.clip @ centre)[:, None]])
    ds.names = ds.names + ["clip_typical_of_moment", "clip_typical_of_shoot"]
    return ds


def combine(datasets: list[Dataset], by_wedding: bool) -> Dataset:
    """Several shoots into one table, with CLIP condensed by one PCA fitted on all of them."""
    names = datasets[0].names
    X = np.vstack([d.X for d in datasets])
    y = np.concatenate([d.y for d in datasets])
    ids = [i for d in datasets for i in d.ids]
    group = [d.meta["wedding"] if by_wedding else g for d in datasets for g in d.group]
    moment = np.concatenate([d.moment + k * 10_000_000 for k, d in enumerate(datasets)])
    out = Dataset(ids, list(names), X, y, group, moment)
    if all(d.clip is not None for d in datasets):
        clip = np.vstack([d.clip for d in datasets])
        mean = clip.mean(axis=0)
        _, _, vt = np.linalg.svd(clip - mean, full_matrices=False)
        comps = (clip - mean) @ vt[:CLIP_COMPONENTS].T
        out.X = np.hstack([out.X, comps.astype(np.float32)])
        out.names += [f"clip_pc{i + 1}" for i in range(comps.shape[1])]
        out.clip = clip
    return out


# --- Models ------------------------------------------------------------------------

class Logistic:
    """Class-balanced logistic regression with L2, by gradient descent (NumPy only)."""
    name = "logistic regression"

    def fit(self, X, y, l2=1.0, iters=2000, lr=0.2):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w_pos = (len(y) - y.sum()) / max(y.sum(), 1)
        sw = np.where(y == 1, w_pos, 1.0)
        sw /= sw.mean()
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ self.w + self.b, -30, 30)))
            g = (p - y) * sw
            self.w -= lr * (Z.T @ g / len(y) + l2 * self.w / len(y))
            self.b -= lr * g.mean()
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b


class GBM:
    """LightGBM gradient-boosted trees (native API, no scikit-learn needed)."""
    name = "LightGBM"

    def __init__(self, rounds: int = 400):
        self.rounds = rounds

    def fit(self, X, y):
        import lightgbm as lgb
        params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                  "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 7,
                  "scale_pos_weight": (len(y) - y.sum()) / max(y.sum(), 1)}
        self.booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=self.rounds)
        return self

    def predict(self, X):
        return self.booster.predict(X)

    def importance(self, names: list[str], top: int = 10) -> list[tuple[str, float]]:
        gain = self.booster.feature_importance(importance_type="gain")
        total = gain.sum() or 1
        order = np.argsort(-gain)[:top]
        return [(names[i], gain[i] / total) for i in order]


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


# --- Honest testing --------------------------------------------------------------

@dataclass
class Score:
    test: str
    photos: int
    picks: int
    auc: float
    caught: dict          # shortlist share -> share of picks caught
    needed: dict          # share of picks wanted -> shortlist share needed


def score(scores: np.ndarray, y: np.ndarray, test: str) -> Score:
    order = np.argsort(-scores, kind="mergesort")
    found = np.cumsum(y[order]) / max(y.sum(), 1)
    n = len(y)
    caught = {f: float(found[max(0, int(math.ceil(f * n)) - 1)]) for f in (0.2, 0.3, 0.5)}
    needed = {r: float((np.searchsorted(found, r - 1e-9) + 1) / n) for r in (0.9, 0.95)}
    return Score(test, n, int(y.sum()), auc(scores, y), caught, needed)


def leave_one_out(ds: Dataset, make_model) -> tuple[list[Score], object]:
    """Train on every group but one, test on that one; for each group in turn.
    Returns the scores and the last model (for feature importance)."""
    known = ds.y >= 0
    groups = sorted({g for g, k in zip(ds.group, known) if k})
    scores, model = [], None
    for test in groups:
        is_test = np.array([g == test for g in ds.group]) & known
        is_train = ~is_test & known
        if ds.y[is_train].sum() == 0 or ds.y[is_test].sum() == 0:
            continue
        model = make_model().fit(ds.X[is_train], ds.y[is_train])
        scores.append(score(model.predict(ds.X[is_test]), ds.y[is_test], test))
    return scores, model
