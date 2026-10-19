"""Phase 4: the cull - KeepSake's own Selected / Removed for a whole shoot.

Two styles:
  studio (default)  Selected looks like the studio's own selection: as many photos as this studio
                    usually keeps (its pick rate in the past weddings), ONE photo per shot (near-
                    identical frames, engine/shots.py), and of each shot its best frame: the frame the
                    studio would keep, never one with closed or half-closed eyes or a turned head when
                    a better frame of the shot exists. Same method as ks blindtest.
  safe (--safe)     a large shortlist that catches TARGET (95%) of the studio's picks, steps 1-4 below.

1. Learn. A model (LightGBM, or logistic regression when LightGBM is missing) learns the
   studio's taste from every past wedding whose studio choices are known (ks match). The shoot
   being culled is never part of its own training: when it is the only wedding with studio
   choices, each of its events is culled by a model trained on its other events.
2. Score. Every photo gets a score. Within each event the scores are put on one scale, and the
   2nd, 3rd ... best frames of one moment are pushed down a little (VARIETY), so a single burst
   cannot fill the shortlist.
3. Cut. How many photos to keep is learned too: the share of photos that had to be kept in the
   training weddings, tested honestly, to catch TARGET (95%) of the studio's picks. The next
   UNSURE_BAND of photos below the cut are also kept, marked 'AI unsure', for a person to check.
4. Explain. Every removed photo gets a plain reason (a similar photo of the same moment is
   kept, it is soft, too dark, or it is simply rated low).

Nothing is ever deleted: photos KeepSake could not read are kept (Selected, 'AI unsure'). In the
safe style eyes are never a reason to remove a photo; in the studio style they only decide which
frame of a shot is kept.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .learn import GBM, Dataset, Logistic, combine, lightgbm_available, photo_features
from .shots import frame_penalty, group_shots, one_per_shot, relative_to_shot, train_frame_ranker

TARGET = 0.95              # catch this share of what the studio would pick...
UNSURE_BAND = 0.05         # ...and keep this share of photos more, just below the cut, as 'AI unsure'
VARIETY = 0.3              # how much the 2nd, 3rd ... frame of one moment is pushed down
FOLDS = 4                  # with a single training event: honest test on 4 parts of it


@dataclass
class CullResult:
    rows: list[tuple]                 # (id, decision, unsure, score 0-100, reason, similar id)
    mode: str                         # 'other weddings' or 'other events'
    trained_on: list[str]
    train_photos: int
    train_picks: int
    keep_share: float                 # share kept before the AI-unsure band
    model: str
    reasons: Counter = field(default_factory=Counter)
    caught: tuple[int, int] | None = None    # (studio picks in Selected, studio picks) if known
    unread: int = 0
    style: str = "safe"
    shots: int = 0                            # studio style: shots in this shoot


def model_factory():
    return (GBM, "LightGBM") if lightgbm_available() else (Logistic, "logistic regression")


def labelled_shoots(cache_dir: Path, exclude: Path | None = None) -> list[Path]:
    """Catalogs of past shoots with the studio's choices and CLIP done (ready to learn from)."""
    out = []
    for db in sorted(cache_dir.glob("*/catalog.db")):
        if exclude is not None and db.resolve() == exclude.resolve():
            continue
        with Catalog(db) as cat:
            n, picks = cat.db.execute("SELECT COUNT(*), COALESCE(SUM(selected), 0) FROM labels").fetchone()
            clip = cat.db.execute("SELECT COUNT(*) FROM clip WHERE ok = 1").fetchone()[0]
        if n and picks and clip:
            out.append(db)
    return out


def share_needed(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Smallest share of photos, best-scored first, holding `target` of the picks."""
    order = np.argsort(-scores, kind="mergesort")
    found = np.cumsum(y[order]) / max(int(y.sum()), 1)
    return float((np.searchsorted(found, target - 1e-9) + 1) / len(y))


def per_event_scale(scores: np.ndarray, events: list[str]) -> np.ndarray:
    out = np.zeros(len(scores))
    by = defaultdict(list)
    for i, e in enumerate(events):
        by[e].append(i)
    for idx in by.values():
        s = scores[idx]
        out[idx] = (s - s.mean()) / (s.std() + 1e-9)
    return out


def with_variety(scores: np.ndarray, moment: np.ndarray, strength: float = VARIETY) -> np.ndarray:
    """Each photo's score minus a little for every better-scored photo of its moment."""
    rank = np.zeros(len(scores))
    by = defaultdict(list)
    for i, m in enumerate(moment):
        by[int(m) if m else -i - 1].append(i)
    for members in by.values():
        if len(members) > 1:
            order = np.argsort(-scores[members], kind="mergesort")
            r = np.empty(len(members))
            r[order] = np.arange(len(members))
            rank[members] = r
    return scores - strength * np.log1p(rank)


def honest_scores(X, y, groups, moment, make_model) -> np.ndarray:
    """Every row scored by a model that did not learn from its group (or, with a single group,
    from its quarter of the moments)."""
    names = sorted(set(groups))
    if len(names) < 2:
        groups = [f"part {int(m) % FOLDS if m else i % FOLDS}" for i, m in enumerate(moment)]
        names = sorted(set(groups))
    groups = np.array(groups)
    out = np.zeros(len(y))
    for g in names:
        test = groups == g
        if y[~test].sum() == 0:
            continue
        model = make_model().fit(X[~test], y[~test])
        out[test] = model.predict(X[test])
    return out


def keep_share_for(X, y, groups, events, moment, make_model, target: float) -> float:
    """The shortlist share that caught `target` of the picks in an honest test on training data."""
    raw = honest_scores(X, y, groups, moment, make_model)
    adj = with_variety(per_event_scale(raw, events), moment)
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    shares = [share_needed(adj[idx], y[idx], target) for idx in by.values() if y[idx].sum()]
    return float(np.mean(shares)) if shares else 1.0


def _feature(ds: Dataset, name: str) -> np.ndarray:
    return ds.X[:, ds.names.index(name)] if name in ds.names else np.zeros(len(ds.ids))


def explain(ds: Dataset, keep: np.ndarray, score: np.ndarray, names: dict[int, str]) -> list[tuple[str, int | None]]:
    """(reason, similar kept photo id) for every photo; kept photos get an empty reason."""
    sharp_peak, sharp_rel = _feature(ds, "sharp_peak"), _feature(ds, "sharp_rel")
    luma_rel, smile = _feature(ds, "luma_rel"), _feature(ds, "smile_mean")
    kept_in = defaultdict(list)
    for i, m in enumerate(ds.moment):
        if m and keep[i]:
            kept_in[int(m)].append(i)
    out = []
    for i, pid in enumerate(ds.ids):
        if keep[i]:
            out.append(("", None))
            continue
        siblings = kept_in.get(int(ds.moment[i]), []) if ds.moment[i] else []
        if siblings:
            best = max(siblings, key=lambda j: score[j])
            why = ""
            if sharp_peak[i] < sharp_peak[best] - 0.25:
                why = " - this one is softer"
            elif smile[i] < smile[best] - 0.15:
                why = " - less smiling in this one"
            out.append((f"Similar photo kept: {names[ds.ids[best]]} (same moment){why}", ds.ids[best]))
        elif sharp_rel[i] <= -1.5:
            out.append(("Soft or blurry compared with this camera's other photos", None))
        elif luma_rel[i] <= -2:
            out.append(("Much darker than the other photos of this event", None))
        elif luma_rel[i] >= 2:
            out.append(("Much brighter than the other photos of this event", None))
        else:
            out.append(("Rated low: less like the photos this studio keeps", None))
    return out


def reason_kind(reason: str) -> str:
    for start, kind in (("Same shot", "same shot: a better frame of it is kept"),
                        ("Similar", "a similar photo of the same moment is kept"), ("Soft", "soft or blurry"),
                        ("Much darker", "too dark"), ("Much brighter", "too bright"), ("Rated", "rated low")):
        if reason.startswith(start):
            return kind
    return "other"


def why_not_this_frame(X: np.ndarray, names: list[str], i: int, best: int) -> str:
    """Why frame i of a shot lost to the shot's kept frame `best`, in plain words."""
    col = {n: k for k, n in enumerate(names)}
    f = lambda name, k: float(X[k, col[name]]) if name in col else 0.0   # noqa: E731
    if f("subjects", i) > 0 and (f("eyes_closed_any", i) > f("eyes_closed_any", best)
                                 or f("eyes_open", i) < f("eyes_open", best) - 0.2):
        return " - eyes closed or half closed in this one"
    if f("turned", i) > 25 and f("turned", i) > f("turned", best) + 10:
        return " - a head is turned in this one"
    if f("sharp_peak", i) < f("sharp_peak", best) - 0.25:
        return " - this one is softer"
    if f("smile_mean", i) < f("smile_mean", best) - 0.15:
        return " - less smiling in this one"
    return ""


def studio_style(cat: Catalog, name: str, me: Dataset, train: list[Dataset], keep_share: float | None,
                 make_model, model_name: str) -> CullResult:
    """Selected like the studio's: its usual share of photos, one per shot, the best frame of each."""
    data = combine(train + [me], by_wedding=True)
    wedding = np.array(data.group)
    mine = np.flatnonzero(wedding == name)
    rows = np.flatnonzero((wedding != name) & (data.y >= 0))
    model = make_model().fit(data.X[rows], data.y[rows])
    my_events = list(me.group)
    score = np.zeros(len(data.y))
    score[mine] = per_event_scale(model.predict(data.X[mine]), my_events)
    shot = group_shots(data.moment, data.clip, list(wedding))
    R = relative_to_shot(data.X, shot)
    ranker = train_frame_ranker(R, data.y, shot, rows)
    frame = np.zeros(len(data.y))
    frame[mine] = (ranker.predict(R[mine]) if ranker is not None else score[mine]) \
        - frame_penalty(data.X[mine], data.names)
    share = keep_share if keep_share is not None else float(data.y[rows].mean())
    n_me = len(mine)
    chosen, kept_in = one_per_shot(mine, score, frame, shot, max(1, int(round(share * n_me))))
    local = {g: k for k, g in enumerate(mine)}
    keep = np.zeros(n_me, bool)
    keep[[local[g] for g in chosen]] = True

    adj = score[mine]
    percentile = np.empty(n_me)
    percentile[np.argsort(adj, kind="mergesort")] = np.linspace(0, 100, n_me) if n_me > 1 else 100
    names = {pid: Path(p).name for pid, p in cat.db.execute("SELECT id, path FROM photos")}
    me_scaled = Dataset(me.ids, data.names, data.X[mine], data.y[mine], my_events, me.moment)
    why = explain(me_scaled, keep, adj, names)
    rows_out, reasons = [], Counter()
    for k, pid in enumerate(me.ids):
        g = mine[k]
        if keep[k]:
            rows_out.append((pid, "selected", 0, round(float(percentile[k]), 1), None, None))
            continue
        best = kept_in.get(g)
        if best is not None:
            best_pid = me.ids[local[best]]
            reason = (f"Same shot as {names[best_pid]}, which is kept"
                      f"{why_not_this_frame(data.X, data.names, g, best)}")
            similar = best_pid
        else:
            reason, similar = why[k]
        reasons[reason_kind(reason)] += 1
        rows_out.append((pid, "removed", 0, round(float(percentile[k]), 1), reason, similar))
    unread = [pid for (pid,) in cat.db.execute("SELECT id FROM photos WHERE status != 'done' "
                                                "AND status != 'missing'")]
    for pid in unread:
        rows_out.append((pid, "selected", 1, None, "Could not be read by KeepSake - kept for you to check", None))
    result = CullResult(rows_out, "other weddings", [d.meta["wedding"] for d in train], len(rows),
                        int(data.y[rows].sum()), share, model_name, reasons, unread=len(unread),
                        style="studio", shots=len(set(shot[mine].tolist())))
    y_me = me.y
    if (y_me == 1).any():
        result.caught = (int(((y_me == 1) & keep).sum()), int((y_me == 1).sum()))
    return result


def cull_shoot(cat: Catalog, name: str, others: list[Path], target: float = TARGET,
               keep_share: float | None = None, make_model=None, style: str = "studio") -> CullResult:
    """Decide Selected / Removed for every photo of one shoot (does not save anything).
    style 'studio': one photo per shot, the studio's usual share (needs other weddings; with
    none, the safe style is used). style 'safe': the 95% shortlist."""
    if make_model is None:
        make_model, model_name = model_factory()
    else:
        model_name = getattr(make_model, "name", "model")
    me = photo_features(cat, name)
    if me.clip is None:
        raise RuntimeError("This shoot has no CLIP embeddings yet. Run first:  ks prepare (or ks clip, ks groups)")
    train = []
    for db in others:
        with Catalog(db) as other:
            ds = photo_features(other, db.parent.name)
        if ds.clip is not None and (ds.y == 1).any():
            train.append(ds)
    if train and style == "studio":
        return studio_style(cat, name, me, train, keep_share, make_model, model_name)
    data = combine(train + [me], by_wedding=False)
    wedding = np.array([d.meta["wedding"] for d in train + [me] for _ in d.ids])
    events = data.group
    mine = wedding == name
    n_me = int(mine.sum())
    known = data.y >= 0
    adj = np.zeros(n_me)
    cut_units: list[tuple[np.ndarray, float]] = []        # (rows of this shoot, share to keep)
    my_events = [events[i] for i in np.flatnonzero(mine)]
    my_moment = data.moment[mine]

    if train:
        mode = "other weddings"
        rows = ~mine & known
        X, y = data.X[rows], data.y[rows]
        g_rows = [wedding[i] if len(train) > 1 else events[i] for i in np.flatnonzero(rows)]
        e_rows = [events[i] for i in np.flatnonzero(rows)]
        share = keep_share if keep_share is not None else keep_share_for(
            X, y, g_rows, e_rows, data.moment[rows], make_model, target)
        model = make_model().fit(X, y)
        raw = model.predict(data.X[mine])
        adj = with_variety(per_event_scale(raw, my_events), my_moment)
        cut_units.append((np.arange(n_me), share))
        trained_on, n_train, n_picks = [d.meta["wedding"] for d in train], int(rows.sum()), int(y.sum())
    else:
        mode = "other events"
        labelled_events = sorted({e for e, k in zip(my_events, known[mine]) if k})
        if len(labelled_events) < 2:
            raise RuntimeError("Nothing to learn from yet: KeepSake needs the studio's choices for a past "
                               "wedding (ks match), or for at least two events of this one.")
        y_me = data.y[mine]
        X_me = data.X[mine]
        trained_on, n_train, n_picks = [name], int(known[mine].sum()), int((y_me == 1).sum())
        ev = np.array(my_events)
        for e in sorted(set(my_events)):
            test = ev == e
            rows = ~test & (y_me >= 0)
            if y_me[rows].sum() == 0:
                continue
            g_rows = [my_events[i] for i in np.flatnonzero(rows)]
            share = keep_share if keep_share is not None else keep_share_for(
                X_me[rows], y_me[rows], g_rows, g_rows, my_moment[rows], make_model, target)
            model = make_model().fit(X_me[rows], y_me[rows])
            raw = model.predict(X_me[test])
            adj[test] = with_variety(per_event_scale(raw, [e] * int(test.sum())), my_moment[test])
            cut_units.append((np.flatnonzero(test), share))

    keep = np.zeros(n_me, bool)
    unsure = np.zeros(n_me, bool)
    for idx, share in cut_units:
        order = idx[np.argsort(-adj[idx], kind="mergesort")]
        n_keep = min(len(idx), math.ceil(share * len(idx)))
        n_unsure = min(len(idx) - n_keep, math.ceil(UNSURE_BAND * len(idx)))
        keep[order[:n_keep + n_unsure]] = True
        unsure[order[n_keep:n_keep + n_unsure]] = True
    covered = np.zeros(n_me, bool)
    for idx, _ in cut_units:
        covered[idx] = True
    keep[~covered] = True                      # an event that could not be scored: keep it all
    unsure[~covered] = True

    percentile = np.empty(n_me)
    percentile[np.argsort(adj, kind="mergesort")] = np.linspace(0, 100, n_me) if n_me > 1 else 100
    names = {pid: Path(p).name for pid, p in cat.db.execute("SELECT id, path FROM photos")}
    me_scaled = Dataset(me.ids, data.names, data.X[mine], data.y[mine], my_events, my_moment)
    why = explain(me_scaled, keep, adj, names)
    rows_out, reasons = [], Counter()
    for i, pid in enumerate(me.ids):
        if unsure[i]:
            reason = ("AI unsure: this event could not be scored - kept for you to check" if not covered[i]
                      else "AI unsure: close to the cut-off - kept for you to check")
            rows_out.append((pid, "selected", 1, round(float(percentile[i]), 1), reason, None))
        elif keep[i]:
            rows_out.append((pid, "selected", 0, round(float(percentile[i]), 1), None, None))
        else:
            reason, similar = why[i]
            reasons[reason_kind(reason)] += 1
            rows_out.append((pid, "removed", 0, round(float(percentile[i]), 1), reason, similar))
    unread = [pid for (pid,) in cat.db.execute("SELECT id FROM photos WHERE status != 'done' "
                                                "AND status != 'missing'")]
    for pid in unread:
        rows_out.append((pid, "selected", 1, None, "Could not be read by KeepSake - kept for you to check", None))
    result = CullResult(rows_out, mode, trained_on, n_train, n_picks,
                        float(np.mean([s for _, s in cut_units])) if cut_units else 1.0, model_name,
                        reasons, unread=len(unread))
    y_me = data.y[mine]
    if (y_me >= 0).any() and (y_me == 1).any():
        result.caught = (int(((y_me == 1) & keep).sum()), int((y_me == 1).sum()))
    return result
