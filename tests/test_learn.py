"""Tests for Phase 3 learning and honest testing. Run: ks test"""
import numpy as np
import pytest

from engine.catalog import Catalog
from engine.clip import to_blob
from engine.learn import GBM, Logistic, combine, leave_one_out, photo_features, score

RNG = np.random.default_rng(1)


def fake_shoot(tmp_path, name, n=120):
    """A catalog where the studio keeps the smiling photos with the couple in them."""
    cat = Catalog(tmp_path / f"{name}.db")
    for pid in range(1, n + 1):
        event = "Haldi" if pid <= n // 2 else "Sangeet"
        smile = float(RNG.random())
        couple = pid % 3 == 0
        keep = int(couple and smile > 0.4)
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, body, taken_s) "
                       "VALUES (?, ?, ?, 'jpeg', 1, 1, 'done', 'A', ?)", (pid, f"{pid}.jpg", event, pid * 5.0))
        cat.db.execute("INSERT INTO quality (id, version, ok, sharp_global, sharp_peak, sharp_rel, luma_mean) "
                       "VALUES (?, 1, 1, ?, ?, ?, 120)", (pid, 100 + RNG.random(), 200 + RNG.random(), RNG.normal()))
        cat.db.execute("INSERT INTO labels (id, selected) VALUES (?, ?)", (pid, keep))
        cat.db.execute("INSERT INTO moments (id, moment, size, pos) VALUES (?, ?, 1, 0)", (pid, pid))
        cat.db.execute("INSERT INTO faces (photo_id, n, x, y, w, h, score, rel_w, sharp, contrast, subject, read, "
                       "smile, yaw, eyes, person) VALUES (?, 0, 500, 300, 200, 200, 0.9, 0.12, 300, 30, 1, 1, ?, 0, "
                       "'open', ?)", (pid, smile, 1 if couple else 7))
        v = RNG.normal(size=512)
        cat.db.execute("INSERT INTO clip (id, version, ok, emb) VALUES (?, 1, 1, ?)", (pid, to_blob(v)))
    cat.db.commit()
    return cat


def test_features_from_a_catalog(tmp_path):
    cat = fake_shoot(tmp_path, "w1")
    ds = photo_features(cat, "w1")
    assert ds.X.shape[0] == 120 and len(ds.names) == ds.X.shape[1]
    assert {"smile_mean", "couple", "p1", "moment_size", "clip_typical_of_shoot",
            "smile_mean_rank_in_moment"} <= set(ds.names)
    assert ds.clip.shape == (120, 512) and set(ds.group) == {"w1 / Haldi", "w1 / Sangeet"}
    assert not np.isnan(ds.X).any()
    cat.close()


def test_honest_testing_learns_the_pattern(tmp_path):
    a, b = fake_shoot(tmp_path, "w1"), fake_shoot(tmp_path, "w2")
    data = combine([photo_features(a, "w1"), photo_features(b, "w2")], by_wedding=True)
    assert set(data.group) == {"w1", "w2"} and data.names[-1] == "clip_pc24"
    scores, _ = leave_one_out(data, Logistic)
    assert len(scores) == 2 and all(s.auc > 0.8 for s in scores)   # each wedding tested unseen
    a.close()
    b.close()


def test_score_numbers():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    perfect = score(np.array([9, 0, 8, 0, 0, 0, 0, 0, 0, 7.0]), y, "t")
    assert perfect.auc == 1.0 and perfect.caught[0.3] == 1.0 and perfect.needed[0.95] == pytest.approx(0.3)


def test_lightgbm_model(tmp_path):
    pytest.importorskip("lightgbm")
    a, b = fake_shoot(tmp_path, "w1", 300), fake_shoot(tmp_path, "w2", 300)
    data = combine([photo_features(a, "w1"), photo_features(b, "w2")], by_wedding=True)
    scores, model = leave_one_out(data, lambda: GBM(rounds=60))
    assert all(s.auc > 0.75 for s in scores)
    top = model.importance(data.names, top=3)[0][0]
    assert top.startswith(("smile", "couple", "p1", "top5", "top10"))     # what really decides
    a.close()
    b.close()
