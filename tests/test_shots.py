"""Tests for one photo per shot and the all-weddings blind test. Run: ks test"""
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.shots import frame_penalty, group_shots, one_per_shot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RNG = np.random.default_rng(5)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_near_identical_frames_form_one_shot_and_only_one_is_kept():
    a, b = unit(RNG.normal(size=(2, 512)))
    clip = np.array([a, unit(a + RNG.normal(0, 0.01, 512)), unit(a + RNG.normal(0, 0.01, 512)), b,
                     unit(a + RNG.normal(0, 0.01, 512))])
    moment = np.array([1, 1, 1, 1, 2])                    # the last one: same picture, another moment
    shot = group_shots(moment, clip)
    assert shot[0] == shot[1] == shot[2] and len({shot[0], shot[3], shot[4]}) == 3
    score = np.array([0.9, 0.8, 0.7, 0.5, 0.1])
    frame = np.array([0.0, 1.0, 0.5, 0.0, 0.0])            # frame 1 is the best frame of the first shot
    chosen, kept_in = one_per_shot(np.arange(5), score, frame, shot, n=2)
    assert list(chosen) == [1, 3] and kept_in[0] == 1 and kept_in[2] == 1


def test_closed_or_half_closed_eyes_and_turned_heads_are_penalised():
    names = ["eyes_open", "eyes_closed_any", "turned", "subjects"]
    X = np.array([[1.0, 0, 5, 2], [0.5, 0, 5, 2], [1.0, 1, 5, 2], [1.0, 0, 60, 2], [0.5, 0, 0, 0]])
    p = frame_penalty(X, names)
    assert p[0] == 0 and p[1] > 0 and p[2] > p[1] and p[3] > 0 and p[4] == 0   # no people: no eye penalty


def test_blind_test_on_all_weddings(tmp_path, monkeypatch):
    import blindtest as script
    from engine import ingest
    from test_cull import fake_wedding
    cache = tmp_path / "cache"
    for name in ("wedding_one", "wedding_two", "wedding_three"):
        (cache / name).mkdir(parents=True)
        cat = fake_wedding(cache / name / "catalog.db", n=60)
        for (pid,) in cat.db.execute("SELECT id FROM photos"):
            p = ingest.cache_path(cache / name, "analysis", pid)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", np.full((20, 30, 3), 120, np.uint8))[1].tofile(str(p))
        cat.close()
    monkeypatch.setattr(script, "CACHE_DIR", cache)
    monkeypatch.setattr(script, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["blindtest.py"])
    assert script.main() == 0
    run = next((tmp_path / "out").iterdir())
    one = run / "wedding_one"
    sel = next(one.glob("KeepSake selected (*)"))
    rem = next(one.glob("KeepSake removed (*)"))
    n_sel, n_rem = len(list(sel.iterdir())), len(list(rem.iterdir()))
    assert n_sel + n_rem == 60 and n_sel == int(sel.name.split("(")[1].rstrip(")"))
    assert "All weddings" in (run / "Summary.txt").read_text()
    studio = next(one.glob("Studio selected (*)"))
    not_studio = next(one.glob("Studio not selected (*)"))
    assert len(list(studio.iterdir())) + len(list(not_studio.iterdir())) == 60
    assert len(list(studio.iterdir())) == int(studio.name.split("(")[1].rstrip(")"))
    both = next(one.glob("Selected by both (*)"))
    assert {p.name for p in both.iterdir()} == {p.name for p in studio.iterdir()} & {p.name for p in sel.iterdir()}
