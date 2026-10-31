import csv
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
    rem = next(one.glob("KeepSake not selected (*)"))
    n_sel, n_rem = len(list(sel.iterdir())), len(list(rem.iterdir()))
    assert n_sel + n_rem == 60 and n_sel == int(sel.name.split("(")[1].rstrip(")"))
    assert "All weddings" in (run / "Summary.txt").read_text()
    studio = next(one.glob("Studio selected (*)"))
    not_studio = next(one.glob("Studio not selected (*)"))
    assert len(list(studio.iterdir())) + len(list(not_studio.iterdir())) == 60
    assert len(list(studio.iterdir())) == int(studio.name.split("(")[1].rstrip(")"))
    both = next(one.glob("Selected by both (*)"))
    assert {p.name for p in both.iterdir()} == {p.name for p in studio.iterdir()} & {p.name for p in sel.iterdir()}


def test_blind_test_of_the_shortlist_in_two_tiers(tmp_path, monkeypatch):
    import blindtest as script
    from test_cull import fake_wedding
    cache = tmp_path / "cache"
    for name in ("wedding_one", "wedding_two", "wedding_three"):
        (cache / name).mkdir(parents=True)
        fake_wedding(cache / name / "catalog.db", n=60).close()
    monkeypatch.setattr(script, "CACHE_DIR", cache)
    monkeypatch.setattr(script, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["blindtest.py", "--shortlist", "--numbers"])
    assert script.main() == 0
    run = next((tmp_path / "out").iterdir())
    text = (run / "Summary.txt").read_text()
    assert "Selected + Maybe" in text and "near-identical frame" in text
    assert not list((run / "wedding_one").glob("KeepSake selected (*)"))            # numbers only: no copies
    rows = list(csv.DictReader(open(run / "wedding_one" / "KeepSake choices.csv", encoding="utf-8")))
    kinds = {r["keepsake"] for r in rows}
    assert kinds <= {"selected", "maybe", "not selected"} and "selected" in kinds and len(rows) == 60


def test_selected_in_two_tiers_best_first_then_maybe_and_never_too_many():
    from engine.shots import shortlist, tiers
    rng = np.random.default_rng(3)
    n = 300
    shot = np.repeat(np.arange(150), 2)                       # 150 shots of 2 frames
    score = rng.normal(size=n)
    frame = rng.normal(size=n)
    rows = np.arange(n)
    best, chosen, kept_in = tiers(rows, score, score, frame, shot, best_n=30, target=0.99, max_share=0.4,
                                  best_by="studio")
    assert len(best) == 30 and list(chosen[:30]) == list(best)                     # Best first
    assert len(chosen) == int(0.4 * n)                                               # capped at 40% of the photos
    assert len({shot[i] for i in chosen}) == len(chosen)                             # one per shot
    assert all(frame[i] >= frame[j] for i in chosen for j in rows if shot[j] == shot[i])   # each shot's best frame
    top = sorted(set(shot), key=lambda k: -score[shot == k].max())[:30]             # Best = the 30 best shots
    assert {shot[i] for i in best} == set(top)
    small, _, _ = tiers(rows, score, score, frame, shot, best_n=30, target=0.2, best_by="studio")
    assert len(small) == 30                                                          # Best is kept whatever the target
    # the default: Selected = the Shortlist's first moments (most picks expected), then Maybe
    best, chosen, _ = tiers(rows, score, score, frame, shot, best_n=30, target=0.99, max_share=0.4)
    plain, _ = shortlist(rows, score, frame, shot, 0.99)
    assert list(best) == list(plain[:30]) and list(chosen) == list(plain[:120])
    small, _, _ = tiers(rows, score, score, frame, shot, best_n=30, target=0.05)
    assert len(small) == 30


def test_each_main_face_is_measured_for_the_frame_choice(tmp_path):
    from engine.catalog import Catalog
    from engine.shots import FACE_DETAILS, face_details
    cat = Catalog(tmp_path / "c.db")
    for pid in (1, 2, 3):
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) "
                       "VALUES (?, ?, '', 'jpeg', 1, 1, 'done')", (pid, f"IMG_{pid}.JPG"))
    faces = [(1, 0, 0.1, 300, 0.05, 0.07, 0.4, 0.1, 0.8, 0.0, 5, 3, "smiling"),     # open, smiling
             (1, 1, 0.08, 200, 0.45, 0.40, 0.3, 0.1, 0.2, 0.1, 40, 2, "unsure"),    # half blink, turned
             (2, 0, 0.1, 250, 0.9, 0.85, 0.2, 0.1, 0.1, 0.0, 2, 1, "closed")]       # blink
    cat.db.executemany("INSERT INTO faces (photo_id, n, rel_w, sharp, blink_l, blink_r, squint, look_down, smile, "
                       "jaw_open, yaw, pitch, eyes, subject, read) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,1)", faces)
    cat.db.commit()
    F = face_details(cat, [1, 2, 3])
    c = {n: i for i, n in enumerate(FACE_DETAILS)}
    assert F[0, c["n_read"]] == 2 and F[0, c["halfblink_n"]] == 1 and F[0, c["yaw_max"]] == 40
    assert F[1, c["closed_n"]] == 1 and F[1, c["blink_worst"]] > 0.8
    assert not F[2].any()                                                        # no faces: zeros


def test_full_clip_content_learns_what_the_studio_keeps():
    from engine.shots import centred_clip, clip_ridge
    rng = np.random.default_rng(2)
    liked = unit(rng.normal(size=512))
    C = unit(rng.normal(size=(600, 512)))
    y = (rng.random(600) < 0.1).astype(int)
    C[y == 1] = unit(C[y == 1] + 0.5 * liked)
    Cc = centred_clip(C, ["a"] * 300 + ["b"] * 300)
    w = clip_ridge(Cc[:400], y[:400])
    s = Cc[400:] @ w
    assert s[y[400:] == 1].mean() > s[y[400:] == 0].mean() + 2 * s.std() / 3


def test_faces_that_break_the_editors_rule(tmp_path):
    from engine.catalog import Catalog
    from engine.shots import EYES_HALF, EYES_SHUT, PROFILE, face_faults, fault_words
    with Catalog(tmp_path / "catalog.db") as cat:
        with cat.db:
            cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) "
                               "VALUES (?, ?, '', 'jpeg', 1, 0, 'done')", [(i, f"{i}.jpg") for i in range(1, 8)])
            faces = [  # photo, blink left, blink right, yaw, smile, squint, subject, read
                (1, 0.05, 0.08, 5, 0.2, 0.1, 1, 1),      # open: fine
                (2, 0.62, 0.58, 3, 0.1, 0.1, 1, 1),      # both shut
                (3, 0.70, 0.40, 0, 0.0, 0.1, 1, 1),      # half closed
                (4, 0.75, 0.70, 0, 0.8, 0.6, 1, 1),      # laughing: fine
                (5, 0.05, 0.05, 62, 0.0, 0.0, 1, 1),     # profile
                (6, 0.90, 0.90, 70, 0.0, 0.0, 0, 1),     # a guest in the background: not counted
                (7, 0.90, 0.90, 0, 0.0, 0.0, 1, 0)]      # not read: not counted
            cat.db.executemany("INSERT INTO faces (photo_id, n, blink_l, blink_r, yaw, smile, squint, subject, read) "
                               "VALUES (?, 0, ?, ?, ?, ?, ?, ?, ?)", faces)
        flags = face_faults(cat, [1, 2, 3, 4, 5, 6, 7])
    assert list(flags) == [0, EYES_SHUT, EYES_HALF, 0, PROFILE, 0, 0]
    assert "closed" in fault_words(EYES_SHUT) and "side" in fault_words(PROFILE)


def test_shortlist_keeps_the_likeliest_moments_until_the_target():
    from engine.shots import shortlist
    shot = np.array([0, 0, 0, 1, 2, 3, 4, 4])                       # moment 0 is a burst of 3 frames
    score = np.array([2.0, 1.5, 1.0, 2.5, -1.0, -2.0, 0.5, 0.0])
    frame = np.array([0.1, 0.9, 0.2, 0.0, 0.0, 0.0, 0.3, 0.1])      # the best frame of moment 0 is row 1
    rows = np.arange(8)
    chosen, kept_in = shortlist(rows, score, frame, shot, target=0.5)
    assert set(chosen.tolist()) <= {1, 3}                             # the likeliest moments first
    assert 1 in chosen.tolist() and kept_in[0] == 1 and kept_in[2] == 1   # the burst is kept by its best frame
    everything, _ = shortlist(rows, score, frame, shot, target=1.0)
    assert len(everything) == 5                                       # every moment, one frame each
