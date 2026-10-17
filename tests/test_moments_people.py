"""Tests for Phase 3 moments (clock alignment, grouping) and people. Run: ks test"""
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.catalog import Catalog
from engine.clip import to_blob
from engine.moments import Photo, align_clocks, densest, group_moments, group_quality
from engine.people import FaceRef, cluster_people

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
RNG = np.random.default_rng(0)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def near(v, noise=0.05):
    return unit(v + RNG.normal(0, noise, v.shape))


def test_densest_run():
    value, n = densest(np.array([5.0, 3600, 3610, 3590, 3605, -200]), 60)
    assert n == 4 and 3590 <= value <= 3610


def test_camera_clocks_are_lined_up_from_matching_photos():
    scenes = unit(RNG.normal(size=(100, 512)))
    photos, vecs = [], []
    for i in range(100):                                   # main camera
        photos.append(Photo(i, "A", 1000.0 + 30 * i, "Haldi"))
        vecs.append(scenes[i])
    for i in range(0, 100, 3):                             # second camera, clock 1 hour behind
        photos.append(Photo(1000 + i, "B", 1000.0 + 30 * i - 3600, "Haldi"))
        vecs.append(near(scenes[i], 0.01))
    for i in range(20):                                    # drone: unrelated pictures
        photos.append(Photo(2000 + i, "C", 5.0 * i, "Haldi"))
        vecs.append(unit(RNG.normal(size=512)))
    a = align_clocks(photos, np.array(vecs))
    assert a.reference == "A" and abs(a.offsets["B"] - 3600) < 5
    assert "C" not in a.offsets and a.support["C"] == 0


def test_moments_join_bursts_and_cameras_but_not_new_scenes():
    s1, s2 = unit(RNG.normal(size=(2, 512)))
    photos = [Photo(1, "A", 0.0, "E"), Photo(2, "A", 0.5, "E"), Photo(3, "A", 1.0, "E"),
              Photo(4, "B", 3601.5, "E"),                   # other camera, clock 1 h fast
              Photo(5, "A", 10.0, "E"),                     # new scene 9 s later
              Photo(6, "A", 0.7, "Other event")]            # different event folder
    vecs = np.array([s1, near(s1, 0.01), near(s1, 0.01), near(s1, 0.01), s2, near(s1, 0.01)])
    a = align_clocks(photos, vecs)
    a.offsets["B"] = -3600.0                                # (too few pairs to measure here)
    groups = group_moments(photos, vecs, a)
    assert sorted(map(sorted, groups)) == [[1, 2, 3, 4], [5], [6]]
    q = group_quality(groups, {1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0})
    assert q["groups"] == 3 and q["multi_groups"] == 1 and q["picks_if_one_per_group"] == 2


def test_runs_split_on_pause_scene_change_and_length():
    from engine.moments import camera_runs
    s1, s2 = unit(RNG.normal(size=(2, 512)))
    times = [0, 1, 2, 9, 10, 11] + [20 + 2 * i for i in range(40)]      # pause at 2 -> 9; then 80 s of one scene
    vecs = [near(s1, 0.01)] * 3 + [near(s2, 0.01)] * 3 + [near(s1, 0.01)] * 40
    photos = [Photo(i, "A", float(t), "E") for i, t in enumerate(times)]
    runs = camera_runs(photos, np.array(vecs))
    sizes = sorted(len(r.members) for r in runs)
    assert sizes[:2] == [3, 3] and sum(sizes) == 46 and len(runs) == 4   # 60 s cap splits the long one


def test_side_faces_do_not_start_people_but_join_them():
    ids = unit(RNG.normal(size=(2, 128)))
    faces = [FaceRef(i, 0, 120.0, near(ids[0], 0.03)) for i in range(1, 21)]
    faces += [FaceRef(100 + i, 0, 120.0, near(ids[1], 0.03), frontal=0.8) for i in range(20)]   # only side views
    faces.append(FaceRef(500, 0, 120.0, near(ids[0], 0.03), frontal=0.8))
    found = cluster_people(faces)
    assert {found[(i, 0)] for i in range(1, 21)} == {1} and found[(500, 0)] == 1
    assert not any((100 + i, 0) in found for i in range(20))


def test_people_found_and_ranked_by_photos():
    ids = unit(RNG.normal(size=(3, 128)))
    faces, n_photos = [], {0: 40, 1: 25, 2: 5}            # bride, groom, a guest
    pid = 0
    for person, count in n_photos.items():
        for _ in range(count):
            pid += 1
            faces.append(FaceRef(pid, 0, 120.0, near(ids[person], 0.03)))
    faces.append(FaceRef(999, 0, 45.0, near(ids[1], 0.03)))   # small face of the groom
    found = cluster_people(faces)
    rank_of = lambda person: {found[(f.photo_id, 0)] for f in faces[:len(faces) - 1]
                              if np.argmax(ids @ f.emb) == person}
    assert rank_of(0) == {1} and rank_of(1) == {2} and rank_of(2) == {3}
    assert found[(999, 0)] == 2


def test_groups_command_end_to_end(tmp_path, monkeypatch):
    import groups as script
    from engine import ingest
    root = tmp_path / "Sharma Wedding" / "Raw"
    shoot = ingest.shoot_for(root, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(script, "shoot_for", lambda r: shoot)
    monkeypatch.setattr(script, "attach_labels", lambda cat, r: None)
    cat = Catalog(shoot.db)
    scene = unit(RNG.normal(size=(6, 512)))
    face_id = unit(RNG.normal(size=(2, 128)))
    for pid in range(1, 31):
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, body, taken_s) "
                       "VALUES (?, ?, 'Haldi', 'jpeg', 1, 1, 'done', 'A', ?)", (pid, f"{pid}.jpg", pid * 0.5))
        cat.db.execute("INSERT INTO clip (id, version, ok, emb) VALUES (?, 1, 1, ?)",
                       (pid, to_blob(near(scene[(pid - 1) // 5], 0.01))))
        cat.db.execute("INSERT INTO labels (id, selected) VALUES (?, ?)", (pid, int(pid % 5 == 0)))
        points = np.array([[80, 90], [120, 90], [100, 110], [85, 130], [115, 130]], np.float32)
        cat.db.execute("INSERT INTO faces (photo_id, n, x, y, w, h, sharp, contrast, lm, emb) "
                       "VALUES (?, 0, 50, 50, 100, 100, 50, 10, ?, ?)",
                       (pid, points.tobytes(), to_blob(near(face_id[pid % 2], 0.02))))
        for kind in ("analysis", "thumbs"):
            path = ingest.cache_path(shoot.cache, kind, pid)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", np.full((200, 300, 3), 120, np.uint8))[1].tofile(str(path))
    cat.db.commit()
    cat.close()
    monkeypatch.setattr(sys, "argv", ["groups.py", str(root)])
    assert script.main() == 0
    cat = Catalog(shoot.db)
    assert cat.db.execute("SELECT COUNT(DISTINCT moment) FROM moments").fetchone()[0] == 6
    assert cat.db.execute("SELECT COUNT(DISTINCT person) FROM faces").fetchone()[0] == 2
    cat.close()
    assert (shoot.cache / "review" / "groups_review.html").exists()
