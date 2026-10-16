"""Tests for Phase 2b faces. The models are replaced by stand-ins, except in the last two tests,
which load the real ones where they are installed (the KeepSake laptop). Run: ks test"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from engine import faces as faces_mod
from engine.catalog import Catalog
from engine.config import MODELS_DIR
from engine.faces import (FACES_VERSION, FaceReader, blendshape_scores, eye_state, faces_task,
                          head_pose, nearest_to_centre, pick_subjects, region_sharpness, square_crop)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_crop_stays_inside_the_photo_and_says_where_the_face_is():
    img = np.full((400, 600, 3), 90, np.uint8)
    img[0:40, 0:40] = (0, 0, 255)                         # a "face" in the top-left corner
    crop, (fx, fy) = square_crop(img, (0, 0, 40, 40), margin=2.0, size=100)
    assert crop.shape == (100, 100, 3)
    assert (fx, fy) == (0.25, 0.25)                       # window slid in: face is top-left
    assert tuple(crop[25, 25]) == (0, 0, 255)
    assert not (crop == 0).all(axis=2).any()              # no black padding bars
    middle, centre = square_crop(img, (280, 180, 40, 40), margin=2.0, size=100)
    assert centre == (0.5, 0.5)


def test_nearest_face_to_the_centre_is_chosen():
    pt = lambda x, y: SimpleNamespace(x=x, y=y)
    neighbour = [pt(0.05, 0.1), pt(0.15, 0.2)]
    ours = [pt(0.45, 0.5), pt(0.55, 0.5)]
    assert nearest_to_centre([neighbour, ours]) == 1
    assert nearest_to_centre([neighbour, ours], centre=(0.1, 0.15)) == 0   # face near the edge
    assert nearest_to_centre([]) is None


def test_blendshapes_and_head_pose():
    cats = [SimpleNamespace(category_name=n, score=s) for n, s in
            [("eyeBlinkLeft", 0.9), ("eyeBlinkRight", 0.7), ("mouthSmileLeft", 0.4),
             ("mouthSmileRight", 0.6), ("jawOpen", 0.1)]]
    b = blendshape_scores(cats)
    assert b["blink_l"] == 0.9 and b["blink_r"] == 0.7 and b["smile"] == pytest.approx(0.5)
    assert b["squint"] is None                           # not given: unknown, not zero
    a = math.radians(30)
    turn = np.eye(4)
    turn[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    pose = head_pose(turn)
    assert pose["yaw"] == pytest.approx(30, abs=0.1) and abs(pose["pitch"]) < 0.1


def test_face_sharpness_drops_when_blurred():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (200, 200)).astype(np.uint8)
    sharp, _ = region_sharpness(gray, (50, 50, 100, 100))
    soft, _ = region_sharpness(cv2.GaussianBlur(gray, (0, 0), 3), (50, 50, 100, 100))
    assert soft < 0.1 * sharp


class FakeDetector:
    def __init__(self, rows):
        self.rows = np.array(rows, np.float32).reshape(-1, 15)

    def setInputSize(self, size):
        self.size = size

    def detect(self, img):
        return 1, (self.rows if len(self.rows) else None)


class FakeLandmarker:
    """Says every face it is shown has its eyes shut and is smiling."""
    def detect(self, image):
        cats = [SimpleNamespace(category_name=n, score=s) for n, s in
                [("eyeBlinkLeft", 0.95), ("eyeBlinkRight", 0.85), ("mouthSmileLeft", 0.7),
                 ("mouthSmileRight", 0.7)]]
        pts = [SimpleNamespace(x=0.5, y=0.5)]
        return SimpleNamespace(face_landmarks=[pts], face_blendshapes=[cats],
                               facial_transformation_matrixes=[np.eye(4)])


def face_row(x, y, w, score=0.9):
    return [x, y, w, w] + [0] * 10 + [score]


def textured(shape=(700, 1000, 3), seed=0):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


def test_reader_reads_main_faces_only():
    img = textured()
    img[50:130, 700:780] = cv2.GaussianBlur(img[50:130, 700:780], (0, 0), 6)   # a soft background face
    reader = FaceReader(detector=FakeDetector([face_row(100, 100, 30), face_row(400, 300, 200),
                                               face_row(700, 50, 80)]),
                        landmarker=FakeLandmarker(), to_image=lambda rgb: rgb)
    found = reader.read(img)
    assert [f["w"] for f in found] == [200, 80, 30]                  # largest first
    big, soft, tiny = found
    assert big["subject"] == 1 and big["read"] == 1 and big["blink_l"] == 0.95
    assert big["eyes"] == "closed" and big["yaw"] == pytest.approx(0)
    assert soft["subject"] == 0 and soft["read"] == 0                # out of focus: background
    assert tiny["subject"] == 0 and tiny["eyes"] is None             # too small to matter
    assert big["rel_w"] == pytest.approx(200 / 1000)


def test_faces_found_in_a_smaller_copy_come_back_in_full_size_pixels():
    reader = FaceReader(detector=FakeDetector([face_row(100, 100, 50)]), landmarker=None,
                        detect_px=500)
    reader.landmarker = None
    (face,) = reader.read(textured((700, 1000, 3)))                  # detected at half size
    assert (face["x"], face["w"]) == (200, 100)
    assert reader.detector.size == (500, 350)


def test_eye_verdicts():
    face = lambda **k: {"read": 1, "yaw": 0, "pitch": 0, "look_down": 0, "smile": 0, "squint": 0,
                        "blink_l": 0, "blink_r": 0, **k}
    assert eye_state(face(blink_l=0.9, blink_r=0.8)) == "closed"
    assert eye_state(face(blink_l=0.9, blink_r=0.2)) == "unsure"      # one eye only: hidden or a wink
    assert eye_state(face(blink_l=0.2, blink_r=0.3)) == "open"
    assert eye_state(face(blink_l=0.9, blink_r=0.9, look_down=0.4)) == "down"   # looking down
    assert eye_state(face(blink_l=0.9, blink_r=0.9, pitch=20)) == "down"        # head tipped down
    assert eye_state(face(blink_l=0.7, blink_r=0.7, smile=0.8, squint=0.6)) == "smiling"
    assert eye_state(face(blink_l=0.9, blink_r=0.9, yaw=40)) == "side"
    assert eye_state({"read": 0}) is None


def test_eye_verdicts_are_refreshed_from_stored_scores(tmp_path):
    cat = Catalog(tmp_path / "c.db")
    cat.db.execute("INSERT INTO faces (photo_id, n, read, yaw, pitch, look_down, blink_l, blink_r, smile, squint, eyes) "
                   "VALUES (1, 0, 1, 0, 25, 0, 0.9, 0.9, 0, 0, 'closed')")      # old rule said closed
    assert cat.refresh_eyes(eye_state) == 1
    assert cat.db.execute("SELECT eyes FROM faces").fetchone() == ("down",)
    assert cat.refresh_eyes(eye_state) == 0
    cat.close()


def test_subjects_are_large_confident_and_in_focus():
    f = lambda w, sharp, score=0.9: {"w": w, "rel_w": w / 1000, "sharp": sharp, "contrast": 20, "score": score}
    faces = [f(200, 500), f(150, 450), f(190, 50), f(60, 500), f(180, 500, score=0.65)]
    pick_subjects(faces)
    assert [x["subject"] for x in faces] == [1, 1, 0, 0, 0]


def test_without_mediapipe_faces_are_still_found():
    reader = FaceReader(detector=FakeDetector([face_row(400, 300, 200)]), landmarker=None)
    reader.landmarker = None
    assert not reader.reads_eyes
    (face,) = reader.read(textured())
    assert face["read"] == 0 and face["w"] == 200 and face["subject"] == 1 and face["eyes"] is None


def test_task_and_catalog_round_trip(tmp_path, monkeypatch):
    photo = tmp_path / "000001.jpg"
    cv2.imencode(".jpg", textured())[1].tofile(str(photo))
    monkeypatch.setattr(faces_mod, "_reader", FaceReader(
        detector=FakeDetector([face_row(400, 300, 200)]), landmarker=FakeLandmarker(),
        to_image=lambda rgb: rgb))
    r = faces_task((1, str(photo)))
    assert r["ok"] and len(r["faces"]) == 1 and r["eyes"]
    bad = faces_task((2, str(tmp_path / "missing.jpg")))
    assert not bad["ok"] and bad["faces"] == []

    cat = Catalog(tmp_path / "c.db")
    cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) "
                       "VALUES (?, ?, '', 'jpeg', 1, 1, 'done')", [(1, "a.jpg"), (2, "b.jpg")])
    assert cat.faces_pending(FACES_VERSION, True) == [1, 2]
    cat.save_faces([r, bad], FACES_VERSION)
    assert cat.faces_pending(FACES_VERSION, True) == [2]               # failed ones are retried
    row = cat.db.execute("SELECT read, blink_l, w, subject, eyes FROM faces WHERE photo_id = 1").fetchone()
    assert row == (1, 0.95, 200.0, 1, "closed")
    cat.close()


def test_burst_check_counts_the_pick_with_open_eyes():
    import faces as script
    s = lambda body, t, eyes, sel: {"body": body, "taken": t, "worst_eyes": eyes, "selected": sel}
    summaries = {1: s("A", 0.0, 0.9, 0), 2: s("A", 0.5, 0.1, 1), 3: s("A", 1.0, 0.8, 0),   # pick: open
                 4: s("A", 100, 0.1, 0), 5: s("A", 100.5, 0.7, 1)}                         # pick: closed
    (line,) = script.burst_lines(summaries)
    assert "2 bursts" in line and "most open eyes 50%" in line


@pytest.mark.skipif(not (MODELS_DIR / "face_detection_yunet_2023mar.onnx").exists(),
                    reason="face model not downloaded (ks models)")
def test_real_face_finder_loads_and_finds_nothing_in_a_blank_photo():
    reader = FaceReader(landmarker=None)
    reader.landmarker = None
    assert reader.read(np.full((1080, 1620, 3), 128, np.uint8)) == []


@pytest.mark.skipif(not (MODELS_DIR / "face_landmarker.task").exists(),
                    reason="eye model not downloaded (ks models)")
def test_real_eye_reader_loads_and_runs():
    pytest.importorskip("mediapipe")
    reader = FaceReader(detector=FakeDetector([face_row(400, 300, 200)]))
    assert reader.reads_eyes
    (face,) = reader.read(textured())                               # noise, not a face: no reading
    assert face["read"] == 0
    reader.close()


def test_old_catalogs_get_the_new_face_columns(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE faces (photo_id INTEGER, n INTEGER, x REAL, PRIMARY KEY (photo_id, n))")
    db.execute("INSERT INTO faces VALUES (1, 0, 5.0)")
    db.commit()
    db.close()
    cat = Catalog(tmp_path / "old.db")
    cols = {r[1] for r in cat.db.execute("PRAGMA table_info(faces)")}
    assert {"subject", "eyes"} <= cols
    assert cat.db.execute("SELECT x FROM faces").fetchone() == (5.0,)    # old data kept
    cat.close()


def test_eye_check_page(tmp_path, monkeypatch):
    import eyecheck
    from engine import ingest
    root = tmp_path / "Sharma Wedding" / "Raw"
    shoot = ingest.shoot_for(root, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(eyecheck, "shoot_for", lambda r: shoot)
    cat = Catalog(shoot.db)
    for pid in range(1, 7):
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) "
                       "VALUES (?, ?, '', 'jpeg', 1, 1, 'done')", (pid, f"{pid}.jpg"))
        path = ingest.cache_path(shoot.cache, "analysis", pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", textured((700, 1000, 3), pid))[1].tofile(str(path))
        cat.db.execute("INSERT INTO faces (photo_id, n, x, y, w, h, subject, eyes) VALUES (?, 0, 100, 100, 200, 200, 1, ?)",
                       (pid, "closed" if pid <= 2 else "open"))
    cat.db.commit()
    cat.close()
    monkeypatch.setattr(sys, "argv", ["eyecheck.py", str(root)])
    assert eyecheck.main() == 0
    page = (shoot.cache / "review" / "eye_check.html").read_text(encoding="utf-8")
    assert page.count('class="card"') == 6 and '"closed": 2' in page and '"open": 4' in page
    assert len(list((shoot.cache / "review" / "eyecheck").glob("*_face.jpg"))) == 6
