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
from engine.faces import (FACES_VERSION, FaceReader, blendshape_scores, faces_task, head_pose,
                          nearest_to_centre, region_sharpness, square_crop)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_crop_is_centred_on_the_face_even_at_the_edge():
    img = np.zeros((400, 600, 3), np.uint8)
    img[0:40, 0:40] = (0, 0, 255)                         # a "face" in the top-left corner
    crop = square_crop(img, (0, 0, 40, 40), margin=2.0, size=100)
    assert crop.shape == (100, 100, 3)
    assert tuple(crop[50, 50]) == (0, 0, 255)             # still in the middle


def test_nearest_face_to_the_centre_is_chosen():
    pt = lambda x, y: SimpleNamespace(x=x, y=y)
    neighbour = [pt(0.05, 0.1), pt(0.15, 0.2)]
    ours = [pt(0.45, 0.5), pt(0.55, 0.5)]
    assert nearest_to_centre([neighbour, ours]) == 1
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


def test_reader_reads_big_faces_and_skips_tiny_ones():
    reader = FaceReader(detector=FakeDetector([face_row(100, 100, 30), face_row(400, 300, 200)]),
                        landmarker=FakeLandmarker(), to_image=lambda rgb: rgb)
    found = reader.read(np.zeros((1080, 1620, 3), np.uint8))
    assert [f["w"] for f in found] == [200, 30]                      # largest first
    big, tiny = found
    assert big["read"] == 1 and big["blink_l"] == 0.95 and big["yaw"] == pytest.approx(0)
    assert tiny["read"] == 0 and tiny["blink_l"] is None              # too small to judge eyes
    assert big["rel_w"] == pytest.approx(200 / 1620)


def test_without_mediapipe_faces_are_still_found():
    reader = FaceReader(detector=FakeDetector([face_row(400, 300, 200)]), landmarker=None)
    reader.landmarker = None
    assert not reader.reads_eyes
    (face,) = reader.read(np.zeros((1080, 1620, 3), np.uint8))
    assert face["read"] == 0 and face["w"] == 200


def test_task_and_catalog_round_trip(tmp_path, monkeypatch):
    photo = tmp_path / "000001.jpg"
    cv2.imencode(".jpg", np.zeros((1080, 1620, 3), np.uint8))[1].tofile(str(photo))
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
    row = cat.db.execute("SELECT read, blink_l, w FROM faces WHERE photo_id = 1").fetchone()
    assert row == (1, 0.95, 200.0)
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
    (face,) = reader.read(np.full((1080, 1620, 3), 128, np.uint8))   # a grey "face": no reading
    assert face["read"] == 0
    reader.close()
