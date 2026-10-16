"""Tests for Phase 2 quality measurements. Run: ks test"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.quality import auc, measure_gray, relative_scores, robust_z

ROOT = Path(__file__).resolve().parents[1]


def scene(seed=0, size=(1080, 1620)):
    """A detailed test picture: random shapes and fine texture."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 120, np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))
        cv2.circle(img, (x, y), int(rng.integers(10, 120)), int(rng.integers(0, 255)), -1)
    img = cv2.add(img, rng.integers(0, 40, size, dtype=np.uint8))
    return img


def test_blur_and_shake_lower_sharpness():
    sharp = measure_gray(scene())
    blurred = measure_gray(cv2.GaussianBlur(scene(), (0, 0), 3))
    shaken = measure_gray(cv2.filter2D(scene(), -1, np.ones((1, 25), np.float32) / 25))
    assert blurred["sharp_peak"] < 0.2 * sharp["sharp_peak"]
    assert blurred["sharp_global"] < 0.2 * sharp["sharp_global"]
    assert shaken["sharp_peak"] < 0.5 * sharp["sharp_peak"]
    assert shaken["motion"] > sharp["motion"] + 0.2           # smeared one way


def test_soft_background_with_sharp_subject_still_counts_as_sharp():
    img = cv2.GaussianBlur(scene(1), (0, 0), 4)              # background out of focus
    img[300:700, 600:1000] = scene(2)[300:700, 600:1000]      # subject in focus
    whole_soft = measure_gray(cv2.GaussianBlur(scene(1), (0, 0), 4))
    m = measure_gray(img)
    assert m["sharp_peak"] > 5 * whole_soft["sharp_peak"]


def test_exposure_numbers():
    dark = measure_gray(np.full((400, 600), 3, np.uint8))
    bright = measure_gray(np.full((400, 600), 254, np.uint8))
    assert dark["luma_mean"] == pytest.approx(3) and dark["clip_low"] == 1.0
    assert bright["clip_high"] == 1.0 and bright["clip_low"] == 0.0


def test_noise_estimate_follows_grain():
    rng = np.random.default_rng(0)
    flat = np.full((600, 900), 128, np.float32)
    clean = measure_gray(flat.astype(np.uint8))["noise"]
    noisy = measure_gray(np.clip(flat + rng.normal(0, 8, flat.shape), 0, 255).astype(np.uint8))["noise"]
    assert clean < 0.5 and 6 < noisy < 10                     # Immerkaer recovers sigma = 8


def test_relative_scores_compare_like_with_like():
    rows = [{"id": i, "body": "A", "event": "Haldi", "sharp_peak": 100.0 + i, "luma_mean": 120}
            for i in range(40)]
    rows += [{"id": 100 + i, "body": "B", "event": "Haldi", "sharp_peak": 1000.0 + 10 * i,
              "luma_mean": 120} for i in range(40)]
    rows.append({"id": 999, "body": "A", "event": "Haldi", "sharp_peak": 10.0, "luma_mean": 120})
    rel = relative_scores(rows)
    assert rel[999]["sharp_rel"] < -5                        # far softer than camera A usually is
    assert abs(rel[120]["sharp_rel"]) < 1                    # normal for camera B
    assert robust_z(np.array([1.0, 1.0, 1.0])).tolist() == [0, 0, 0]


def test_auc():
    labels = np.array([0, 0, 1, 1])
    assert auc(np.array([1, 2, 3, 4.0]), labels) == 1.0
    assert auc(np.array([4, 3, 2, 1.0]), labels) == 0.0
    assert auc(np.array([1, 1, 1, 1.0]), labels) == 0.5
    assert math.isnan(auc(np.array([1, 2.0]), np.array([1, 1])))


def test_model_download_is_checked_and_atomic(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import models
    src = tmp_path / "model.bin"
    src.write_bytes(b"x" * 5000)
    target = tmp_path / "models" / "model.bin"
    sha = models.download(src.as_uri(), target, 1000)
    assert target.read_bytes() == src.read_bytes() and len(sha) == 64
    with pytest.raises(OSError):
        models.download(src.as_uri(), tmp_path / "models" / "small.bin", 10_000)
    assert not (tmp_path / "models" / "small.bin").exists()
    assert not (tmp_path / "models" / "small.bin.part").exists()
