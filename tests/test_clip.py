"""Tests for Phase 3 CLIP embeddings. The model is replaced by a stand-in, except in the last
test, which loads the real one where it is installed (the KeepSake laptop). Run: ks test"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from engine import clip as clip_mod
from engine.catalog import Catalog
from engine.clip import (CLIP_FILE, CLIP_VERSION, ClipEmbedder, clip_task, from_blob, preprocess,
                         to_blob)
from engine.config import MODELS_DIR


def test_preprocess_is_clips_own_recipe():
    img = np.zeros((1080, 1620, 3), np.uint8)
    img[:, :, 2] = 255                                         # pure red in BGR
    x = preprocess(img)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    red = (1 - 0.48145466) / 0.26862954
    assert x[0, 100, 100] == pytest.approx(red, abs=1e-4)     # channel 0 = red after BGR->RGB
    assert x[2, 100, 100] == pytest.approx((0 - 0.40821073) / 0.27577711, abs=1e-4)


def test_blobs_are_unit_float16():
    v = np.arange(1, 513, dtype=np.float32)
    b = to_blob(v)
    assert len(b) == 1024
    assert np.linalg.norm(from_blob(b)) == pytest.approx(1, abs=1e-3)


class FakeSession:
    """Stand-in for ONNX Runtime: the 'embedding' is the mean colour, padded to 512 numbers."""
    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state"), SimpleNamespace(name="image_embeds")]

    def run(self, names, feeds):
        assert names == ["image_embeds"]
        x = feeds["pixel_values"]
        out = np.zeros((len(x), 512), np.float32)
        out[:, :3] = x.mean(axis=(2, 3)) + 3
        return [out]


def test_task_embeds_a_batch_and_reports_bad_files(tmp_path, monkeypatch):
    paths = []
    for i, colour in enumerate([(0, 0, 255), (0, 0, 250), (255, 0, 0)]):
        p = tmp_path / f"{i}.jpg"
        cv2.imencode(".jpg", np.full((300, 450, 3), colour, np.uint8))[1].tofile(str(p))
        paths.append(p)
    monkeypatch.setattr(clip_mod, "_embedder", ClipEmbedder(session=FakeSession()))
    r = clip_task((1, [(1, str(paths[0])), (2, str(paths[1])), (3, str(paths[2])),
                       (4, str(tmp_path / "missing.jpg"))]))
    by_id = {item["id"]: item for item in r["items"]}
    assert r["id"] == 1 and by_id[4]["ok"] is False and all(by_id[i]["ok"] for i in (1, 2, 3))
    a, b, c = (from_blob(by_id[i]["emb"]) for i in (1, 2, 3))
    assert a @ b > a @ c                                       # two reds are closer than red/blue

    cat = Catalog(tmp_path / "c.db")
    cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) "
                       "VALUES (?, ?, '', 'jpeg', 1, 1, 'done')", [(i, f"{i}.jpg") for i in (1, 2, 3, 4)])
    assert cat.clip_pending(CLIP_VERSION) == [1, 2, 3, 4]
    cat.save_clip(r["items"], CLIP_VERSION)
    assert cat.clip_pending(CLIP_VERSION) == [4]               # the failed one is retried
    ids, matrix = cat.clip_vectors()
    assert ids == [1, 2, 3] and matrix.shape == (3, 512)
    cat.close()


@pytest.mark.skipif(not (MODELS_DIR / CLIP_FILE).exists(), reason="CLIP not downloaded (ks models)")
def test_real_clip_model():
    pytest.importorskip("onnxruntime")
    embedder = ClipEmbedder()
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (300, 450, 3), dtype=np.uint8)
    b = cv2.GaussianBlur(a, (0, 0), 1)                         # nearly the same picture
    c = np.full((300, 450, 3), (40, 160, 40), np.uint8)        # a different one
    va, vb, vc = embedder.embed(np.stack([preprocess(x) for x in (a, b, c)]))
    assert va.shape == (512,) and va @ vb > va @ vc
