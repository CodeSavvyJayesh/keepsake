"""Phase 3: CLIP image embeddings - what each photo shows, as 512 numbers.

CLIP (OpenAI, ViT-B/32) was trained on 400 million photos with captions, so photos with
similar content and look get similar embeddings: two frames of the same moment are nearly
identical, a couple portrait sits near other couple portraits, a decor shot near decor shots.
KeepSake uses the embeddings to (1) group near-duplicate photos across cameras, and (2) give
the learned model a sense of content and composition.

The model runs on this laptop with ONNX Runtime (int8-quantised export by Xenova/Hugging Face,
about 90 MB). No internet is needed once it is downloaded (ks models).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

CLIP_VERSION = 1
CLIP_FILE = "clip_vit_b32_vision_int8.onnx"
CLIP_SIZE = 224
CLIP_DIM = 512
BATCH = 16
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """CLIP's own preparation: shortest side to 224 (bicubic), centre 224 x 224, normalised.

    Returns a 3 x 224 x 224 float32 array (channels first, RGB).
    """
    h, w = bgr.shape[:2]
    scale = CLIP_SIZE / min(h, w)
    nw, nh = max(CLIP_SIZE, round(w * scale)), max(CLIP_SIZE, round(h * scale))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    x0, y0 = (nw - CLIP_SIZE) // 2, (nh - CLIP_SIZE) // 2
    img = img[y0:y0 + CLIP_SIZE, x0:x0 + CLIP_SIZE, ::-1].astype(np.float32) / 255.0
    return ((img - _MEAN) / _STD).transpose(2, 0, 1)


def to_blob(vector: np.ndarray) -> bytes:
    """Unit-length float16 bytes for the catalog (1 KB per photo)."""
    v = np.asarray(vector, np.float32).ravel()
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float16).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


class ClipEmbedder:
    """Wraps the ONNX model. One per worker process."""

    def __init__(self, model: Path = MODELS_DIR / CLIP_FILE, threads: int = 0, session=None):
        if session is None:
            import onnxruntime as ort
            if not model.exists():
                raise FileNotFoundError(f"{model.name} is missing. Run:  ks models")
            options = ort.SessionOptions()
            if threads:
                options.intra_op_num_threads = threads
            session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.session = session
        self.input = session.get_inputs()[0].name
        names = [o.name for o in session.get_outputs()]
        self.output = "image_embeds" if "image_embeds" in names else names[0]

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """N x 3 x 224 x 224 -> N x 512 unit vectors."""
        out = self.session.run([self.output], {self.input: batch.astype(np.float32)})[0]
        if out.ndim == 3:                         # a model without projection: take the CLS token
            out = out[:, 0, :]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


_embedder: ClipEmbedder | None = None


def clip_task(task: tuple) -> dict:
    """Worker job: (first photo id, [(photo id, analysis copy path), ...]) ->
    {'id', 'items': [{'id', 'ok', 'emb' | 'error'}]}. Never raises."""
    global _embedder
    first_id, items = task
    results, good, arrays = [], [], []
    for photo_id, path in items:
        try:
            bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"cannot decode {Path(path).name}")
            arrays.append(preprocess(bgr))
            good.append(photo_id)
        except Exception as e:  # one bad file: record it, carry on
            results.append({"id": photo_id, "ok": False, "error": f"{type(e).__name__}: {e}"})
    if good:
        try:
            if _embedder is None:
                import multiprocessing
                inside_worker = multiprocessing.current_process().name != "MainProcess"
                _embedder = ClipEmbedder(threads=2 if inside_worker else 0)
            vectors = _embedder.embed(np.stack(arrays))
            results += [{"id": pid, "ok": True, "emb": to_blob(v)} for pid, v in zip(good, vectors)]
        except Exception as e:  # model missing or broken: every photo of the batch fails
            results += [{"id": pid, "ok": False, "error": f"{type(e).__name__}: {e}"} for pid in good]
    return {"id": first_id, "items": results}
