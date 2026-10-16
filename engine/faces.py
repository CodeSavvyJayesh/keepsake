"""Phase 2b: every face in a photo, with eyes, smile and head pose.

Two models, both run on this laptop:
  YuNet (OpenCV Zoo)          finds faces, down to about 10 px wide, so group photos work.
  Face Landmarker (MediaPipe)  reads each face: 52 expression scores ("blendshapes") such as
                               eyeBlinkLeft / eyeBlinkRight (0 = wide open, 1 = shut),
                               mouthSmile, and the head's turn and tilt.
The Face Landmarker is made for faces that fill the picture, so each face YuNet finds is cut
out with a margin, enlarged to 256 px, and read on its own.

Only measurements are made here. Whether closed eyes are a fault depends on context (a
blink in a group photo vs. a bride looking down at her mehndi), which is decided later.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

FACES_VERSION = 1
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
LANDMARKER_FILE = "face_landmarker.task"

MIN_SCORE = 0.6            # YuNet confidence for a face to count
MIN_EYE_PX = 36            # narrower faces (in the 1620 px copy): eyes too small to judge
CROP_MARGIN = 1.8          # face cut-out side = 1.8 x the face box
CROP_PX = 256              # cut-outs are enlarged to this size for the Face Landmarker

BLENDSHAPES = {            # catalog column: MediaPipe blendshape name(s), averaged
    "blink_l": ("eyeBlinkLeft",),
    "blink_r": ("eyeBlinkRight",),
    "squint": ("eyeSquintLeft", "eyeSquintRight"),
    "look_down": ("eyeLookDownLeft", "eyeLookDownRight"),
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "jaw_open": ("jawOpen",),
}
FACE_COLUMNS = ("x", "y", "w", "h", "score", "rel_w", "sharp", "contrast", "read",
                *BLENDSHAPES, "yaw", "pitch", "roll")


# --- Pure helpers (tested without the models) --------------------------------------

def square_crop(img: np.ndarray, box: tuple[float, float, float, float],
                margin: float = CROP_MARGIN, size: int = CROP_PX) -> np.ndarray:
    """A square cut-out centred on the face, `margin` x its size, enlarged to size x size.

    Parts outside the photo are filled by repeating the edge, so the face stays centred.
    """
    x, y, w, h = box
    side = max(w, h) * margin
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    H, W = img.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = img[y0:y1, x0:x1]
    interp = cv2.INTER_CUBIC if crop.shape[0] < size else cv2.INTER_AREA
    return cv2.resize(crop, (size, size), interpolation=interp)


def nearest_to_centre(landmark_sets: list) -> int | None:
    """Index of the face whose landmarks sit closest to the middle of the cut-out.

    A cut-out of one guest can include half of the next guest's face; the one we asked
    about is in the middle.
    """
    best, best_d = None, math.inf
    for i, points in enumerate(landmark_sets):
        if not points:
            continue
        mx = sum(p.x for p in points) / len(points)
        my = sum(p.y for p in points) / len(points)
        d = (mx - 0.5) ** 2 + (my - 0.5) ** 2
        if d < best_d:
            best, best_d = i, d
    return best


def blendshape_scores(categories) -> dict:
    """{'blink_l': 0.03, 'smile': 0.71, ...} from MediaPipe's list of Category(name, score)."""
    by_name = {c.category_name: float(c.score) for c in categories}
    out = {}
    for column, names in BLENDSHAPES.items():
        values = [by_name[n] for n in names if n in by_name]
        out[column] = sum(values) / len(values) if values else None
    return out


def head_pose(matrix) -> dict:
    """Turn (yaw), nod (pitch) and tilt (roll) in degrees from MediaPipe's 4x4 face matrix."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    angles = cv2.RQDecomp3x3(rotation)[0]          # (x, y, z) = (pitch, yaw, roll)
    return {"pitch": float(angles[0]), "yaw": float(angles[1]), "roll": float(angles[2])}


def region_sharpness(gray: np.ndarray, box) -> tuple[float, float]:
    """(Laplacian variance, grey-level spread) of the middle of the face: eyes, nose, mouth."""
    x, y, w, h = box
    x0, y0 = int(max(0, x + 0.2 * w)), int(max(0, y + 0.2 * h))
    x1, y1 = int(min(gray.shape[1], x + 0.8 * w)), int(min(gray.shape[0], y + 0.8 * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0, 0.0
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    return float(cv2.Laplacian(patch, cv2.CV_32F).var()), float(patch.std())


# --- The reader ---------------------------------------------------------------------

class FaceReader:
    """Finds and reads faces. Load once per worker process; reuse for every photo."""

    def __init__(self, models_dir: Path = MODELS_DIR, detector=None, landmarker=None,
                 to_image=None):
        yunet = models_dir / YUNET_FILE
        if detector is None:
            if not yunet.exists():
                raise FileNotFoundError(f"{yunet.name} is missing. Run:  ks models")
            detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), MIN_SCORE, 0.3, 5000)
        self.detector = detector
        self.landmarker = landmarker if landmarker is not None else _load_landmarker(models_dir)
        self.to_image = to_image or _mediapipe_image

    @property
    def reads_eyes(self) -> bool:
        return self.landmarker is not None

    def find(self, bgr: np.ndarray) -> np.ndarray:
        """YuNet rows: x, y, w, h, 5 landmark points (10 numbers), score."""
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        return faces if faces is not None else np.zeros((0, 15), np.float32)

    def read(self, bgr: np.ndarray) -> list[dict]:
        """Every face in a BGR photo, largest first."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        long_side = max(bgr.shape[:2])
        faces = []
        for row in self.find(bgr):
            x, y, w, h = (float(v) for v in row[:4])
            sharp, contrast = region_sharpness(gray, (x, y, w, h))
            face = {"x": x, "y": y, "w": w, "h": h, "score": float(row[14]),
                    "rel_w": w / long_side, "sharp": sharp, "contrast": contrast, "read": 0}
            face.update({c: None for c in (*BLENDSHAPES, "yaw", "pitch", "roll")})
            if self.landmarker is not None and w >= MIN_EYE_PX:
                face.update(self._expression(bgr, (x, y, w, h)))
            faces.append(face)
        faces.sort(key=lambda f: -f["w"])
        return faces

    def _expression(self, bgr: np.ndarray, box) -> dict:
        crop = square_crop(bgr, box)
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        result = self.landmarker.detect(self.to_image(rgb))
        i = nearest_to_centre(result.face_landmarks)
        if i is None or i >= len(result.face_blendshapes):
            return {}
        out = blendshape_scores(result.face_blendshapes[i])
        if i < len(result.facial_transformation_matrixes):
            out.update(head_pose(result.facial_transformation_matrixes[i]))
        out["read"] = 1
        return out

    def close(self) -> None:
        if self.landmarker is not None and hasattr(self.landmarker, "close"):
            self.landmarker.close()


def _mediapipe_image(rgb: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _load_landmarker(models_dir: Path):
    """MediaPipe's Face Landmarker, or None when MediaPipe or its model is not installed."""
    model = models_dir / LANDMARKER_FILE
    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        return None
    if not model.exists():
        return None
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,                          # the face we want, plus a neighbour at most
        min_face_detection_confidence=0.3,    # the cut-out is centred on a face YuNet found
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True)
    return vision.FaceLandmarker.create_from_options(options)


# --- Worker job ---------------------------------------------------------------------

_reader: FaceReader | None = None


def faces_task(task: tuple) -> dict:
    """(photo id, analysis copy path) -> {'id', 'ok', 'faces': [...], 'eyes': bool}. Never raises."""
    global _reader
    photo_id, path = task
    try:
        if _reader is None:
            _reader = FaceReader()
        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot decode {Path(path).name}")
        return {"id": photo_id, "ok": True, "faces": _reader.read(bgr), "eyes": _reader.reads_eyes}
    except Exception as e:  # one bad file or a model problem: record it, carry on
        return {"id": photo_id, "ok": False, "error": f"{type(e).__name__}: {e}", "faces": []}
