r"""Check that this KeepSake can run every step: each library loads, each AI model is the exact
file the studio's past weddings were measured with and opens, ExifTool answers, the learning
model trains, and the data folder can be written. For the packaged app and any new computer:

    ks selfcheck                      (from the code)
    KeepSake --run selfcheck          (the packaged app)

Ends with code 0 when everything is OK, 1 otherwise.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import config  # noqa: E402

RESULTS: list[bool] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append(True)
        print(f"  OK    {label:<34} {detail or ''}", flush=True)
    except Exception as e:  # noqa: BLE001 - report every check, keep going
        RESULTS.append(False)
        print(f"  FAIL  {label:<34} {type(e).__name__}: {e}", flush=True)


def model(name: str) -> Path:
    from scripts.models import SHA256, file_sha256
    path = config.MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    if file_sha256(path) != SHA256[name]:
        raise ValueError(f"{name} is not the file KeepSake's past weddings were measured with")
    return path


def rawpy_check() -> str:
    import rawpy
    return f"{rawpy.__version__} (LibRaw {'.'.join(map(str, rawpy.libraw_version))})"


def heif_check() -> str:
    import pillow_heif
    pillow_heif.register_heif_opener()
    return pillow_heif.__version__


def exiftool_check() -> str:
    exe = config.exiftool_path()
    out = subprocess.run([exe, "-ver"], capture_output=True, text=True, timeout=60, check=True)
    return f"{out.stdout.strip()} ({exe})"


def yunet_check() -> str:
    import cv2
    cv2.FaceDetectorYN.create(str(model("face_detection_yunet_2023mar.onnx")), "", (320, 320))
    return f"OpenCV {getattr(cv2, '__version__', '?')}"


def sface_check() -> str:
    import cv2
    cv2.FaceRecognizerSF.create(str(model("face_recognition_sface_2021dec.onnx")), "")
    return "same file as the laptop"


def landmarker_check() -> str:
    import mediapipe
    from mediapipe.tasks.python import BaseOptions, vision
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model("face_landmarker.task"))))
    vision.FaceLandmarker.create_from_options(options).close()
    return f"MediaPipe {mediapipe.__version__}"


def clip_check() -> str:
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(model("clip_vit_b32_vision_int8.onnx")), providers=["CPUExecutionProvider"])
    first = session.get_inputs()[0]
    out = session.run(None, {first.name: np.zeros((1, 3, 224, 224), np.float32)})
    return f"ONNX Runtime {ort.__version__}, output {tuple(out[0].shape)}"


def lightgbm_check() -> str:
    import lightgbm
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)
    booster = lightgbm.train({"objective": "binary", "verbose": -1, "num_threads": 2},
                             lightgbm.Dataset(x, y), num_boost_round=20)
    accuracy = float(((booster.predict(x) > 0.5) == y).mean())
    return f"{lightgbm.__version__}, trained (accuracy {accuracy:.2f})"


def data_check() -> str:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    probe = config.DATA_DIR / "selfcheck.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return str(config.DATA_DIR)


def main() -> int:
    print(f"KeepSake self-check ({'packaged app' if config.FROZEN else 'from the code'}, {sys.platform})")
    check("RAW previews (rawpy)", rawpy_check)
    check("HEIC photos (pillow-heif)", heif_check)
    check("ExifTool", exiftool_check)
    check("Faces: YuNet", yunet_check)
    check("Faces: SFace fingerprints", sface_check)
    check("Eyes and smiles: Face Landmarker", landmarker_check)
    check("Understanding: CLIP", clip_check)
    check("Learning: LightGBM", lightgbm_check)
    check("Catalog: SQLite", lambda: sqlite3.sqlite_version)
    check("Data folder", data_check)
    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed} of {len(RESULTS)} OK" + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
