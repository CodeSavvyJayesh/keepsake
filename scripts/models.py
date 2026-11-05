"""Download the AI model files KeepSake uses into tools/models (not in Git).

    ks models            download any that are missing
    ks models --force    download them all again

YuNet (OpenCV Zoo, MIT licence): finds faces, including small ones in group photos.
Face Landmarker (Google MediaPipe, Apache 2.0): eyes, smile and head pose for each face.
SFace (OpenCV Zoo, Apache 2.0): a fingerprint per face, to find the same person again.
CLIP ViT-B/32 vision model (OpenAI, MIT; int8 ONNX export by Xenova): what a photo shows.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import MODELS_DIR  # noqa: E402

MODELS = {   # file name: (address, smallest believable size in bytes)
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx", 100_000),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/1/face_landmarker.task", 1_000_000),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx", 20_000_000),
    "clip_vit_b32_vision_int8.onnx": (
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/"
        "vision_model_quantized.onnx", 50_000_000),
}
# The exact files every studio's past weddings were measured with (SHA-256). A different file
# would measure new weddings differently from the past ones KeepSake learns from, so a download
# that does not match is refused rather than used.
SHA256 = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    "clip_vit_b32_vision_int8.onnx": "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, min_bytes: int) -> str:
    """Fetch url to target (via a temporary file, so a broken download never looks finished).
    Returns the file's SHA-256."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
        while chunk := response.read(1 << 16):
            out.write(chunk)
            digest.update(chunk)
    size = part.stat().st_size
    if size < min_bytes:
        part.unlink()
        raise OSError(f"only {size:,} bytes arrived (expected at least {min_bytes:,})")
    part.replace(target)
    return digest.hexdigest()


def check(name: str, path: Path) -> str:
    """Load the model once to prove it works here."""
    if name.startswith("face_detection"):
        import cv2
        cv2.FaceDetectorYN.create(str(path), "", (320, 320))
        return "loads in OpenCV"
    if name.startswith("face_recognition"):
        import cv2
        cv2.FaceRecognizerSF.create(str(path), "")
        return "loads in OpenCV"
    if name.startswith("clip"):
        try:
            import onnxruntime as ort
        except ImportError:
            return "downloaded (ONNX Runtime not installed yet, so not loaded)"
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outputs = [o.name for o in session.get_outputs()]
        return f"loads in ONNX Runtime (outputs: {', '.join(outputs)})"
    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        return "downloaded (MediaPipe not installed yet, so not loaded)"
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)))
    vision.FaceLandmarker.create_from_options(options).close()
    return "loads in MediaPipe"


def main() -> int:
    force = "--force" in sys.argv[1:]
    problems = 0
    for name, (url, min_bytes) in MODELS.items():
        path = MODELS_DIR / name
        try:
            if force or not path.exists():
                print(f"Downloading {name} ...", flush=True)
                sha = download(url, path, min_bytes)
                if sha != SHA256[name]:
                    path.replace(path.with_name(name + ".different"))
                    raise OSError(f"the file online has changed (SHA-256 {sha[:12]}..., expected "
                                  f"{SHA256[name][:12]}...): copy tools\\models from a KeepSake laptop instead")
                (MODELS_DIR / (name + ".sha256")).write_text(sha + "\n", encoding="utf-8")
            elif file_sha256(path) != SHA256[name]:
                raise OSError("this file differs from the one KeepSake's past weddings were measured with; "
                              "run  ks models --force  or copy tools\\models from a KeepSake laptop")
            print(f"  OK    {name:<36} {path.stat().st_size / 1e6:5.1f} MB  {check(name, path)}")
        except Exception as e:  # report each model, keep going
            problems += 1
            print(f"  FAIL  {name:<36} {type(e).__name__}: {e}")
    if problems:
        print("\nCheck the internet connection and run  ks models  again.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
