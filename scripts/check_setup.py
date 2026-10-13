r"""Phase 0: check every tool KeepSake needs and describe this laptop.

Run:  ks check
Saves the result to docs\phase0_check.txt.
"""
from __future__ import annotations

import importlib
import io
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import config  # noqa: E402

lines: list[str] = []
failures = 0


def report(status: str, name: str, detail: str = "") -> None:
    global failures
    if status == "FAIL":
        failures += 1
    line = f"{status:<4}  {name:<26} {detail}"
    lines.append(line)
    print(line)


def section(title: str) -> None:
    lines.append("")
    lines.append(title)
    print(f"\n{title}")


# --- Python ---------------------------------------------------------------
section("Python")
v = sys.version_info
report("OK" if (v.major, v.minor) == (3, 11) else "WARN", "Python",
       f"{platform.python_version()}  ({sys.executable})")
in_venv = sys.prefix != sys.base_prefix
report("OK" if in_venv else "WARN", "Virtual environment",
       "active" if in_venv else "not active: run scripts through ks.cmd")

# --- Libraries ------------------------------------------------------------
section("Libraries")
LIBS = [
    ("rawpy", "rawpy (RAW previews)"),
    ("exiftool", "PyExifTool"),
    ("PIL", "Pillow"),
    ("pillow_heif", "pillow-heif (HEIC)"),
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("psutil", "psutil"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
]
for module, label in LIBS:
    try:
        m = importlib.import_module(module)
        report("OK", label, str(getattr(m, "__version__", "")))
    except Exception as e:  # noqa: BLE001
        report("FAIL", label, f"{type(e).__name__}: {e}")

try:
    import rawpy
    lib = getattr(rawpy, "libraw_version", "")
    lib = ".".join(map(str, lib)) if isinstance(lib, tuple) else str(lib)
    report("OK", "LibRaw", lib)
except Exception as e:  # noqa: BLE001
    report("FAIL", "LibRaw", str(e))

try:
    from PIL import features
    turbo = features.check_feature("libjpeg_turbo")
    report("OK" if turbo else "WARN", "Fast JPEG (libjpeg-turbo)",
           "built into Pillow" if turbo else "missing: JPEG decoding will be slower")
except Exception as e:  # noqa: BLE001
    report("FAIL", "Fast JPEG (libjpeg-turbo)", str(e))

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    report("OK", "HEIC opener", "registered")
except Exception as e:  # noqa: BLE001
    report("FAIL", "HEIC opener", str(e))

# --- ExifTool -------------------------------------------------------------
section("ExifTool")
try:
    exe = config.exiftool_path()
    out = subprocess.run([exe, "-ver"], capture_output=True, text=True, timeout=60)
    report("OK", "ExifTool", f"{out.stdout.strip()}  ({exe})")
except Exception as e:  # noqa: BLE001
    report("FAIL", "ExifTool", str(e))

# --- This laptop ----------------------------------------------------------
section("This laptop")
try:
    import psutil
    ram = psutil.virtual_memory().total / 1024**3
    report("OK" if ram >= 15 else "WARN", "RAM",
           f"{ram:.1f} GB" + ("" if ram >= 15 else "  (fine; keep test sets to ~500 photos)"))
    report("OK", "CPU cores",
           f"{psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical; "
           f"engine will use {config.worker_count()} workers")
except Exception as e:  # noqa: BLE001
    report("WARN", "Hardware info", str(e))

free_gb = shutil.disk_usage(config.ROOT).free / 1024**3
report("OK" if free_gb >= 150 else "WARN", "Free disk space",
       f"{free_gb:.0f} GB" + ("" if free_gb >= 150 else
                              "  (a full wedding needs 100-150 GB; use an external SSD for it)"))

# Quick JPEG speed test: full decode vs. reduced-size decode (the trick Phase 1 uses)
try:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = np.empty((4000, 6000, 3), np.uint8)              # a 24 MP test image, ~70 MB
    pixels[:] = np.linspace(0, 230, 6000).astype(np.uint8)[None, :, None]
    pixels += rng.integers(0, 24, pixels.shape, dtype=np.uint8)   # grain, like a real photo
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, "JPEG", quality=90)
    data = buf.getvalue()

    def decode(reduced: bool) -> float:
        start = time.perf_counter()
        for _ in range(3):
            img = Image.open(io.BytesIO(data))
            if reduced:
                img.draft("RGB", (1500, 1000))   # decode at 1/4 size: much less work
            img.load()
        return (time.perf_counter() - start) / 3 * 1000

    full_ms, small_ms = decode(False), decode(True)
    report("OK", "JPEG decode, 24 MP",
           f"full {full_ms:.0f} ms  vs  reduced {small_ms:.0f} ms  "
           f"({full_ms / max(small_ms, 0.1):.1f}x faster)")
except Exception as e:  # noqa: BLE001
    report("WARN", "JPEG speed test", str(e))

# --- Project folders --------------------------------------------------------
section("Project folders")
for folder in (config.TEST_PHOTOS, config.WEDDINGS_DIR, config.SCRATCH_DIR):
    report("OK" if folder.exists() else "WARN", folder.relative_to(config.ROOT).as_posix(),
           "exists" if folder.exists() else "missing: run ks edge")

# --- Summary --------------------------------------------------------------
summary = ("All good. Phase 0 setup is complete." if failures == 0
           else f"{failures} FAIL line(s). Fix them, then run ks check again.")
print(f"\n{summary}")
lines.append("")
lines.append(summary)

config.DOCS_DIR.mkdir(exist_ok=True)
stamp = time.strftime("%Y-%m-%d %H:%M")
(config.DOCS_DIR / "phase0_check.txt").write_text(
    f"KeepSake setup check - {stamp}\n" + "\n".join(lines) + "\n", encoding="utf-8")
sys.exit(1 if failures else 0)
