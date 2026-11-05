"""KeepSake shared settings: folders and file types used by every phase."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# --- Folders --------------------------------------------------------------
FROZEN = bool(getattr(sys, "frozen", False))     # running as the packaged app (PyInstaller)
# C:\KeepSake; in the packaged app, the folder its files were unpacked to (read only)
ROOT = Path(getattr(sys, "_MEIPASS", "")) if FROZEN else Path(__file__).resolve().parents[1]


def app_data_dir(platform: str = sys.platform, home: Path | None = None,
                 env: dict | None = None) -> Path:
    """Where the packaged app keeps its catalog and cache: the user's own app-data folder,
    never inside the app itself (which is read-only once installed)."""
    home = home or Path.home()
    env = os.environ if env is None else env
    if platform == "darwin":
        return home / "Library" / "Application Support" / "KeepSake"
    if platform.startswith("win"):
        return Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local") / "KeepSake"
    return home / ".keepsake"


# KEEPSAKE_DATA / KEEPSAKE_DOCS move these elsewhere (used by the tests)
DATA_DIR = Path(os.environ.get("KEEPSAKE_DATA")
                or (app_data_dir() if FROZEN else ROOT / "data"))           # never in Git
TEST_PHOTOS = DATA_DIR / "test_photos"
WEDDINGS_DIR = DATA_DIR / "weddings"
SCRATCH_DIR = DATA_DIR / "scratch"
CACHE_DIR = DATA_DIR / "cache"                   # previews + catalog: on this laptop, never the shoot drive
TOOLS_DIR = ROOT / "tools"                       # ExifTool is downloaded here
MODELS_DIR = TOOLS_DIR / "models"                # AI model files (ks models), not in Git
DOCS_DIR = Path(os.environ.get("KEEPSAKE_DOCS")
                or (DATA_DIR / "reports" if FROZEN else ROOT / "docs"))

# --- File types -----------------------------------------------------------
RAW_TYPES = frozenset({
    ".cr2", ".cr3", ".crw",          # Canon
    ".nef", ".nrw",                  # Nikon
    ".arw", ".srf", ".sr2",          # Sony
    ".raf",                          # Fujifilm
    ".rw2",                          # Panasonic
    ".orf",                          # Olympus / OM System
    ".pef",                          # Pentax
    ".srw",                          # Samsung
    ".dng",                          # Leica, phones, converted RAW
    ".3fr", ".iiq", ".x3f",          # Hasselblad, Phase One, Sigma
})
JPEG_TYPES = frozenset({".jpg", ".jpeg"})
HEIF_TYPES = frozenset({".heic", ".heif", ".hif"})   # .hif = HEIF from Canon, Sony, Fuji
PHOTO_TYPES = RAW_TYPES | JPEG_TYPES | HEIF_TYPES

# Folders that never contain the studio's photos
SKIP_DIRS = frozenset({
    "_culling", "$recycle.bin", "system volume information",
    ".trashes", ".spotlight-v100", ".fseventsd", ".venv", "node_modules",
})


def exiftool_path() -> str:
    """Path to ExifTool: the copy in tools/ first, then anything on PATH."""
    exe = "exiftool.exe" if os.name == "nt" else "exiftool"
    bundled = TOOLS_DIR / "exiftool" / exe
    if bundled.exists():
        return str(bundled)
    found = shutil.which("exiftool")
    if found:
        return found
    raise FileNotFoundError("ExifTool not found. Run setup.ps1 again.")


def script_command(script: str, *args) -> list[str]:
    """How to start one of KeepSake's steps (scripts/<name>.py) as its own process: with this
    Python, or, in the packaged app (which has no separate Python), the app itself with
    `--run <name>` (app/keepsake_main.py)."""
    name = Path(script).stem
    if FROZEN:
        return [sys.executable, "--run", name, *(str(a) for a in args)]
    return [sys.executable, str(ROOT / "scripts" / f"{name}.py"), *(str(a) for a in args)]


SPEED_FILE = DATA_DIR / "speed.json"            # ks speedtest: the best worker counts for this laptop


def measured_workers(step: str) -> int | None:
    """The worker count `ks speedtest` found fastest for this step on this laptop, if measured."""
    try:
        import json
        n = json.loads(SPEED_FILE.read_text(encoding="utf-8")).get(step)
    except (OSError, ValueError, AttributeError):
        return None
    return n if isinstance(n, int) and n > 0 else None


def worker_count() -> int:
    """Physical CPU cores minus one, so the laptop stays responsive while the engine runs.

    Hyper-threads (the 'logical' cores) add little for image decoding, so they are not counted.
    The best number for a given drive is measured by `ks ingest --bench`.
    """
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except ImportError:
        cores = None
    if not cores:
        cores = max(1, (os.cpu_count() or 2) // 2)
    return max(1, cores - 1)


# --- Ingest (Phase 1) -----------------------------------------------------
# The copy of each photo that the AI analyses. 1620 px is the size of the Full-HD preview that
# Nikon and Canon hide inside every JPEG (Sony: 1616 px), so for most photos this copy is read
# straight out of the file instead of decoding the full 24-45 megapixel image.
ANALYSIS_PX = 1620
# A built-in preview smaller than this is not used. 900 lets in DJI drone DNGs (960 x 540
# preview): rendering their 73 MB RAW instead took 49 s per photo at the Dinesh & Ruchi wedding.
MIN_PREVIEW_PX = 900
MAX_PREVIEW_BYTES = 4 * 2**20  # a "preview" bigger than this is really a second full photo
THUMB_PX = 400                 # thumbnails for the photo grid
ANALYSIS_QUALITY = 88          # JPEG quality of the cached copies
THUMB_QUALITY = 80
