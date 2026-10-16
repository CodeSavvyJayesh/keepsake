"""KeepSake shared settings: folders and file types used by every phase."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# --- Folders --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]      # C:\KeepSake
# KEEPSAKE_DATA / KEEPSAKE_DOCS move these elsewhere (used by the tests)
DATA_DIR = Path(os.environ.get("KEEPSAKE_DATA") or ROOT / "data")   # never in Git
TEST_PHOTOS = DATA_DIR / "test_photos"
WEDDINGS_DIR = DATA_DIR / "weddings"
SCRATCH_DIR = DATA_DIR / "scratch"
CACHE_DIR = DATA_DIR / "cache"                   # previews + catalog: on this laptop, never the shoot drive
TOOLS_DIR = ROOT / "tools"                       # ExifTool is downloaded here
DOCS_DIR = Path(os.environ.get("KEEPSAKE_DOCS") or ROOT / "docs")

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
