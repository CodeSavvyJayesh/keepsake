"""KeepSake shared settings: folders and file types used by every phase."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# --- Folders --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]      # C:\KeepSake
DATA_DIR = ROOT / "data"                         # photos live here, never in Git
TEST_PHOTOS = DATA_DIR / "test_photos"
WEDDINGS_DIR = DATA_DIR / "weddings"
SCRATCH_DIR = DATA_DIR / "scratch"
TOOLS_DIR = ROOT / "tools"                       # ExifTool is downloaded here
DOCS_DIR = ROOT / "docs"

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
    """Use every core but one, so the laptop stays responsive while the engine runs."""
    return max(1, (os.cpu_count() or 2) - 1)
