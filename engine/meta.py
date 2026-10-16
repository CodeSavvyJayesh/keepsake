"""Read photo metadata (camera, capture time) with ExifTool."""
from __future__ import annotations

import exiftool

from .config import exiftool_path

# -G      group names in keys, e.g. "EXIF:Model"
# -n      numbers as numbers, not text
# -charset filename=utf8   lets Windows file names with Hindi/Marathi characters work
# -fast2  skips the large maker-notes and trailer sections, which we never use: much
#         faster on external drives. Camera, capture time and orientation are unaffected.
EXIFTOOL_ARGS = ["-G", "-n", "-charset", "filename=utf8", "-fast2"]

BATCH_SIZE = 200   # files per ExifTool call: far faster than one call per file


def open_exiftool() -> exiftool.ExifToolHelper:
    """One long-running ExifTool process. Use it in a `with` block."""
    return exiftool.ExifToolHelper(
        executable=exiftool_path(),
        common_args=EXIFTOOL_ARGS,
        encoding="utf-8",
        check_execute=False,   # one bad file must not stop the other 199
    )


def tag(row: dict, name: str):
    """Get a tag by its short name: 'Model' finds 'EXIF:Model' or 'QuickTime:Model'."""
    for key, value in row.items():
        if key.rsplit(":", 1)[-1] == name:
            return value
    return None


def error_of(row: dict) -> str | None:
    """ExifTool's error message for a file it could not read, if any."""
    return row.get("ExifTool:Error")
