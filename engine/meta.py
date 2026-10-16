"""Photo metadata: camera, capture time, exposure.

Most files are read with Pillow from the EXIF block at the start of the file, which costs
nothing extra because the same read gives us the preview. ExifTool is the fallback for files
Pillow cannot read (Canon CR3, Fuji RAF, HEIF...) and for files without a capture time.
"""
from __future__ import annotations

import re
from datetime import datetime

from .config import exiftool_path

# -G      group names in keys, e.g. "EXIF:Model"
# -n      numbers as numbers, not text
# -charset filename=utf8   lets Windows file names with Hindi/Marathi characters work
# -fast2  skips the large maker-notes and trailer sections, which we never use: much
#         faster on external drives. Camera, capture time and orientation are unaffected.
EXIFTOOL_ARGS = ["-G", "-n", "-charset", "filename=utf8", "-fast2"]

BATCH_SIZE = 200   # files per ExifTool call: far faster than one call per file

# The fields every photo gets in the catalog (None when the camera did not record it)
FIELDS = ("camera", "serial", "lens", "taken", "taken_s", "tz", "orientation",
          "iso", "shutter", "aperture", "focal", "flash")

_EPOCH = datetime(1970, 1, 1)
_DATE = re.compile(r"(\d{4})[:\-](\d\d)[:\-](\d\d)[ T](\d\d):(\d\d):(\d\d)")


def open_exiftool():
    """One long-running ExifTool process (PyExifTool's ExifToolHelper). Use it in a `with` block."""
    import exiftool
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


# --- Cleaning values ---------------------------------------------------------

def text(value) -> str | None:
    """Camera text without the padding cameras add ('NIKKOR Z 35mm f/1.8 S\\x00\\x00...')."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    value = str(value).replace("\x00", "").strip()
    return value or None


def number(value) -> float | None:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if result == result else None     # NaN (0/0 rationals) -> None


def integer(value) -> int | None:
    result = number(value)
    return int(result) if result is not None else None


def orientation_of(value) -> int:
    """EXIF orientation 1-8; anything else counts as 1 (upright)."""
    result = integer(value)
    return result if result in range(1, 9) else 1


def camera_name(make: str | None, model: str | None) -> str | None:
    """'NIKON Z 9', 'Canon EOS R5'; adds the brand when the model lacks it: 'DJI FC8282'."""
    if not model:
        return make
    if make:
        brand = make.split()[0]
        if brand.lower() not in model.lower():
            return f"{brand} {model}"
    return model


def capture_time(taken: str | None, subsec=None) -> dict:
    """{'taken': '2026-08-27 16:41:34.52', 'taken_s': seconds} from EXIF date + fraction.

    The time is the camera's clock, as set by the photographer. Comparing clocks between
    cameras is done later (Phase 3); here it is only recorded.
    """
    m = _DATE.search(taken or "")
    if not m:
        return {"taken": None, "taken_s": None}
    try:
        dt = datetime(*(int(x) for x in m.groups()))
    except ValueError:                       # '0000:00:00 00:00:00' and similar
        return {"taken": None, "taken_s": None}
    digits = "".join(ch for ch in str(subsec if subsec is not None else "") if ch.isdigit())
    fraction = float("0." + digits) if digits else 0.0
    stamp = dt.strftime("%Y-%m-%d %H:%M:%S") + (f".{digits[:3]}" if digits else "")
    return {"taken": stamp, "taken_s": (dt - _EPOCH).total_seconds() + fraction}


def body_of(camera: str | None, serial: str | None) -> str | None:
    """One physical camera: two Nikon Z 9s at a wedding are told apart by serial number."""
    if not camera:
        return None
    return f"{camera} #{serial}" if serial else camera


# --- Reading -------------------------------------------------------------------

def exif_fields(exif) -> dict:
    """The catalog fields from a Pillow Image.Exif (JPEG, and TIFF-based RAW like NEF/DNG)."""
    sub = exif.get_ifd(0x8769)            # the EXIF sub-directory holds the shooting details
    fields = {
        "camera": camera_name(text(exif.get(0x010F)), text(exif.get(0x0110))),
        "serial": text(sub.get(0xA431)),          # BodySerialNumber
        "lens": text(sub.get(0xA434)),            # LensModel
        "tz": text(sub.get(0x9011)),              # OffsetTimeOriginal, e.g. '+05:30'
        "orientation": orientation_of(exif.get(0x0112)),
        "iso": integer(sub.get(0x8827)),
        "shutter": number(sub.get(0x829A)),       # seconds
        "aperture": number(sub.get(0x829D)),      # f-number
        "focal": number(sub.get(0x920A)),         # mm
        "flash": integer(sub.get(0x9209)),        # bit 0 = flash fired
    }
    fields.update(capture_time(text(sub.get(0x9003)), text(sub.get(0x9291))))
    return fields


_FALLBACK_TAGS = ["Make", "Model", "SerialNumber", "LensModel", "Lens", "DateTimeOriginal",
                  "SubSecTimeOriginal", "OffsetTimeOriginal", "CreateDate", "Orientation",
                  "ISO", "ExposureTime", "FNumber", "FocalLength", "Flash"]
_exiftool = None


def exiftool_fields(path) -> dict:
    """The catalog fields read by ExifTool, for files Pillow cannot read.

    Starts one ExifTool on first use in each worker process and keeps it running; it is
    stopped automatically when the process ends.
    """
    global _exiftool
    if _exiftool is None:
        from multiprocessing.util import Finalize
        et = open_exiftool()
        et.run()
        Finalize(et, et.terminate, exitpriority=10)
        _exiftool = et
    rows = _exiftool.get_tags([str(path)], tags=_FALLBACK_TAGS)
    row = rows[0] if rows else {}
    if error_of(row):
        raise OSError(f"ExifTool: {error_of(row)}")
    fields = {
        "camera": camera_name(text(tag(row, "Make")), text(tag(row, "Model"))),
        "serial": text(tag(row, "SerialNumber")),
        "lens": text(tag(row, "LensModel")) or text(tag(row, "Lens")),
        "tz": text(tag(row, "OffsetTimeOriginal")),
        "orientation": orientation_of(tag(row, "Orientation")),
        "iso": integer(tag(row, "ISO")),
        "shutter": number(tag(row, "ExposureTime")),
        "aperture": number(tag(row, "FNumber")),
        "focal": number(tag(row, "FocalLength")),
        "flash": integer(tag(row, "Flash")),
    }
    taken = text(tag(row, "DateTimeOriginal")) or text(tag(row, "CreateDate"))
    fields.update(capture_time(taken, tag(row, "SubSecTimeOriginal")))
    return fields
