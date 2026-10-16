"""Turn any photo file into an upright preview image, reading as little of the drive as possible.

Reading is the slow part of culling a shoot from a USB drive, not decoding. So, in order:

1. JPEG with a built-in preview. Nikon and Canon hide a ready-made 1620 x 1080 copy of each
   photo at the end of the JPEG (the Multi-Picture Format "Full HD" preview). Reading only the
   header and that copy reads about 5% of a 12 MB file.
2. JPEG without one (Leica, edited files): decoded at reduced size (libjpeg-turbo DCT scaling,
   Pillow's draft mode), which is several times faster than a full decode.
3. RAW: the JPEG preview the camera embedded (LibRaw via rawpy); only if there is no usable
   preview, a half-size render of the sensor data (slow, rare).
4. HEIF/HEIC: decoded with pillow-heif.

Every result is turned upright using the EXIF orientation, and checked against the shape of the
full photo, so a letterboxed or already-rotated preview is never used by mistake.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import (ANALYSIS_PX, HEIF_TYPES, JPEG_TYPES, MAX_PREVIEW_BYTES, MIN_PREVIEW_PX,
                     RAW_TYPES)
from .meta import FIELDS, exif_fields, orientation_of

Image.MAX_IMAGE_PIXELS = 300_000_000   # 100-150 MP medium-format photos are real photos

# Where the analysis copy came from
EMBEDDED = "embedded"        # the camera's built-in JPEG preview
DECODED = "decoded"          # the photo itself, decoded at reduced size
RAW_PREVIEW = "raw-preview"  # the JPEG preview inside a RAW file
RAW_RENDER = "raw-render"    # rendered from RAW sensor data (no usable preview)

_TRANSPOSE = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT],
    3: [Image.Transpose.ROTATE_180],
    4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE],
    6: [Image.Transpose.ROTATE_270],     # turn 90 degrees clockwise
    7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],      # turn 90 degrees anticlockwise
}
# LibRaw's 'flip' code -> the matching EXIF orientation
_FLIP_TO_EXIF = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5, 5: 8, 6: 6, 7: 7}

READ_BUFFER = 128 * 1024     # one read gets the whole header of a camera JPEG


@dataclass
class Loaded:
    image: Image.Image                 # upright RGB, long side at most ANALYSIS_PX
    source: str                        # EMBEDDED, DECODED, RAW_PREVIEW or RAW_RENDER
    width: int                         # the full photo, upright
    height: int
    meta: dict = field(default_factory=dict)
    bytes_read: int | None = None      # from the drive; None when LibRaw did the reading


class _CountingFile(io.FileIO):
    """A read-only file that counts the bytes it reads from the drive."""
    bytes_read = 0

    def readinto(self, buffer):
        n = super().readinto(buffer)
        self.bytes_read += n or 0
        return n

    def readall(self):
        data = super().readall()
        self.bytes_read += len(data)
        return data

    def read(self, size=-1):
        data = super().read(size)
        self.bytes_read += len(data or b"")
        return data


def _open_counted(path: Path) -> io.BufferedReader:
    return io.BufferedReader(_CountingFile(path, "r"), buffer_size=READ_BUFFER)


# --- Small helpers ----------------------------------------------------------------

def fit(img: Image.Image, px: int) -> Image.Image:
    """Shrink so the long side is at most px. Never enlarges."""
    w, h = img.size
    if max(w, h) <= px:
        return img
    k = px / max(w, h)
    size = (max(1, round(w * k)), max(1, round(h * k)))
    return img.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def draft_size(w: int, h: int, px: int) -> tuple[int, int] | None:
    """The size to ask libjpeg for, so it decodes at 1/2, 1/4 or 1/8 scale but not below px."""
    k = px / max(w, h)
    if k >= 1:
        return None
    return max(1, math.ceil(w * k)), max(1, math.ceil(h * k))


def same_shape(w: int, h: int, full_w: int, full_h: int, tolerance: float = 0.02) -> bool:
    """True when w x h has the same aspect ratio as the full photo (within 2%)."""
    if min(w, h, full_w, full_h) <= 0:
        return False
    a, b = w / h, full_w / full_h
    return abs(a - b) / b <= tolerance


def upright(img: Image.Image, orientation: int, full_w: int, full_h: int) -> Image.Image:
    """Rotate/flip to the way the photographer held the camera.

    full_w x full_h is the full photo as stored (before rotation). A preview that is already
    portrait while the stored photo is landscape was rotated by the camera: leave it.
    """
    steps = _TRANSPOSE.get(orientation)
    if not steps:
        return img
    if orientation >= 5 and full_w != full_h and (img.width > img.height) != (full_w > full_h):
        return img
    for step in steps:
        img = img.transpose(step)
    return img


def _rgb(img: Image.Image) -> Image.Image:
    """Decode now (while the file is open) into an RGB image in memory."""
    return img.convert("RGB")


def _empty_meta() -> dict:
    return {name: None for name in FIELDS}


# --- JPEG ---------------------------------------------------------------------------

def _use_embedded(im: Image.Image, full_w: int, full_h: int) -> bool:
    """Switch `im` to the camera's built-in preview if a suitable one is in the file.

    Tries the largest preview first; it must be a colour JPEG, at least MIN_PREVIEW_PX on its
    long side, and the same shape as the full photo (so it is not letterboxed or cropped).
    """
    frames = getattr(im, "n_frames", 1)
    mp = getattr(im, "mpinfo", None)        # only Multi-Picture JPEGs have this
    if frames < 2 or not mp:
        return False
    entries = mp.get(0xB002) or []
    order = sorted(range(1, min(frames, len(entries))),
                   key=lambda i: entries[i].get("Size", 0), reverse=True)
    for i in order:
        entry = entries[i]
        kind = str((entry.get("Attribute") or {}).get("MPType", ""))
        if not (kind.startswith("Large Thumbnail") or kind == "Undefined"):
            continue                                   # e.g. a 3D/panorama frame
        if not 0 < entry.get("Size", 0) <= MAX_PREVIEW_BYTES:
            continue
        try:
            im.seek(i)
        except (OSError, ValueError, SyntaxError, EOFError):
            continue
        w, h = im.size
        if im.mode != "RGB":
            continue                                   # e.g. an HDR gain map (grey)
        if max(w, h) < MIN_PREVIEW_PX:
            break                                      # the rest are smaller still
        if same_shape(w, h, full_w, full_h) or same_shape(h, w, full_w, full_h):
            return True
    if im.tell() != 0:
        im.seek(0)
    return False


def _load_jpeg(path: Path, prefer_embedded: bool) -> Loaded:
    with _open_counted(path) as f:
        im = Image.open(f)                     # reads the header only
        if im.format not in ("JPEG", "MPO"):   # a PNG or HEIC renamed to .jpg
            return _load_other(path)
        meta = exif_fields(im.getexif())
        icc = im.info.get("icc_profile")
        full_w, full_h = im.size
        img, source = None, DECODED
        if prefer_embedded and _use_embedded(im, full_w, full_h):
            try:
                img, source = _rgb(im), EMBEDDED
            except (OSError, ValueError, SyntaxError):   # damaged preview: use the photo
                im.seek(0)
        if img is None:
            size = draft_size(full_w, full_h, ANALYSIS_PX)
            if size:
                im.draft("RGB", size)          # decode at 1/2, 1/4 or 1/8 scale
            img = _rgb(im)
        bytes_read = f.raw.bytes_read
    img = upright(fit(img, ANALYSIS_PX), meta["orientation"], full_w, full_h)
    if icc:
        img.info["icc_profile"] = icc
    w, h = (full_h, full_w) if meta["orientation"] >= 5 else (full_w, full_h)
    return Loaded(img, source, w, h, meta, bytes_read)


# --- RAW ------------------------------------------------------------------------------

def _raw_meta(path: Path) -> dict:
    """EXIF of TIFF-based RAW files (NEF, DNG, CR2, ARW, PEF...) via Pillow, header only.

    Other RAW types (CR3, RAF, RW2, ORF) come back empty; ExifTool fills them in later.
    """
    try:
        with Image.open(path) as im:
            return exif_fields(im.getexif())
    except Exception:          # Pillow does not know this RAW format
        return _empty_meta()


def _load_raw(path: Path, prefer_embedded: bool) -> Loaded:
    import rawpy

    meta = _raw_meta(path)
    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        full_w, full_h = sizes.width, sizes.height
        orientation = _FLIP_TO_EXIF.get(sizes.flip, 1)
        meta["orientation"] = orientation
        img, source = None, RAW_PREVIEW
        if prefer_embedded:
            img = _raw_preview(raw, full_w, full_h)
        if img is None:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
            img, source = Image.fromarray(rgb), RAW_RENDER     # LibRaw already rotated it
            orientation_for_preview = 1
        else:
            orientation_for_preview = orientation
    img = upright(fit(img, ANALYSIS_PX), orientation_for_preview, full_w, full_h)
    w, h = (full_h, full_w) if orientation >= 5 else (full_w, full_h)
    return Loaded(img, source, w, h, meta, None)


def _raw_preview(raw, full_w: int, full_h: int) -> Image.Image | None:
    import rawpy
    try:
        thumb = raw.extract_thumb()
    except (rawpy.LibRawError, OSError, ValueError):
        return None
    try:
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data))
            w, h = img.size
            size = draft_size(w, h, ANALYSIS_PX)
            if size:
                img.draft("RGB", size)
        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            img = Image.fromarray(thumb.data)
            w, h = img.size
        else:
            return None
    except (OSError, ValueError, SyntaxError):
        return None
    if max(w, h) < MIN_PREVIEW_PX:
        return None
    if not (same_shape(w, h, full_w, full_h) or same_shape(h, w, full_w, full_h)):
        return None
    return _rgb(img)


# --- HEIF and anything else -------------------------------------------------------------

_heif_ready = False


def _register_heif() -> None:
    global _heif_ready
    if not _heif_ready:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _heif_ready = True


def _load_heif(path: Path) -> Loaded:
    _register_heif()
    return _load_other(path)


def _load_other(path: Path) -> Loaded:
    """Full decode with Pillow. pillow-heif rotates HEIF itself and resets the tag to 1."""
    with _open_counted(path) as f:
        im = Image.open(f)
        exif = im.getexif()
        meta = exif_fields(exif)
        original = im.info.get("original_orientation")    # set by pillow-heif
        full_w, full_h = im.size
        size = draft_size(full_w, full_h, ANALYSIS_PX)
        if size:
            im.draft("RGB", size)      # only JPEG supports this; others ignore it
        img = _rgb(im)
        bytes_read = f.raw.bytes_read
    img = upright(fit(img, ANALYSIS_PX), meta["orientation"], full_w, full_h)
    if meta["orientation"] >= 5:
        full_w, full_h = full_h, full_w
    if original:
        meta["orientation"] = orientation_of(original)
    return Loaded(img, DECODED, full_w, full_h, meta, bytes_read)


# --- Entry point ---------------------------------------------------------------------------

def load_photo(path: Path, prefer_embedded: bool = True) -> Loaded:
    """An upright analysis-size copy of any photo, plus its metadata.

    prefer_embedded=False always decodes the photo itself (used to measure the difference).
    Raises an exception for files that cannot be read; the caller records it.
    """
    ext = path.suffix.lower()
    if ext in RAW_TYPES:
        return _load_raw(path, prefer_embedded)
    if ext in HEIF_TYPES:
        return _load_heif(path)
    if ext in JPEG_TYPES:
        return _load_jpeg(path, prefer_embedded)
    return _load_other(path)
