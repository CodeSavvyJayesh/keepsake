"""Small fake camera files for the tests (no real photos needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RED, BLUE, WHITE = (200, 30, 30), (30, 30, 200), (255, 255, 255)


def camera_exif(orientation: int = 1, make: str = "NIKON CORPORATION", model: str = "NIKON Z 9",
                taken: str | None = "2026:08:27 16:41:34", subsec: str | None = "52",
                serial: str | None = "7600802") -> Image.Exif:
    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    if taken:
        sub[0x9003] = taken
    if subsec:
        sub[0x9291] = subsec
    if serial:
        sub[0xA431] = serial
    sub[0xA434] = "NIKKOR Z 35mm f/1.8 S\x00\x00\x00"     # cameras pad with zero bytes
    sub[0x9011] = "+05:30"
    sub[0x8827] = 1600
    return exif


def marked(size: tuple[int, int], colour: tuple) -> Image.Image:
    """A plain image with a white square in its top-left corner, to check rotations."""
    img = Image.new("RGB", size, colour)
    s = max(4, min(size) // 8)
    img.paste(WHITE, (0, 0, s, s))
    return img


def camera_jpeg(path: Path, size=(3000, 2000), preview=(1620, 1080), orientation: int = 1,
                noise: bool = False, preview_mode: str = "RGB", **exif) -> Path:
    """A JPEG like a Nikon/Canon writes: red photo + (optionally) a blue built-in preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if noise:   # hard to compress, so the file is big like a real one
        pixels = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        main = Image.fromarray(pixels)
    else:
        main = marked(size, RED)
    tags = camera_exif(orientation, **exif)
    if preview:
        small = marked(preview, BLUE).convert(preview_mode)
        main.save(path, "MPO", save_all=True, append_images=[small], exif=tags, quality=90)
    else:
        main.save(path, "JPEG", exif=tags, quality=90)
    return path


def is_red(px) -> bool:
    return px[0] > 150 and px[2] < 90


def is_blue(px) -> bool:
    return px[2] > 150 and px[0] < 90


def is_white(px) -> bool:
    return min(px[:3]) > 200


def white_corner(img: Image.Image) -> str:
    """Which corner the white marker ended up in."""
    w, h = img.size
    corners = {"top-left": (3, 3), "top-right": (w - 4, 3),
               "bottom-left": (3, h - 4), "bottom-right": (w - 4, h - 4)}
    found = [name for name, xy in corners.items() if is_white(img.getpixel(xy))]
    return found[0] if len(found) == 1 else f"unclear {found}"
