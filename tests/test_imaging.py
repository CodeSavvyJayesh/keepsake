"""Tests for turning photo files into upright analysis copies. Run: ks test"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from engine.imaging import DECODED, EMBEDDED, RAW_PREVIEW, RAW_RENDER, load_photo
from engine.meta import camera_name, capture_time
from makephotos import RED, camera_jpeg, is_blue, is_red, marked, white_corner


# --- JPEG ---------------------------------------------------------------------------

def test_uses_the_cameras_built_in_preview(tmp_path):
    r = load_photo(camera_jpeg(tmp_path / "DSC_0001.JPG"))
    assert r.source == EMBEDDED
    assert r.image.size == (1620, 1080) and (r.width, r.height) == (3000, 2000)
    assert is_blue(r.image.getpixel((800, 500)))          # the preview, not the photo


def test_reads_only_a_small_part_of_a_big_file(tmp_path):
    p = camera_jpeg(tmp_path / "big.JPG", size=(4000, 2667), noise=True)
    r = load_photo(p)
    assert r.source == EMBEDDED
    assert r.bytes_read < 0.25 * p.stat().st_size


@pytest.mark.parametrize("orientation, size, corner", [
    (1, (1620, 1080), "top-left"),
    (2, (1620, 1080), "top-right"),
    (3, (1620, 1080), "bottom-right"),
    (4, (1620, 1080), "bottom-left"),
    (5, (1080, 1620), "top-left"),
    (6, (1080, 1620), "top-right"),
    (7, (1080, 1620), "bottom-right"),
    (8, (1080, 1620), "bottom-left"),
])
def test_every_exif_orientation_comes_out_upright(tmp_path, orientation, size, corner):
    r = load_photo(camera_jpeg(tmp_path / "p.JPG", orientation=orientation))
    assert r.image.size == size
    assert white_corner(r.image) == corner
    assert (r.width, r.height) == ((2000, 3000) if orientation >= 5 else (3000, 2000))


def test_no_preview_decodes_the_photo_at_reduced_size(tmp_path):
    p = camera_jpeg(tmp_path / "L1950113.JPG", size=(6000, 4000), preview=None,
                    make="LEICA CAMERA AG", model="LEICA Q (Typ 116)")
    r = load_photo(p)
    assert r.source == DECODED and r.image.size == (1620, 1080)
    assert is_red(r.image.getpixel((800, 500)))
    assert r.meta["camera"] == "LEICA Q (Typ 116)"


def test_letterboxed_preview_is_not_used(tmp_path):
    r = load_photo(camera_jpeg(tmp_path / "wide.JPG", size=(3200, 1800), preview=(1620, 1080)))
    assert r.source == DECODED and r.image.size == (1620, 911)


def test_too_small_preview_is_not_used(tmp_path):
    r = load_photo(camera_jpeg(tmp_path / "a.JPG", preview=(640, 427)))
    assert r.source == DECODED and r.image.size == (1620, 1080)


def test_grey_extra_image_is_not_a_preview(tmp_path):
    r = load_photo(camera_jpeg(tmp_path / "hdr.JPG", preview_mode="L"))
    assert r.source == DECODED and is_red(r.image.getpixel((800, 500)))


def test_full_decode_when_asked(tmp_path):
    r = load_photo(camera_jpeg(tmp_path / "a.JPG"), prefer_embedded=False)
    assert r.source == DECODED and is_red(r.image.getpixel((800, 500)))


def test_small_photo_is_never_enlarged(tmp_path):
    r = load_photo(camera_jpeg(tmp_path / "small.JPG", size=(800, 600), preview=None))
    assert r.image.size == (800, 600)


def test_damaged_preview_falls_back_to_the_photo(tmp_path):
    good = camera_jpeg(tmp_path / "a.JPG")
    cut = tmp_path / "cut.JPG"
    cut.write_bytes(good.read_bytes()[:-2000])            # end of the preview missing
    r = load_photo(cut)
    assert r.source == DECODED and is_red(r.image.getpixel((800, 500)))


def test_damaged_and_fake_files_raise(tmp_path):
    good = camera_jpeg(tmp_path / "a.JPG", preview=None)
    cut = tmp_path / "cut.JPG"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    junk = tmp_path / "junk.JPG"
    junk.write_bytes(b"not a photo" * 100)
    for bad in (cut, junk):
        with pytest.raises(Exception):
            load_photo(bad)


# --- Metadata ----------------------------------------------------------------------------

def test_reads_camera_details(tmp_path):
    m = load_photo(camera_jpeg(tmp_path / "a.JPG")).meta
    assert m["camera"] == "NIKON Z 9" and m["serial"] == "7600802"
    assert m["lens"] == "NIKKOR Z 35mm f/1.8 S"            # padding removed
    assert m["tz"] == "+05:30" and m["iso"] == 1600
    assert m["taken"] == "2026-08-27 16:41:34.52"


def test_capture_time_and_camera_names():
    assert capture_time("2026:08:27 16:41:34", "09")["taken"] == "2026-08-27 16:41:34.09"
    a = capture_time("2026:08:27 16:41:34", "52")["taken_s"]
    b = capture_time("2026:08:27 16:41:35", None)["taken_s"]
    assert abs((b - a) - 0.48) < 1e-6
    assert capture_time("0000:00:00 00:00:00")["taken"] is None
    assert capture_time(None)["taken_s"] is None
    assert camera_name("DJI", "FC8282") == "DJI FC8282"
    assert camera_name("NIKON CORPORATION", "NIKON Z 9") == "NIKON Z 9"
    assert camera_name("SONY", "ILCE-7M4") == "SONY ILCE-7M4"
    assert camera_name(None, None) is None


# --- RAW (LibRaw is replaced by a stand-in, so no real RAW file is needed) -------------------

class FakeRaw:
    def __init__(self, thumb, flip=0, size=(6000, 4000), render=None):
        self.thumb, self.render, self.rendered = thumb, render, False
        self.sizes = SimpleNamespace(width=size[0], height=size[1], flip=flip)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_thumb(self):
        if isinstance(self.thumb, Exception):
            raise self.thumb
        return self.thumb

    def postprocess(self, **kwargs):
        self.rendered = True
        return self.render


@pytest.fixture
def rawpy(monkeypatch):
    mod = types.ModuleType("rawpy")

    class LibRawError(Exception):
        pass

    mod.LibRawError = LibRawError
    mod.ThumbFormat = SimpleNamespace(JPEG="jpeg", BITMAP="bitmap")
    monkeypatch.setitem(sys.modules, "rawpy", mod)
    return mod


def jpeg_thumb(size, colour=(30, 30, 200)):
    import io
    buf = io.BytesIO()
    marked(size, colour).save(buf, "JPEG")
    return SimpleNamespace(format="jpeg", data=buf.getvalue())


def raw_file(tmp_path):
    p = tmp_path / "DSC_0001.NEF"
    p.write_bytes(b"raw sensor data")
    return p


def test_raw_uses_its_preview_and_turns_it_upright(tmp_path, rawpy):
    raw = FakeRaw(jpeg_thumb((1620, 1080)), flip=6)
    rawpy.imread = lambda path: raw
    r = load_photo(raw_file(tmp_path))
    assert r.source == RAW_PREVIEW and not raw.rendered
    assert r.image.size == (1080, 1620) and white_corner(r.image) == "top-right"
    assert (r.width, r.height) == (4000, 6000) and r.meta["orientation"] == 6


def test_raw_full_size_preview_is_shrunk(tmp_path, rawpy):
    rawpy.imread = lambda path: FakeRaw(jpeg_thumb((6000, 4000)))
    r = load_photo(raw_file(tmp_path))
    assert r.source == RAW_PREVIEW and r.image.size == (1620, 1080)


def test_drone_dng_small_preview_is_used(tmp_path, rawpy):
    raw = FakeRaw(jpeg_thumb((960, 540)), size=(8192, 4536))       # DJI: 16:9, 960 px preview
    rawpy.imread = lambda path: raw
    r = load_photo(raw_file(tmp_path))
    assert r.source == RAW_PREVIEW and not raw.rendered and r.image.size == (960, 540)


@pytest.mark.parametrize("thumb", ["tiny", "error"])
def test_raw_without_usable_preview_is_rendered(tmp_path, rawpy, thumb):
    rendered = np.zeros((3000, 2000, 3), dtype=np.uint8)          # LibRaw output is upright
    rendered[..., 0] = 200
    bad = jpeg_thumb((160, 120)) if thumb == "tiny" else rawpy.LibRawError("no thumbnail")
    raw = FakeRaw(bad, flip=6, render=rendered)
    rawpy.imread = lambda path: raw
    r = load_photo(raw_file(tmp_path))
    assert raw.rendered and r.source == RAW_RENDER
    assert r.image.size == (1080, 1620) and is_red(r.image.getpixel((500, 800)))


# --- HEIF (only where pillow-heif is installed, as on the KeepSake laptop) ---------------

def test_heif_photo(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    p = tmp_path / "IMG_0001.HEIC"
    marked((2000, 1500), RED).save(p, format="HEIF", quality=90)
    r = load_photo(p)
    assert r.source == DECODED and r.image.size == (1620, 1215)
    assert is_red(r.image.getpixel((800, 600)))
