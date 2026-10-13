"""Tests for the photo finder. Run: ks test"""
from pathlib import Path

from engine.scan import find_photos, raw_jpeg_pairs


def make(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def found(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in find_photos(root))


def test_finds_every_photo_type(tmp_path):
    for name in ["a.CR3", "b.cr2", "c.NEF", "d.arw", "e.RAF", "f.rw2", "g.dng",
                 "h.JPG", "i.jpeg", "j.HEIC", "k.hif"]:
        make(tmp_path / name)
    assert len(found(tmp_path)) == 11


def test_skips_junk_and_system_folders(tmp_path):
    make(tmp_path / "keep.jpg")
    make(tmp_path / "._keep.jpg")                     # macOS resource file
    make(tmp_path / ".DS_Store")
    make(tmp_path / "clip.MP4")                       # video
    make(tmp_path / "notes.txt")
    make(tmp_path / "_culling" / "thumbs" / "t.jpg")  # our own cache
    make(tmp_path / ".hidden" / "h.jpg")
    make(tmp_path / "$RECYCLE.BIN" / "old.jpg")
    assert found(tmp_path) == ["keep.jpg"]


def test_nested_folders_unicode_and_same_names(tmp_path):
    make(tmp_path / "Day 1" / "shooter_A" / "IMG_0001.CR3")
    make(tmp_path / "Day 1" / "shooter_B" / "IMG_0001.CR3")
    make(tmp_path / "Mehendi फोटो 01.jpg")
    assert found(tmp_path) == [
        "Day 1/shooter_A/IMG_0001.CR3",
        "Day 1/shooter_B/IMG_0001.CR3",
        "Mehendi फोटो 01.jpg",
    ]


def test_missing_folder_gives_nothing(tmp_path):
    assert found(tmp_path / "does_not_exist") == []


def test_raw_jpeg_pairs(tmp_path):
    files = [
        tmp_path / "IMG_0101.CR3", tmp_path / "IMG_0101.JPG",   # pair
        tmp_path / "IMG_0102.CR3",                              # RAW only
        tmp_path / "IMG_0103.jpg",                              # JPEG only
        tmp_path / "x" / "IMG_0104.NEF", tmp_path / "y" / "IMG_0104.JPG",  # different folders
    ]
    pairs = raw_jpeg_pairs(files)
    assert [(r.name, j.name) for r, j in pairs] == [("IMG_0101.CR3", "IMG_0101.JPG")]
