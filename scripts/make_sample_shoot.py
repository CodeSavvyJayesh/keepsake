r"""Make a small fake shoot (no client photos) for checking a build end to end.

    python scripts/make_sample_shoot.py "<folder>"

30 photos: 24 Nikon-style JPEGs with built-in previews (some rotated), 2 without previews,
2 HEIC (if pillow-heif is installed), 1 RAW+JPEG pair and 1 damaged file.
Ingesting it must give: done 29, failed 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from makephotos import RED, camera_jpeg, marked  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1])
    for i in range(1, 25):
        orientation = 6 if i % 4 == 0 else 8 if i % 6 == 0 else 1
        camera_jpeg(root / "Haldi" / f"DSC_{i:04d}.JPG", orientation=orientation)
    for i in (1, 2):
        camera_jpeg(root / "Haldi" / f"L100000{i}.JPG", size=(4000, 2667), preview=None,
                    make="LEICA CAMERA AG", model="LEICA Q (Typ 116)", serial=None)
    camera_jpeg(root / "Sangeet" / "DSC_0100.JPG")
    (root / "Sangeet" / "DSC_0100.NEF").write_bytes(b"raw sensor data")   # pair: one photo
    (root / "Sangeet" / "broken.JPG").write_bytes(b"not a photo" * 50)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        for i in (1, 2):
            marked((2000, 1500), RED).save(root / "Sangeet" / f"IMG_000{i}.HEIC", format="HEIF")
    except ImportError:
        print("pillow-heif not installed: no HEIC files (expect done 27, failed 1)")
    print(f"Sample shoot written to {root}")


if __name__ == "__main__":
    main()
