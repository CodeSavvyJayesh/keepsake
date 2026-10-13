r"""Phase 0: create the photo folders and the edge-case files that try to break the engine.

Run:  ks edge
Safe to run again: existing files are left alone.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw  # noqa: E402

from engine.config import PHOTO_TYPES, SCRATCH_DIR, TEST_PHOTOS, WEDDINGS_DIR  # noqa: E402

LABEL_FOLDERS = ["good", "blur", "eyes_closed", "exposure", "bursts", "mixed_formats", "edge_cases"]
EDGE = TEST_PHOTOS / "edge_cases"


def main() -> None:
    for name in LABEL_FOLDERS:
        (TEST_PHOTOS / name).mkdir(parents=True, exist_ok=True)
    WEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    made: list[str] = []

    def write(name: str, data: bytes) -> None:
        path = EDGE / name
        if not path.exists():
            path.write_bytes(data)
            made.append(name)

    write("broken.cr2", b"This is not a real RAW file.\n")   # must be skipped, not crash
    write("._IMG_0001.CR2", b"")                             # macOS hidden file: ignore
    write("clip.MP4", b"")                                   # video: ignore
    write("empty.jpg", b"")                                  # copy cut short: skip and log

    unicode_name = EDGE / "Mehendi फोटो 01.jpg"               # spaces + Devanagari
    if not unicode_name.exists():
        img = Image.new("RGB", (1200, 800), (176, 112, 64))
        ImageDraw.Draw(img).ellipse([450, 250, 750, 550], fill=(240, 216, 190))
        img.save(unicode_name, quality=90)
        made.append(unicode_name.name)

    # Two photographers whose cameras both named a file IMG_0001
    good = sorted(p for p in (TEST_PHOTOS / "good").iterdir()
                  if p.is_file() and p.suffix.lower() in PHOTO_TYPES)
    if good:
        for shooter in ("shooter_A", "shooter_B"):
            dest = EDGE / shooter / f"IMG_0001{good[0].suffix}"
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(good[0], dest)
                made.append(f"{shooter}/{dest.name}")

    print(f"Photo folders ready in {TEST_PHOTOS}")
    print(f"Created {len(made)} edge-case file(s): {', '.join(made) if made else 'all already there'}")
    if not good:
        print("\nAdd photos to test_photos\\good, then run 'ks edge' again to create the "
              "two-photographer files.")
    print("\nStill to add by hand (they need a real camera):")
    print("  - a RAW + JPEG pair with the same name, e.g. IMG_0101.CR3 and IMG_0101.JPG")
    print("  - one vertical (portrait) photo straight from a camera")


if __name__ == "__main__":
    main()
