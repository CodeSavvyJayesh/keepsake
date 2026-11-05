r"""The re-pick test: a folder of about 400 photos from a past wedding, for the studio's editor to
select again without looking at their old selection.

    ks testset "D:\All Raw images\Arjun & Riya" [400] ["C:\KeepSake Output\Re-pick test"] [--small]

--small: 2048-pixel copies (about 1 MB each instead of 13 MB), small enough to zip and send
on WhatsApp as one document.

It shows how often the studio picks the same photos twice: the best score any software can
reach. The photos are one continuous stretch of the wedding (all cameras, lined-up clocks)
where the studio picked about as often as in the whole wedding. The drive is only read; every
copy is checked. The studio's old choices for these photos are kept on this laptop
(data\repick), never in the folder given to the editor.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import DATA_DIR  # noqa: E402
from engine.deliver import OutputError, check_output, copy_verified  # noqa: E402
from engine.ingest import shoot_for  # noqa: E402
from engine.labels import attach_labels  # noqa: E402

REPICK_DIR = DATA_DIR / "repick"
README = """Re-pick test

1. Open this folder. It has {n} photos from one part of a wedding.
2. Choose the photos you would give to the couple, the way you normally do.
   Please do not look at your old selection: just choose fresh.
3. Copy the photos you choose into a new folder named "Repick" (inside this folder is fine).

That's all. Thank you!

---
Bas ek kaam hai: is folder ki {n} photos mein se woh photos choose karo jo aap couple ko doge,
aur unhe "Repick" naam ke naye folder mein copy karo. Purani selection mat dekhna.
"""


def save_small(src: Path, dst: Path, px: int = 2048) -> None:
    """A 2048-pixel JPEG of the original, upright (the original is only read)."""
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((px, px), Image.LANCZOS)
        im.convert("RGB").save(dst, "JPEG", quality=90)


def choose_window(rows: list[tuple], n: int) -> int:
    """Start of the stretch of n photos (in lined-up time order) whose pick rate is closest to the
    wedding's, among stretches shot by at least 3 cameras."""
    y = np.array([r[3] for r in rows], dtype=float)
    rate = y.mean()
    best, start = None, 0
    for a in range(0, max(1, len(rows) - n + 1), 25):
        bodies = len({r[1] for r in rows[a:a + n]})
        key = (bodies < 3, abs(y[a:a + n].mean() - rate))
        if best is None or key < best:
            best, start = key, a
    return start


def main() -> int:
    small = "--small" in sys.argv
    args = [a for a in sys.argv[1:] if a.strip() and not a.startswith("--")]
    if not args:
        print('Give the wedding\'s photo folder, for example:  ks testset "D:\\All Raw images\\Arjun & Riya"')
        return 1
    root = clean_path(args[0])
    n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 400
    out_arg = next((a for a in args[1:] if not a.isdigit()), None)
    shoot = shoot_for(root)
    out = clean_path(out_arg) if out_arg else Path("C:/KeepSake Output") / (
        f"Re-pick test - {root.name}" + (" (small)" if small else ""))
    if not shoot.db.exists():
        print(f'This wedding has not been prepared yet. Run first:  ks prepare "{root}"')
        return 1
    if not root.is_dir():
        print(f"Photo folder not found: {root}  (is the drive connected?)")
        return 1
    with Catalog(shoot.db) as cat:
        attach_labels(cat, root)
        offsets = dict(cat.db.execute("SELECT body, offset FROM clocks WHERE offset IS NOT NULL"))
        rows = sorted(
            [(pid, body, taken + offsets[body], sel or 0, path) for pid, body, taken, sel, path in cat.db.execute(
                "SELECT p.id, COALESCE(p.body, p.camera, ''), p.taken_s, l.selected, p.path FROM photos p "
                "LEFT JOIN labels l ON l.id = p.id WHERE p.status = 'done' AND p.taken_s IS NOT NULL")
             if body in offsets], key=lambda r: r[2])
        if not any(r[3] for r in rows):
            print("The studio's choices for this wedding are not known (ks match), so the test cannot be checked.")
            return 1
    if len(rows) < n:
        print(f"Only {len(rows)} photos have lined-up clocks; using all of them.")
        n = len(rows)
    start = choose_window(rows, n)
    chosen = rows[start:start + n]
    try:
        if not (out / "READ ME - what to do.txt").exists():      # a second run may refill its own folder
            check_output(root, out)
    except OutputError as e:
        print(e)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    REPICK_DIR.mkdir(parents=True, exist_ok=True)
    key_file = REPICK_DIR / f"{shoot.name}.csv"
    print(f"Copying {n} photos ({(chosen[-1][2] - chosen[0][2]) / 60:.0f} minutes of the wedding, "
          f"{len({r[1] for r in chosen})} cameras) to {out} ...")
    with open(key_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["copy", "original", "photo_id", "studio_selected"])
        for k, (pid, _body, _t, sel, path) in enumerate(chosen, 1):
            name = f"{k:04d}_{Path(path).name}"          # numbered in time order, all cameras together
            if small:
                save_small(root / path, out / name)
            else:
                copy_verified(root / path, out / name)
            writer.writerow([name, path, pid, sel])
    (out / "READ ME - what to do.txt").write_text(README.format(n=n), encoding="utf-8")
    import json
    (REPICK_DIR / f"{shoot.name}.json").write_text(json.dumps({"folder": str(out), "photos": n}), encoding="utf-8")
    print(f"\nReady: {out}")
    print(f"  {n} photos, in time order. Give this folder to the studio's editor.")
    print(f"  The studio's old choices for them are kept on this laptop only: {key_file}")
    print(f"  (made {now_text()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
