r"""The blind test: KeepSake chooses from the same re-pick test photos, without ever having seen
this wedding's studio choices.

    ks cull "D:\All Raw images\Rishi & Aashi"         (first: learns only from the OTHER weddings)
    ks testcull "D:\All Raw images\Rishi & Aashi" [--keep 10]

After a studio-style cull (the default) KeepSake's choice is the cull's own Selected photos among
the test photos: one photo per shot. With --keep (or after ks cull --safe) it keeps that share of
the test photos, best scores first. The result goes to C:\KeepSake Output\KeepSake choice - <wedding>\Selected
and \Removed (copies of the test photos, numbered like the editor's folder), and the comparison
with the studio's old choices is printed and saved in data\repick. Do not give this folder to
the editor before they have done their re-pick.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import CACHE_DIR, DATA_DIR  # noqa: E402
from engine.cull import labelled_shoots  # noqa: E402
from engine.ingest import shoot_for  # noqa: E402

REPICK_DIR = DATA_DIR / "repick"
OUTPUT_ROOT = Path("C:/KeepSake Output")


def studio_rate(others: list[Path]) -> float:
    picks = photos = 0
    for db in others:
        with Catalog(db) as cat:
            n, k = cat.db.execute("SELECT COUNT(*), COALESCE(SUM(selected), 0) FROM labels").fetchone()
        photos, picks = photos + n, picks + k
    return picks / photos if photos else 0.10


def main() -> int:
    args = [a for a in sys.argv[1:] if a.strip() and not a.startswith("--")]
    keep = None
    if "--keep" in sys.argv:
        i = sys.argv.index("--keep")
        keep = float(sys.argv[i + 1].strip("%")) / 100
        args = [a for a in args if a != sys.argv[i + 1]]
    if not args:
        print('Give the wedding\'s photo folder, for example:  ks testcull "D:\\All Raw images\\Rishi & Aashi"')
        return 1
    root = clean_path(args[0])
    shoot = shoot_for(root)
    key_file, info_file = REPICK_DIR / f"{shoot.name}.csv", REPICK_DIR / f"{shoot.name}.json"
    if not key_file.exists():
        print(f'Make the test folder first:  ks testset "{root}"')
        return 1
    if info_file.exists():
        folder = Path(json.loads(info_file.read_text(encoding="utf-8"))["folder"])
    else:                                   # made by an older ks testset: look in the usual places
        candidates = [OUTPUT_ROOT / f"Re-pick test - {root.name}{s}" for s in ("", " (small)")]
        folder = next((c for c in candidates if c.is_dir()), candidates[0])
    if not folder.is_dir():
        print(f"The test folder is missing: {folder}. Make it again (same photos):  ks testset \"{root}\"")
        return 1
    key = list(csv.DictReader(open(key_file, encoding="utf-8")))
    with Catalog(shoot.db) as cat:
        scores = dict(cat.db.execute("SELECT id, score FROM cull"))
        decisions = dict(cat.db.execute("SELECT id, decision FROM cull"))
        settings = json.loads(cat.get("cull") or "{}")
    if not scores:
        print(f'Run the cull first:  ks cull "{root}"')
        return 1
    if settings.get("mode") != "other weddings" or shoot.name in settings.get("trained_on", []):
        print("Not a blind test: the last cull learned from this wedding's own choices. Prepare at least one "
              f'other wedding with the studio\'s choices, then run  ks cull "{root}"  again.')
        return 1
    others = labelled_shoots(CACHE_DIR, exclude=shoot.db)
    n = len(key)
    if keep is None and settings.get("style") == "studio":
        chosen = {r["copy"] for r in key if decisions.get(int(r["photo_id"])) == "selected"}
        n_keep = len(chosen)
        how = "one photo per shot, this studio's usual share"
    else:
        share = keep if keep is not None else studio_rate(others)
        n_keep = max(1, round(share * n))
        ranked = sorted(key, key=lambda r: -(scores.get(int(r["photo_id"])) or 0))
        chosen = {r["copy"] for r in ranked[:n_keep]}
        how = "best scores" if keep is not None else "best scores, this studio's usual share"
    out = OUTPUT_ROOT / f"KeepSake choice - {root.name}"
    for sub in ("Selected", "Removed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for r in key:
        src = folder / r["copy"]
        here, other = ("Selected", "Removed") if r["copy"] in chosen else ("Removed", "Selected")
        old = out / other / r["copy"]
        if old.exists():                       # KeepSake's own copy from an earlier run
            old.unlink()
        if src.exists() and not (out / here / r["copy"]).exists():
            shutil.copy2(src, out / here / r["copy"])

    studio = {r["copy"] for r in key if r["studio_selected"] == "1"}
    both = chosen & studio
    chance = n_keep * len(studio) / n
    lines = [f"KeepSake blind test - {root.name} - {now_text()}",
             f"  KeepSake learned from {len(settings.get('trained_on', []))} other weddings; it never saw this "
             "wedding's studio choices.",
             f"  Photos in the test          {n}",
             f"  KeepSake chose              {n_keep} ({n_keep / n:.0%}; {how})",
             f"  The studio had chosen       {len(studio)}",
             f"  Chosen by both              {len(both)}",
             f"    = {len(both) / max(n_keep, 1):.0%} of KeepSake's choices were also the studio's",
             f"    = KeepSake found {len(both) / max(len(studio), 1):.0%} of the studio's choices",
             f"  By pure chance, about {chance:.0f} would match.",
             f"  Folders: {out}\\Selected and \\Removed"]
    text = "\n".join(lines)
    print(text)
    (REPICK_DIR / f"{shoot.name}_keepsake.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
