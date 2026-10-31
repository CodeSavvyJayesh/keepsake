r"""The blind test on every past wedding: KeepSake chooses like the studio, without seeing its answers.

    ks blindtest                 every prepared wedding with the studio's choices (needs 2 or more)
    ks blindtest --list-removed  copy only the Selected photos; Not selected are listed in a file
    ks blindtest --shortlist     Shortlist, as the app makes it: Selected (the best) + Maybe
    ks blindtest --shortlist --numbers   the same, numbers only (no copies: a few minutes)
    ks blindtest --add-studio "C:\KeepSake Output\KeepSake blind test 2026-10-19 19.36"
                                 add the studio's folders to an earlier run (no new decisions)

For each wedding, a model trained only on the OTHER weddings scores every photo. KeepSake then keeps
as many photos as this studio usually does (its pick rate in the other weddings), one photo per
shot (near-identical frames grouped, see engine/shots.py), choosing the best frame of each shot.

Output, on this laptop (copies of KeepSake's own previews, no drive needed):
    C:\KeepSake Output\KeepSake blind test <date>\<wedding>\KeepSake selected (612)\...
                                                           \KeepSake not selected (5,672)\...
                                                           \Studio selected (624)\...
                                                           \Studio not selected (5,660)\...
                                                           \Selected by both (132)\...
    ...\Summary.txt     how many of KeepSake's choices match the studio's, per wedding
Files are numbered in time order (all cameras together), e.g. 00412_DSC_8037.jpg.
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.config import CACHE_DIR  # noqa: E402
from engine.cull import labelled_shoots, model_factory, per_event_scale  # noqa: E402
from engine.ingest import cache_path, duration  # noqa: E402
from engine.learn import combine, past_wedding  # noqa: E402
from engine.shots import (CLIP_WEIGHT, FAULT_FRAME_W, FAULT_SHOT_W, centred_clip, clip_ridge, face_details,  # noqa: E402
                          BEST_BY, SHORTLIST_FAULT_W, SHORTLIST_MAX, SHORTLIST_TARGET, face_faults, frame_penalty, group_shots, shortlist, tiers,
                          one_per_shot, relative_to_shot, train_frame_ranker)

OUTPUT_ROOT = Path("C:/KeepSake Output")
PREVIEW_MB = 0.3


def display_name(cat: Catalog, fallback: str) -> str:
    root = cat.get("root")
    if not root:
        return fallback
    p = Path(root)
    return p.parent.name if "raw" in p.name.lower() and p.parent.name else p.name


def time_order(cat: Catalog) -> dict[int, tuple[int, str]]:
    """{photo id: (place in time order, file name)}, with camera clocks lined up."""
    offsets = dict(cat.db.execute("SELECT body, offset FROM clocks WHERE offset IS NOT NULL"))
    rows = []
    for pid, path, body, taken in cat.db.execute(
            "SELECT id, path, COALESCE(body, camera, ''), taken_s FROM photos WHERE status = 'done'"):
        t = taken + offsets.get(body, 0.0) if taken is not None else float("inf")
        rows.append((t, path, pid))
    rows.sort()
    return {pid: (k, Path(path).name) for k, (_, path, pid) in enumerate(rows, 1)}


def link_or_copy(src: Path, dst: Path) -> None:
    """A second name for the same file (no extra space on the laptop), or a copy if that fails."""
    import os
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def add_studio_folders(folder: Path) -> tuple[int, int, int] | None:
    """Next to KeepSake's folders of one wedding: the studio's own choices, and the photos both chose."""
    choices = folder / "KeepSake choices.csv"
    if not choices.exists():
        return None
    rows = list(csv.DictReader(open(choices, encoding="utf-8")))
    sel_dir = next(folder.glob("KeepSake selected (*)"), None)
    rem_dir = next(folder.glob("KeepSake not selected (*)"), None) or next(folder.glob("KeepSake removed (*)"), None)
    studio = [r for r in rows if r["studio_picked"] == "1"]
    both = [r for r in studio if r["keepsake"] == "selected"]
    targets = {"Studio selected": studio, "Studio not selected": [r for r in rows if r["studio_picked"] != "1"],
               "Selected by both": both}
    for label, members in targets.items():
        out = folder / f"{label} ({len(members):,})"
        out.mkdir(exist_ok=True)
        for r in members:
            src_dir = sel_dir if r["keepsake"] == "selected" else rem_dir
            src = src_dir / r["file"] if src_dir is not None else None
            if src is not None and src.exists() and not (out / r["file"]).exists():
                link_or_copy(src, out / r["file"])
    return len(studio), len(rows) - len(studio), len(both)


def couple_of(shoot: str) -> str:
    """'all_raw_images_shevaun_brian_jaisalmer' and 'shevaun_brian_jaisalmer_22_11_2025_..._sangeet' -> 'shevaun_brian'."""
    name = shoot[len("all_raw_images_"):] if shoot.startswith("all_raw_images_") else shoot
    return "_".join(name.split("_")[:2])


def main() -> int:
    if "--add-studio" in sys.argv:
        i = sys.argv.index("--add-studio")
        run = Path(sys.argv[i + 1].strip('"')) if i + 1 < len(sys.argv) else None
        if run is None or not run.is_dir():
            print('Give the blind test folder, for example:  ks blindtest --add-studio "C:\\KeepSake Output\\KeepSake blind test 2026-10-19 19.36"')
            return 1
        for folder in sorted(p for p in run.iterdir() if p.is_dir()):
            counts = add_studio_folders(folder)
            if counts:
                print(f"  {folder.name:<34} studio selected {counts[0]:,}, studio not selected {counts[1]:,}, "
                      f"selected by both {counts[2]:,}")
        print(f"Done: {run}")
        return 0
    list_removed = "--list-removed" in sys.argv
    numbers_only = "--numbers" in sys.argv
    dbs = labelled_shoots(CACHE_DIR)
    if len(dbs) < 2:
        print("Needs at least 2 prepared weddings with the studio's choices (ks prepare).")
        return 1
    start = time.perf_counter()
    make_model, model_name = model_factory()
    print(f"Reading {len(dbs)} weddings...")
    datasets, details, names, orders, faults = [], [], {}, {}, []
    wide = float(sys.argv[sys.argv.index("--wide") + 1]) if "--wide" in sys.argv else 1.0
    short = None
    if "--shortlist" in sys.argv:
        i = sys.argv.index("--shortlist")
        nxt = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        short = float(nxt.strip("%")) / 100 if nxt.strip("%").replace(".", "", 1).isdigit() else SHORTLIST_TARGET
    for db in dbs:
        ds = past_wedding(db, face_details)            # saved from an earlier run when unchanged
        details.append(ds.meta["faces"])
        with Catalog(db) as cat:
            faults.append(face_faults(cat, list(ds.ids)))
            names[db.parent.name] = display_name(cat, db.parent.name)
            orders[db.parent.name] = time_order(cat)
        datasets.append(ds)
        print(f"  {names[db.parent.name]:<34}{len(ds.ids):>7,} photos {int((ds.y == 1).sum()):>6,} studio picks")
    data = combine(datasets, by_wedding=True)
    wedding = np.array(data.group)
    couple = np.array([couple_of(w) for w in wedding])      # parts of one wedding are never learned from for each other
    ids = np.array(data.ids)
    y = data.y
    known = y >= 0
    shot = group_shots(data.moment, data.clip, list(wedding))
    R = relative_to_shot(np.hstack([data.X, np.vstack(details)]), shot)     # frame choice: + each face's details
    C = centred_clip(data.clip, wedding)                                      # shot choice: + full CLIP content
    bad = (np.concatenate(faults) > 0).astype(float)                         # closed eyes / profile faces
    penalty = frame_penalty(data.X, data.names) + FAULT_FRAME_W * bad

    total_files = len(ids) if not list_removed else int(0.12 * len(ids))
    out_root = OUTPUT_ROOT / f"KeepSake blind test {time.strftime('%Y-%m-%d %H.%M')}"
    out_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root).free / 1e6
    if not numbers_only and free < total_files * PREVIEW_MB + 2000:
        print(f"Not enough free space for all copies (about {total_files * PREVIEW_MB / 1000:.0f} GB); "
              "copying only Selected and listing Not selected.")
        list_removed = True

    lines = [f"KeepSake blind test - {now_text()}",
             f"Each wedding was decided by a {model_name} model trained only on the other weddings.",
             (f"Shortlist: KeepSake keeps the moments expected to hold {short:.0%} of the studio's picks, one photo per shot (best frame)."
              if short else
              f"KeepSake keeps this studio's usual share of photos{f' x {wide:g}' if wide != 1 else ''}, one photo per shot (best frame)."),
             "Shots ranked by the model plus the photo's full content (CLIP); best frame by each face's eyes,",
             "head turn, sharpness and smile compared with the other frames of the burst.", "",
             *([f"Selected: the best, this studio's usual share; Maybe: the rest of the Shortlist (at most "
                f"{SHORTLIST_MAX:.0%} of the photos in all). Studio picks found as the same photo, and counting a",
                "near-identical frame of the same shot (+frame).", "",
                f"{'Wedding':<30}{'photos':>8}{'Selected':>10}{'Maybe':>8}{'Not sel.':>10}{'studio':>8}"
                f"{'in Selected':>13}{'+frame':>8}{'in Sel.+Maybe':>15}{'+frame':>8}"] if short else
              [f"{'Wedding':<30}{'photos':>8}{'KeepSake selected':>19}{'not selected':>14}"
               f"{'studio picks':>14}{'same photo':>12}{'same shot':>11}{'chance':>8}"])]
    tier_totals = np.zeros(11)
    faulty = 0
    totals = np.zeros(7)
    for w in sorted(set(wedding), key=lambda k: names[k].lower()):
        t0 = time.perf_counter()
        test = np.flatnonzero(wedding == w)
        train = (couple != couple_of(w)) & known
        model = make_model().fit(data.X[train], y[train])
        score = np.zeros(len(y))
        content = clip_ridge(C[train], y[train])
        score[test] = ((1 - CLIP_WEIGHT) * per_event_scale(model.predict(data.X[test]), [w] * len(test))
                       + CLIP_WEIGHT * per_event_scale(C[test] @ content, [w] * len(test)))
        ranker = train_frame_ranker(R, y, shot, np.flatnonzero(train))
        frame_score = np.zeros(len(y))
        frame_score[test] = (ranker.predict(R[test]) if ranker is not None else score[test]) - penalty[test]
        best_set = set()
        if short:
            n_best = max(1, int(round(y[train].mean() * len(test))))
            best, chosen, kept_in = tiers(test, score - FAULT_SHOT_W * bad, score - SHORTLIST_FAULT_W * bad,
                                          frame_score, shot, n_best, short)
            best_set = set(best.tolist())
            other_way = "studio" if BEST_BY == "shortlist" else "shortlist"     # the other way to rank Selected
            alt, _, _ = tiers(test, score - FAULT_SHOT_W * bad, score - SHORTLIST_FAULT_W * bad, frame_score,
                              shot, n_best, short, best_by=other_way)
        else:
            n = int(round(y[train].mean() * len(test) * wide))
            chosen, kept_in = one_per_shot(test, score - FAULT_SHOT_W * bad, frame_score, shot, n)
        chosen_set = set(chosen.tolist())

        order = orders[w]
        folder = out_root / names[w]
        n_sel = len(best_set) if short else len(chosen)
        sel_dir = folder / f"KeepSake selected ({n_sel:,})"
        maybe_dir = folder / f"KeepSake maybe ({len(chosen) - n_sel:,})"
        rem_dir = folder / f"KeepSake not selected ({len(test) - len(chosen):,})"
        if not numbers_only:
            sel_dir.mkdir(parents=True, exist_ok=True)
            if short:
                maybe_dir.mkdir(parents=True, exist_ok=True)
            if not list_removed:
                rem_dir.mkdir(parents=True, exist_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        cache = dbs[[d.parent.name for d in dbs].index(w)].parent
        with open(folder / "KeepSake choices.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "keepsake", "reason", "studio_picked"])
            for i in test:
                pid = int(ids[i])
                k, base = order.get(pid, (0, f"{pid}.jpg"))
                name = f"{k:05d}_{Path(base).stem}.jpg"
                if i in chosen_set and (not short or i in best_set):
                    target, decision, reason = sel_dir / name, "selected", "best frame of a chosen shot"
                elif i in chosen_set:
                    target, decision, reason = maybe_dir / name, "maybe", "best frame of a Shortlist shot"
                else:
                    best = kept_in.get(i)
                    if best is not None:
                        kb, bb = order.get(int(ids[best]), (0, ""))
                        reason = f"similar to {kb:05d}_{Path(bb).stem}.jpg (same shot), which is kept"
                    else:
                        reason = "shot not among the best"
                    target, decision = rem_dir / name, "not selected"
                writer.writerow([name, decision, reason, int(y[i] == 1)])
                if not numbers_only and (decision != "not selected" or not list_removed):
                    src = cache_path(cache, "analysis", pid)
                    if src.exists():
                        shutil.copyfile(src, target)

        if not list_removed and not numbers_only:
            add_studio_folders(folder)
        picks = test[y[test] == 1]
        chosen_shots = {shot[i] for i in chosen}
        same = int(sum(y[i] == 1 for i in chosen))
        same_shot = int(sum(shot[i] in chosen_shots for i in picks))     # studio picks whose moment is in Selected
        chance = len(chosen) * len(picks) / len(test)
        totals += [len(test), len(chosen), len(test) - len(chosen), len(picks), same, same_shot, chance]
        faulty += int(bad[chosen].sum())
        if short:
            best_shots = {shot[i] for i in best_set}
            alt_shots, alt_set = {shot[i] for i in alt}, set(alt.tolist())
            row = [len(test), len(best_set), len(chosen) - len(best_set), len(test) - len(chosen), len(picks),
                   sum(i in best_set for i in picks), sum(shot[i] in best_shots for i in picks), same, same_shot,
                   sum(shot[i] in alt_shots for i in picks), sum(i in alt_set for i in picks)]
            tier_totals += row
            lines.append(f"{names[w][:29]:<30}{row[0]:>8,}{row[1]:>10,}{row[2]:>8,}{row[3]:>10,}{row[4]:>8,}"
                         f"{row[5]:>8,} {row[5] / max(row[4], 1):>3.0%}{row[6] / max(row[4], 1):>8.0%}"
                         f"{row[7]:>10,} {row[7] / max(row[4], 1):>3.0%}{row[8] / max(row[4], 1):>8.0%}")
        else:
            lines.append(f"{names[w][:29]:<30}{len(test):>8,}{len(chosen):>19,}{len(test) - len(chosen):>14,}"
                         f"{len(picks):>14,}{same:>12,}{same_shot:>11,}{chance:>8.0f}")
        print(f"  {names[w]:<34} selected {len(chosen):,}, not selected {len(test) - len(chosen):,}; "
              f"{same} same photos as the studio, {same_shot} same shots ({duration(time.perf_counter() - t0)})")
    if short:
        t = tier_totals
        lines += [f"{'All weddings':<30}{int(t[0]):>8,}{int(t[1]):>10,}{int(t[2]):>8,}{int(t[3]):>10,}{int(t[4]):>8,}"
                  f"{int(t[5]):>8,} {t[5] / t[4]:>3.0%}{t[6] / t[4]:>8.0%}{int(t[7]):>10,} {t[7] / t[4]:>3.0%}"
                  f"{t[8] / t[4]:>8.0%}", "",
                  f"Selected ({t[1] / t[0]:.0%} of the photos) holds {t[5] / t[4]:.0%} of the studio's picks as the same photo, "
                  f"{t[6] / t[4]:.0%} counting a near-identical frame.",
                  f"Selected + Maybe ({(t[1] + t[2]) / t[0]:.0%} of the photos) holds {t[7] / t[4]:.0%} as the same photo, "
                  f"{t[8] / t[4]:.0%} counting a near-identical frame.",
                  f"Selected is the {'Shortlist' if BEST_BY == 'shortlist' else 'Studio style'} order's first "
                  f"moments. Ranked the other way ({'like Studio style' if BEST_BY == 'shortlist' else 'the Shortlist order'}) "
                  f"it would hold {t[10] / t[4]:.0%} as the same photo, {t[9] / t[4]:.0%} counting a near-identical frame.",
                  f"Faces: {faulty / max(t[1] + t[2], 1):.1%} of Selected + Maybe have a main person with closed or "
                  "half-closed eyes or a face turned to the side.",
                  f"Time {duration(time.perf_counter() - start)}"]
        text = "\n".join(lines)
        (out_root / "Summary.txt").write_text(text + "\n", encoding="utf-8")
        print("\n" + text)
        print(f"\nFolders: {out_root}")
        return 0
    lines += [f"{'All weddings':<30}{int(totals[0]):>8,}{int(totals[1]):>19,}{int(totals[2]):>14,}"
              f"{int(totals[3]):>14,}{int(totals[4]):>12,}{int(totals[5]):>11,}{totals[6]:>8.0f}", "",
              f"Studio's picks found: {totals[4] / totals[3]:.0%} of the studio's own photos are in KeepSake's Selected; "
              f"{totals[5] / totals[3]:.0%} counting a near-identical frame of the same moment.",
              f"Faces: {faulty / max(totals[1], 1):.1%} of KeepSake's Selected have a main person with closed or half-closed "
              "eyes or a face turned to the side (rule: avoided whenever the moment has a better frame).",
              f"Same photo: {totals[4] / totals[1]:.0%} of KeepSake's choices were also the studio's "
              f"(pure chance: {totals[6] / totals[1]:.0%}).",
              f"Same shot: for {totals[5] / totals[3]:.0%} of the studio's picks, KeepSake's Selected has that photo or a "
              "near-identical frame of the same moment.",
              "'Same photo' counts each KeepSake choice once; 'same shot' counts the studio's picks whose moment KeepSake kept; 'chance' = matches expected by luck.",
              f"Time {duration(time.perf_counter() - start)}"]
    text = "\n".join(lines)
    (out_root / "Summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nFolders: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
