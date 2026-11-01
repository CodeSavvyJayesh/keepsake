r"""Phase 2b: find every face in an ingested shoot and read eyes, smiles and head pose.

    ks faces "E:\Sharma Wedding\Raw"

Works from the analysis copies on this laptop (the shoot drive need not be plugged in).
Needs the model files (ks models) and MediaPipe for the eyes. If the studio's choices are
known (ks match), it reports how well open eyes and smiles match them, and writes a review
page of the faces with the most and least closed eyes.
"""
from __future__ import annotations

import argparse
import html
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.bursts import group_bursts  # noqa: E402
from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import measured_workers  # noqa: E402
from engine.faces import FACES_VERSION, FaceReader, eye_state, faces_task, square_crop  # noqa: E402
from engine.ingest import Progress, Runner, cache_path, duration, shoot_for  # noqa: E402
from engine.labels import attach_labels  # noqa: E402
from engine.quality import auc  # noqa: E402

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ks faces", description="Find faces; read eyes and smiles.")
    ap.add_argument("folder", help="the shoot folder you ingested, in quotes")
    ap.add_argument("--workers", type=int, help="number of worker processes")
    ap.add_argument("--redo", action="store_true", help="read every photo again")
    return ap.parse_args()


def photo_summaries(cat: Catalog) -> dict[int, dict]:
    """Per photo: faces, main faces, their worst (most closed) eyes, closed-eye flag, smile."""
    out: dict[int, dict] = {}
    for pid, body, taken, n, sel in cat.db.execute(
            "SELECT p.id, p.body, p.taken_s, s.faces, l.selected FROM photos p "
            "JOIN face_scan s ON s.id = p.id LEFT JOIN labels l ON l.id = p.id WHERE s.ok = 1"):
        out[pid] = {"body": body, "taken": taken, "faces": n, "selected": sel, "main": 0,
                    "worst_eyes": None, "closed": False, "smile": None, "largest": 0.0}
    smiles: dict[int, list] = defaultdict(list)
    for pid, rel_w, subject, eyes, bl, br, smile in cat.db.execute(
            "SELECT photo_id, rel_w, subject, eyes, blink_l, blink_r, smile FROM faces"):
        s = out.get(pid)
        if s is None:
            continue
        s["largest"] = max(s["largest"], rel_w or 0)
        if not subject:
            continue
        s["main"] += 1
        if eyes in ("open", "closed", "smiling", "unsure"):
            worst = max(bl or 0, br or 0)
            s["worst_eyes"] = worst if s["worst_eyes"] is None else max(s["worst_eyes"], worst)
            s["closed"] = s["closed"] or eyes == "closed"
            smiles[pid].append(smile or 0.0)
    for pid, values in smiles.items():
        out[pid]["smile"] = sum(values) / len(values)
    return out


def label_lines(summaries: dict[int, dict]) -> list[str]:
    rows = [s for s in summaries.values() if s["selected"] is not None]
    if not rows:
        return []
    y = np.array([s["selected"] for s in rows])
    lines = ["", f"  How well each reading picks the studio's choices "
                 f"({int(y.sum()):,} selected of {len(y):,}; 0.50 = chance, 1.00 = perfect):"]

    def score(name, values, subset=None):
        v = np.array(values, dtype=float)
        keep = ~np.isnan(v) if subset is None else (~np.isnan(v) & subset)
        lines.append(f"    {name:<34} {auc(v[keep], y[keep]):.2f}   ({int(keep.sum()):,} photos)")

    nan = float("nan")
    score("has a face", [1.0 if s["faces"] else 0.0 for s in rows])
    score("size of the largest face", [s["largest"] for s in rows])
    score("eyes open (most closed main face)",
          [-s["worst_eyes"] if s["worst_eyes"] is not None else nan for s in rows])
    score("smile (main faces)", [s["smile"] if s["smile"] is not None else nan for s in rows])
    closed = [s for s in rows if s["closed"]]
    if closed:
        rejected = 1 - sum(s["selected"] for s in closed) / len(closed)
        lines.append(f"  Photos where a main face has closed eyes: {len(closed):,}; the studio rejected "
                     f"{rejected:.0%} of them (all photos: {1 - y.mean():.0%}).")
    return lines


def burst_lines(summaries: dict[int, dict]) -> list[str]:
    """Inside bursts with exactly one studio pick: is the pick the frame with the most open eyes?"""
    bursts = group_bursts((pid, s["body"], s["taken"]) for pid, s in summaries.items())
    hits = chance = used = 0
    for ids in bursts:
        frames = [summaries[i] for i in ids if summaries[i]["worst_eyes"] is not None
                  and summaries[i]["selected"] is not None]
        if len(frames) < 2 or sum(f["selected"] for f in frames) != 1:
            continue
        best = min(f["worst_eyes"] for f in frames)
        pick = next(f for f in frames if f["selected"])
        ties = sum(1 for f in frames if f["worst_eyes"] == best)
        hits += (pick["worst_eyes"] == best) / ties
        chance += 1 / len(frames)
        used += 1
    if not used:
        return []
    return [f"  Inside {used:,} bursts with one studio pick, the pick is the frame with the most "
            f"open eyes {hits / used:.0%} of the time (chance: {chance / used:.0%})."]


def review_page(shoot, cat: Catalog, n: int = 36) -> Path:
    """faces_review.html: the faces read as most closed and most open, with the studio's choice."""
    rows = list(cat.db.execute(
        "SELECT f.photo_id, f.n, f.x, f.y, f.w, f.h, MAX(f.blink_l, f.blink_r), f.smile, f.look_down, "
        "l.selected, p.path FROM faces f JOIN photos p ON p.id = f.photo_id "
        "LEFT JOIN labels l ON l.id = f.photo_id WHERE f.subject = 1 AND f.eyes IN ('open', 'closed') "
        "ORDER BY 7 DESC"))
    folder = shoot.cache / "review" / "faces"
    folder.mkdir(parents=True, exist_ok=True)

    def tiles(chosen) -> str:
        cells = []
        for pid, k, x, y, w, h, eyes, smile, down, sel, path in chosen:
            crop_file = folder / f"{pid:06d}_{k}.jpg"
            source = cache_path(shoot.cache, "analysis", pid)
            if not crop_file.exists() and not source.exists():
                continue                  # read on another computer: no copy here to cut from
            if not crop_file.exists():
                bgr = cv2.imdecode(np.fromfile(str(source), np.uint8), cv2.IMREAD_COLOR)
                cv2.imencode(".jpg", square_crop(bgr, (x, y, w, h), size=200)[0])[1].tofile(str(crop_file))
            verdict = {1: "studio: picked", 0: "studio: rejected"}.get(sel, "")
            cells.append(f'<figure><img src="faces/{crop_file.name}"><figcaption>eyes closed '
                         f'{eyes:.2f} &middot; smile {smile or 0:.2f} &middot; looking down {down or 0:.2f}'
                         f'<br><b>{verdict}</b><br>{html.escape(Path(path).name)}</figcaption></figure>')
        return "\n".join(cells)

    page = shoot.cache / "review" / "faces_review.html"
    page.write_text(f"""<!doctype html><meta charset="utf-8"><title>KeepSake faces review</title>
<style>body{{font:14px system-ui;margin:24px;background:#fafafa}} h2{{margin-top:32px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}} figure{{margin:0;width:200px}}
img{{width:200px;height:200px;border-radius:6px;display:block}} figcaption{{font-size:12px;color:#444}}</style>
<h1>Faces review: {html.escape(shoot.name)}</h1>
<p>Each face is the cut-out the eye reader saw. "Eyes closed" runs from 0 (wide open) to 1 (shut).</p>
<h2>Read as most closed</h2><div class="grid">{tiles(rows[:n])}</div>
<h2>Read as most open</h2><div class="grid">{tiles(rows[::-1][:n])}</div>
""", encoding="utf-8")
    return page


def main() -> int:
    args = parse_args()
    root = clean_path(args.folder)
    shoot = shoot_for(root)
    if not shoot.db.exists():
        print(f"This shoot has not been ingested yet. Run first:  ks ingest \"{root}\"")
        return 1
    try:
        probe = FaceReader()
    except FileNotFoundError as e:
        print(e)
        return 1
    eyes = probe.reads_eyes
    probe.close()
    if not eyes:
        print("MediaPipe is not installed, so faces are found but eyes are not read.")
    start = time.perf_counter()
    with Catalog(shoot.db) as cat:
        labelled = attach_labels(cat, root)
        if labelled:
            print(f"Studio choices: {labelled[0]:,} photos labelled from {labelled[1]}")
        if args.redo:
            cat.db.execute("DELETE FROM face_scan")
            cat.db.commit()
        todo = cat.faces_pending(FACES_VERSION, eyes)
        # each worker uses one CPU thread, so all hardware threads but two are used
        workers = args.workers or measured_workers("faces") or max(1, (os.cpu_count() or 2) - 2)
        if todo:
            print(f"Reading faces in {len(todo):,} photos with {workers} workers...")
            tasks = [(pid, str(cache_path(shoot.cache, "analysis", pid))) for pid in todo]
            bar, batch = Progress(len(tasks), label="Faces"), []

            def on_result(r: dict) -> None:
                batch.append(r)
                bar.add(None)
                if len(batch) >= 300:
                    cat.save_faces(batch, FACES_VERSION)
                    batch.clear()

            try:
                with Runner(workers, shoot.cache, fn=faces_task) as runner:
                    runner.run(tasks, on_result)
            finally:
                cat.save_faces(batch, FACES_VERSION)
                bar.finish()
        else:
            print("Every photo has been read already.")
        changed = cat.refresh_eyes(eye_state)
        if changed:
            print(f"Eye verdicts updated with the current rule: {changed:,} faces changed.")
        scanned, failed, with_faces = cat.db.execute(
            "SELECT COUNT(*), SUM(ok = 0), SUM(faces > 0) FROM face_scan").fetchone()
        n_faces, n_main, n_read = cat.db.execute(
            "SELECT COUNT(*), SUM(subject), SUM(read) FROM faces").fetchone()
        verdicts = dict(cat.db.execute(
            "SELECT eyes, COUNT(*) FROM faces WHERE subject = 1 AND eyes IS NOT NULL GROUP BY eyes"))
        errors = [e for (e,) in cat.db.execute("SELECT error FROM face_scan WHERE ok = 0 LIMIT 3")]
        summaries = photo_summaries(cat)
        lines = [f"KeepSake faces report - {shoot.name} - {now_text()}",
                 f"  Photos read         {scanned:,}" + (f"   ({failed} failed)" if failed else ""),
                 f"  Photos with faces   {with_faces or 0:,} ({(with_faces or 0) / max(scanned, 1):.0%})",
                 f"  Faces found         {n_faces:,}; main people {n_main or 0:,}; eyes read for {n_read or 0:,}",
                 "  Eyes of main people " + ", ".join(f"{k} {verdicts.get(k, 0):,}" for k in
                                                      ("open", "closed", "smiling", "unsure", "down", "side")),
                 f"  Time                {duration(time.perf_counter() - start)}"
                 + (f" ({len(todo) / (time.perf_counter() - start):.1f} photos/s)" if todo else "")]
        lines += [f"  Failed: {e}" for e in errors]
        lines += label_lines(summaries) + burst_lines(summaries)
        page = review_page(shoot, cat) if n_read else None
    text = "\n".join(lines)
    print("\n" + text)
    (shoot.cache / "faces_report.txt").write_text(text + "\n", encoding="utf-8")
    if page:
        print(f"\nReview page (open it in your browser): {page}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
        sys.exit(2)
