r"""Phase 3: moments and people for an ingested shoot.

    ks groups "E:\Sharma Wedding\Raw"

Needs ks clip (for moments) and ks faces (for people) to have run. Lines up the camera
clocks, groups photos of the same moment across all cameras, finds the people who appear most
(the couple), and writes a review page. Works from the laptop copies; no drive needed.
"""
from __future__ import annotations

import html
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.faces import square_crop  # noqa: E402
from engine.ingest import cache_path, duration, shoot_for  # noqa: E402
from engine.labels import attach_labels  # noqa: E402
from engine.moments import Photo, align_clocks, group_moments, group_quality  # noqa: E402
from engine.people import FaceRef, cluster_people, frontal_from_points  # noqa: E402


def load_photos(cat: Catalog):
    ids, vectors = cat.clip_vectors()
    meta = {pid: (body, taken, folder) for pid, body, taken, folder in cat.db.execute(
        "SELECT id, COALESCE(body, camera, ''), taken_s, folder FROM photos WHERE status = 'done'")}
    photos = [Photo(pid, meta[pid][0], meta[pid][1], (meta[pid][2] or "").split("/")[0]) for pid in ids]
    return photos, vectors


def load_faces(cat: Catalog) -> list[FaceRef]:
    faces = []
    for pid, n, w, emb, lm, sharp, contrast in cat.db.execute(
            "SELECT photo_id, n, w, emb, lm, sharp, contrast FROM faces WHERE emb IS NOT NULL"):
        frontal = frontal_from_points(np.frombuffer(lm, np.float32)) if lm else 1.0
        faces.append(FaceRef(pid, n, w, np.frombuffer(emb, np.float16).astype(np.float32),
                             focus=(sharp or 0) / ((contrast or 0) ** 2 + 10), frontal=frontal))
    return faces


def review_page(shoot, cat: Catalog, groups: list[list[int]], selected: dict[int, int]) -> Path:
    folder = shoot.cache / "review"
    crops = folder / "people"
    crops.mkdir(parents=True, exist_ok=True)
    people_html = []
    for person in range(1, 9):
        rows = list(cat.db.execute(
            "SELECT f.photo_id, f.n, f.x, f.y, f.w, f.h FROM faces f WHERE f.person = ? "
            "ORDER BY f.w DESC LIMIT 8", (person,)))
        n_photos, n_picks = cat.db.execute(
            "SELECT COUNT(DISTINCT f.photo_id), COUNT(DISTINCT CASE WHEN l.selected = 1 THEN f.photo_id END) "
            "FROM faces f LEFT JOIN labels l ON l.id = f.photo_id WHERE f.person = ?", (person,)).fetchone()
        imgs = []
        for pid, n, x, y, w, h in rows:
            out = crops / f"p{person}_{pid:06d}_{n}.jpg"
            if not out.exists():
                bgr = cv2.imdecode(np.fromfile(str(cache_path(shoot.cache, "analysis", pid)), np.uint8),
                                   cv2.IMREAD_COLOR)
                cv2.imencode(".jpg", square_crop(bgr, (x, y, w, h), margin=1.5, size=140)[0])[1].tofile(str(out))
            imgs.append(f'<img src="people/{out.name}">')
        picks = f", in {n_picks:,} of the studio's picks" if any(selected.values()) else ""
        people_html.append(f"<h3>Person {person}: in {n_photos:,} photos{picks}</h3><div class=row>{''.join(imgs)}</div>")
    rng = np.random.default_rng(1)
    multi = [g for g in groups if len(g) >= 3]
    shown = [multi[i] for i in sorted(rng.choice(len(multi), min(12, len(multi)), replace=False))] if multi else []
    moments_html = []
    for g in shown:
        cells = []
        for pid in g[:12]:
            mark = ' class="pick"' if selected.get(pid) else ""
            thumb = cache_path(shoot.cache, "thumbs", pid).relative_to(shoot.cache).as_posix()
            cells.append(f'<img{mark} src="../{thumb}">')
        more = f"<span>+{len(g) - 12} more</span>" if len(g) > 12 else ""
        moments_html.append(f"<h3>Moment of {len(g)} photos</h3><div class=row>{''.join(cells)}{more}</div>")
    page = folder / "groups_review.html"
    page.write_text(f"""<!doctype html><meta charset="utf-8"><title>KeepSake people and moments</title>
<style>body{{font:14px system-ui;margin:24px;background:#fafafa}} .row{{display:flex;flex-wrap:wrap;gap:6px}}
img{{height:140px;border-radius:6px}} img.pick{{outline:4px solid #2f6fde}} h3{{margin:20px 0 6px}}</style>
<h1>{html.escape(shoot.name)}</h1>
<h2>The most photographed people</h2><p>Each row should be one person. Persons 1 and 2 are usually the couple.</p>
{''.join(people_html)}
<h2>Some moments</h2><p>Each row is one group of photos of the same moment. Blue outline = the studio picked it.</p>
{''.join(moments_html)}""", encoding="utf-8")
    return page


def main() -> int:
    if len(sys.argv) < 2:
        print('Give the shoot folder, for example:  ks groups "E:\\Sharma Wedding\\Raw"')
        return 1
    root = clean_path(sys.argv[1])
    shoot = shoot_for(root)
    if not shoot.db.exists():
        print(f"This shoot has not been ingested yet. Run first:  ks ingest \"{root}\"")
        return 1
    start = time.perf_counter()
    with Catalog(shoot.db) as cat:
        attach_labels(cat, root)
        selected = dict(cat.db.execute("SELECT id, selected FROM labels"))
        photos, vectors = load_photos(cat)
        if not photos:
            print(f"No CLIP embeddings yet. Run first:  ks clip \"{root}\"")
            return 1
        print(f"Lining up camera clocks for {len(photos):,} photos...")
        alignment = align_clocks(photos, vectors)
        groups = group_moments(photos, vectors, alignment)
        cat.save_moments(groups, alignment.offsets, alignment.support)
        faces = load_faces(cat)
        print(f"Finding people among {len(faces):,} face fingerprints...")
        assignment = cluster_people(faces)
        cat.save_people(assignment)
        page = review_page(shoot, cat, groups, selected)

    counts = Counter(alignment.support)
    lines = [f"KeepSake moments and people - {shoot.name} - {now_text()}", "", "  Camera clocks (lined up to "
             f"{alignment.reference}):"]
    for body in sorted(alignment.support, key=lambda b: -counts[b]):
        offset = alignment.offsets.get(body)
        shown = "reference" if body == alignment.reference else (
            f"{offset / 3600:+.2f} h" if offset is not None and abs(offset) >= 3600 else
            f"{offset:+.0f} s" if offset is not None else "unknown (kept on its own)")
        lines.append(f"    {body[:38]:<40}{shown:<26}({alignment.support[body]} matching moments)")
    q = group_quality(groups, selected)
    lines += ["", f"  Moments             {q['groups']:,} from {q['photos']:,} photos "
                  f"({q['multi_groups']:,} with 2+ photos, holding {q['in_multi']:,} photos; largest {q['largest']})"]
    if q["picks"]:
        lines.append(f"  Studio picks        {q['picks']:,} in {q['groups_with_pick']:,} moments; keeping just the "
                     f"best photo of every moment could keep at most {q['picks_if_one_per_group']:,} of them")
    people = Counter(person for person in assignment.values())
    by_photo = defaultdict(set)
    for (pid, _), person in assignment.items():
        by_photo[person].add(pid)
    lines += ["", f"  People found        {len(people):,} (from {len(assignment):,} faces)"]
    for person in range(1, 6):
        if person in by_photo:
            in_photos = by_photo[person]
            picks = sum(selected.get(p, 0) for p in in_photos)
            rate = f"; studio kept {picks / len(in_photos):.0%} of them" if selected else ""
            lines.append(f"    Person {person}: in {len(in_photos):,} photos{rate}")
    lines += ["", f"  Time                {duration(time.perf_counter() - start)}"]
    text = "\n".join(lines)
    print("\n" + text)
    (shoot.cache / "groups_report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nReview page (open it in your browser): {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
