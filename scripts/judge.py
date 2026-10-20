r"""Claude as a second editor, tested on 400 photos of a past wedding (paid: needs the API key).

    ks judge "Shevaun" --test                   400 photos of that wedding, Claude Opus
    ks judge "Shevaun" --test --model haiku     other models: haiku, sonnet, opus, fable (several: haiku,opus)
    ks judge "D:\All Raw images\Rishi & Aashi" --test   Rishi: the re-pick test (studio picked twice)
    options:  --now     no waiting, full price (default: batch, half price, usually a few minutes)
              --yes     do not ask before spending
              --max 3   never spend more than $3 in this run (default 3)
              --effort low|medium|high   how long Claude thinks (default low; higher costs more)
              --photos 400  how many test photos (only the first time for a wedding)
              --window  use a new 400-photo stretch even where a re-pick test exists
              --no-scene  only round 1 (each shot alone)

Round 1: each shot alone (engine/judge_prompt.md): best frame, blinks, turned heads, a 1-10 score.
Round 2: the whole sequence (engine/judge_scene_prompt.md): ~40 consecutive shots side by side,
each scored 0-100 the way the studio chooses (one photo per beat, best expression, family and
groups). Round 2 decides which shots are kept; round 1 decides which frame of each shot.
A shot whose chosen frame has a blink, a half blink, a turned head or a soft face (as Claude saw
it) is only kept when there are not enough clean shots.

The wedding can be its photo folder or part of its name. The test photos are one continuous
stretch of the wedding (all cameras, like ks testset), taken from the previews on this laptop, so
the drive is not needed. KeepSake decides them blind (learned only from the other weddings), and
they are kept in data\judge\<wedding>\test_photos.csv: every later run uses the same photos.
The test photos' small copies are sent, one request per shot (near-identical frames together).
Nothing is sent before the price is shown and you type yes; answers are kept on this laptop
(data\judge), so running it again costs nothing for shots already answered.

Three lists of the same size are compared with the studio's picks (for Rishi also its re-pick):
    KeepSake alone       ks cull's own choice among the test photos
    Claude               shots ranked by Claude's score, Claude's best frame
    KeepSake + Claude    shots ranked by both scores together, Claude's best frame
Output: C:\KeepSake Output\Claude test - <wedding>\
    Test photos (400), Studio selected / not selected, KeepSake selected / not selected
    <model>\Claude selected / not selected, KeepSake + Claude selected / not selected, Claude notes.csv
    Summary.txt
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine import judge  # noqa: E402
from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import CACHE_DIR, DATA_DIR  # noqa: E402
from engine.ingest import Shoot, cache_path, duration, shoot_for, slug  # noqa: E402
from engine.learn import photo_features  # noqa: E402
from engine.shots import group_shots  # noqa: E402

REPICK_DIR = DATA_DIR / "repick"
OUTPUT_ROOT = Path("C:/KeepSake Output")


def option(name: str, default: str | None = None) -> str | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def link_or_copy(src: Path, dst: Path) -> None:
    """A second name for the same file (no extra space), or a copy where that is not possible."""
    import os
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


def find_shoot(text: str) -> Shoot | None:
    """The prepared wedding for a photo folder, or for part of its name ("Shevaun")."""
    shoot = shoot_for(clean_path(text))
    if shoot.db.exists():
        return shoot
    want = slug(text)
    hits = [d.parent for d in sorted(CACHE_DIR.glob("*/catalog.db")) if want and want in d.parent.name]
    return Shoot(clean_path(text), hits[0].name, hits[0]) if len(hits) == 1 else None


def wedding_name(cat: Catalog, shoot: Shoot) -> str:
    root = Path(cat.get("root") or shoot.root)
    return root.parent.name if "raw" in root.name.lower() and root.parent.name else root.name


def repick_rows(cat: Catalog, shoot: Shoot, name: str):
    """The re-pick test (ks testset): its photos, KeepSake's last blind ks cull, the studio's re-pick."""
    info_file = REPICK_DIR / f"{shoot.name}.json"
    if info_file.exists():
        folder = Path(json.loads(info_file.read_text(encoding="utf-8"))["folder"])
    else:                                   # made by an older ks testset: look in the usual places
        candidates = [OUTPUT_ROOT / f"Re-pick test - {name}{s}" for s in ("", " (small)")]
        folder = next((c for c in candidates if c.is_dir()), candidates[0])
    if not folder.is_dir():
        return f"The test folder is missing: {folder}"
    settings = json.loads(cat.get("cull") or "{}")
    decision = dict(cat.db.execute("SELECT id, decision FROM cull"))
    score = dict(cat.db.execute("SELECT id, score FROM cull"))
    if not decision:
        return f'Run the cull first:  ks cull "{shoot.root}"'
    if settings.get("mode") != "other weddings" or shoot.name in settings.get("trained_on", []):
        return f'Not blind: run  ks cull "{shoot.root}"  again (it must learn only from the other weddings).'
    rows = list(csv.DictReader(open(REPICK_DIR / f"{shoot.name}.csv", encoding="utf-8")))
    for r in rows:
        r["src"] = folder / r["copy"]
    new_file = REPICK_DIR / f"{shoot.name}_new.txt"
    new = ({x.strip() for x in new_file.read_text(encoding="utf-8").splitlines() if x.strip()}
           if new_file.exists() else set())
    return rows, decision, score, new


def window_rows(cat: Catalog, shoot: Shoot, n: int):
    """A stretch of n photos of any past wedding (same rule as ks testset: all cameras, lined-up
    clocks, about the wedding's usual pick rate), KeepSake's blind choice (learned only from the
    other weddings) and the studio's choice. Uses the previews on this laptop: no drive needed.
    Kept in data\\judge\\<wedding>\\test_photos.csv, so every run uses the same photos."""
    from testset import choose_window
    from engine.cull import cull_shoot, labelled_shoots
    test_file = judge.JUDGE_DIR / shoot.name / "test_photos.csv"
    if not test_file.exists():
        if not cat.db.execute("SELECT COUNT(*) FROM labels WHERE selected = 1").fetchone()[0]:
            return "The studio's choices for this wedding are not known, so there is nothing to compare with."
        offsets = dict(cat.db.execute("SELECT body, offset FROM clocks WHERE offset IS NOT NULL"))
        photos = sorted(
            [(pid, body, taken + offsets[body], sel or 0, path) for pid, body, taken, sel, path in cat.db.execute(
                "SELECT p.id, COALESCE(p.body, p.camera, ''), p.taken_s, l.selected, p.path FROM photos p "
                "LEFT JOIN labels l ON l.id = p.id WHERE p.status = 'done' AND p.taken_s IS NOT NULL")
             if body in offsets], key=lambda r: r[2])
        n = min(n, len(photos))
        chosen = photos[choose_window(photos, n):][:n]
        others = labelled_shoots(CACHE_DIR, exclude=shoot.db)
        if not others:
            return "KeepSake needs other prepared weddings to learn from."
        print(f"Choosing {n} test photos and letting KeepSake decide them blind "
              f"(learning from {len(others)} other weddings, a few minutes)...")
        result = cull_shoot(cat, shoot.name, others, style="studio")
        dec = {r[0]: (r[1], r[3]) for r in result.rows}
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["copy", "original", "photo_id", "studio_selected", "keepsake", "keepsake_score"])
            for k, (pid, _body, _t, sel, path) in enumerate(chosen, 1):
                d, sc = dec.get(pid, ("removed", 0))
                w.writerow([f"{k:04d}_{Path(path).stem}.jpg", path, pid, sel, d, sc if sc is not None else 0])
    rows = list(csv.DictReader(open(test_file, encoding="utf-8")))
    for r in rows:
        r["src"] = cache_path(shoot.cache, "analysis", int(r["photo_id"]))
    decision = {int(r["photo_id"]): r["keepsake"] for r in rows}
    score = {int(r["photo_id"]): float(r["keepsake_score"] or 0) for r in rows}
    return rows, decision, score, set()


def main() -> int:
    values = {option(o) for o in ("--max", "--effort", "--photos")}
    # model names may arrive as "haiku,opus" or, through ks.cmd (which splits at commas), "haiku" "opus"
    words = [w for a in sys.argv[1:] for w in a.lower().split(",") if w.strip()]
    models = [w for w in words if w in judge.MODELS] if "--model" in sys.argv else ["opus"]
    args = [a for a in sys.argv[1:] if a.strip() and not a.startswith("--") and a not in values
            and not all(w in judge.MODELS for w in a.lower().split(",") if w)]
    if not args or "--test" not in sys.argv:
        print('Give a past wedding, for example:  ks judge "Shevaun" --test')
        return 1
    if not models:
        print(f"Unknown model. Use: {', '.join(judge.MODELS)}  (for example --model opus)")
        return 1
    limit = float(option("--max", "3").strip("$"))
    effort = option("--effort", "low")
    now = "--now" in sys.argv
    shoot = find_shoot(args[0])
    if shoot is None:
        print(f"No prepared wedding matches {args[0]!r}. Give its photo folder or part of its name, e.g. \"Shevaun\".")
        return 1
    try:
        api_key = judge.load_key()
    except judge.JudgeError as e:
        print(e)
        return 1
    with Catalog(shoot.db) as cat:
        name = wedding_name(cat, shoot)
        repick = REPICK_DIR / f"{shoot.name}.csv"
        if repick.exists() and "--window" not in sys.argv:
            loaded = repick_rows(cat, shoot, name)
        else:
            loaded = window_rows(cat, shoot, int(option("--photos", "400")))
        if isinstance(loaded, str):
            print(loaded)
            return 1
        key_rows, decision, score, new_picks = loaded
        ds = photo_features(cat, shoot.name)
    index = {pid: i for i, pid in enumerate(ds.ids)}
    rows = [r for r in key_rows if int(r["photo_id"]) in index and r["src"].exists()]
    if len(rows) < len(key_rows):
        print(f"  {len(key_rows) - len(rows)} test photos are missing; they are left out.")
    idx = [index[int(r["photo_id"])] for r in rows]
    shot_of = group_shots(ds.moment[idx], ds.clip[idx])
    members: dict[int, list[int]] = defaultdict(list)          # shot -> positions in rows, time order
    for k, s in enumerate(shot_of):
        members[int(s)].append(k)
    shown: dict[int, list[int]] = {}
    for s, ks in members.items():
        if len(ks) > judge.MAX_FRAMES:                            # long burst: KeepSake's best frames
            best = sorted(ks, key=lambda k: -(score.get(int(rows[k]["photo_id"])) or 0))[:judge.MAX_FRAMES]
            ks = sorted(best)
        shown[s] = ks
    shots = {s: [rows[k]["src"] for k in ks] for s, ks in shown.items()}
    rid_of = {s: judge.request_id([Path(f).name for f in files]) for s, files in shots.items()}  # as judge.prepare

    ks_list = {r["copy"] for r in rows if decision.get(int(r["photo_id"])) == "selected"}
    n = max(1, len(ks_list))
    scene_on = "--no-scene" not in sys.argv
    order = sorted(shots, key=lambda s: members[s][0])                 # shots in time order
    groups = [order[i:i + judge.SCENE_SIZE] for i in range(0, len(order), judge.SCENE_SIZE)]
    keep_of = [max(1, round(n * sum(len(members[s]) for s in g) / len(rows))) for g in groups]

    def frames_chosen(answers: dict) -> dict[int, int]:
        """Each shot's chosen frame (position in rows): Claude's best frame, else KeepSake's."""
        return {s: shown[s][judge.best_frame(answers[s], len(shown[s]))] if s in answers
                else max(members[s], key=lambda k: score.get(int(rows[k]["photo_id"])) or 0) for s in order}

    def scene_job(j):
        answers = {s: j.done[rid_of[s]] for s in order if rid_of[s] in j.done}
        chosen = frames_chosen(answers)
        scenes = {gi: ([rows[chosen[s]]["src"] for s in g], keep_of[gi]) for gi, g in enumerate(groups)}
        return judge.prepare_scenes(scenes, j.model, judge.JUDGE_DIR / shoot.name), scenes

    print(f"Test photos: {len(rows)}, in {len(shots)} shots. Preparing small copies...")
    jobs = [judge.prepare(shots, m, judge.JUDGE_DIR / shoot.name, effort) for m in models]
    total = 0.0
    for j in jobs:
        cost = j.estimate(batch=not now)
        line = f"  {judge.NAMES[j.model]:<11} round 1 (each shot): {len(j.todo):>4} to send, {len(j.done)} answered before   about ${cost:.2f}"
        if scene_on:
            if not j.todo:
                sj, _ = scene_job(j)
                c2 = sj.estimate(batch=not now)
                line += f"\n  {'':<11} round 2 (whole sequence): {len(sj.todo):>3} sequences to send   about ${c2:.2f}"
            else:
                c2 = judge.scene_estimate(j.model, len(order), len(groups), batch=not now)
                line += f"\n  {'':<11} round 2 (whole sequence): {len(groups):>3} sequences   about ${c2:.2f}"
            cost += c2
        total += cost
        print(line)
    print(f"  Total about ${total:.2f} ({'full price, now' if now else 'batch, half price'}; "
          "Claude's thinking can add a little)")
    if total > limit:
        print(f"That is more than the limit of ${limit:.2f}. Use fewer models (--model opus) or a higher --max.")
        return 1
    if total > 0.005 and "--yes" not in sys.argv:
        if input("Type yes to send: ").strip().lower() != "yes":
            print("Nothing sent.")
            return 1
    start = time.perf_counter()
    spent = {j.model: 0.0 for j in jobs}

    def run(batch_jobs) -> None:
        waiting = [j for j in batch_jobs if j.todo]
        if not waiting:
            return
        if now:
            for j in waiting:
                spent[j.model] += judge.send_now(j, api_key)
        else:
            ids = {j.model: judge.submit(j, api_key) for j in waiting}
            for model, cost in judge.collect(waiting, ids, api_key).items():
                spent[model] += cost
    scene_jobs = {}
    try:
        run(jobs)
        if scene_on:
            for j in jobs:
                scene_jobs[j.model] = scene_job(j)
            if any(sj.todo for sj, _ in scene_jobs.values()):
                print("Round 2: Claude compares the shots of each sequence...")
            run([sj for sj, _ in scene_jobs.values()])
    except judge.JudgeError as e:
        print(e)
        print("Answers received so far are saved; run the same command again to continue.")
        return 1

    copies = [r["copy"] for r in rows]
    old = {r["copy"] for r in rows if r["studio_selected"] == "1"}
    new = new_picks & set(copies)
    both_times = old & new
    shot_by_copy = {copies[k]: int(shot_of[k]) for k in range(len(rows))}
    shot_ids = sorted(shots)
    ks_shot = np.array([max(score.get(int(rows[k]["photo_id"])) or 0 for k in members[s]) for s in shot_ids])

    out_root = OUTPUT_ROOT / f"Claude test - {name}"
    src_of = {r["copy"]: r["src"] for r in rows}
    every = [r["copy"] for r in rows]
    test_dir = out_root / f"Test photos ({len(every)})"
    if not test_dir.exists():
        test_dir.mkdir(parents=True)
        for c in every:
            shutil.copy2(src_of[c], test_dir / c)

    def selected_and_not(where: Path, label: str, chosen: set[str]) -> None:
        """'<label> selected (N)' and '<label> not selected (M)': second names for the test photos
        (no extra space on the laptop)."""
        for kind, members in (("selected", [c for c in every if c in chosen]),
                              ("not selected", [c for c in every if c not in chosen])):
            for old_dir in where.glob(f"{label} {kind} (*)"):
                shutil.rmtree(old_dir)
            target = where / f"{label} {kind} ({len(members)})"
            target.mkdir(parents=True)
            for c in members:
                link_or_copy(test_dir / c, target / c)
    head = (f"{'':<26}{'chose':>6}{'old picks' if new else 'studio picks':>13}{'same shot':>11}"
            + (f"{'new picks':>11}{'same shot':>11}{'both times':>12}" if new else "") + f"{'rule breaks':>13}")
    lines = [f"Claude test - {name} - {now_text()}",
             f"{len(rows)} test photos, {len(shots)} shots. The studio picked {len(old)}"
             + (f" before, {len(new)} in the re-pick, {len(both_times)} both times." if new else "."),
             f"Each list has {n} photos (KeepSake's own number). 'same' = the very same photo; 'same shot' = the "
             "same shot (a near-identical frame); rule breaks = eyes closed or half closed, a turned head or a "
             "soft face in the chosen frame, as Claude sees it.", ""]

    def row(label: str, chosen: set[str], answers: dict[int, dict]) -> str:
        def same_shot(picks):
            return len({shot_by_copy[c] for c in chosen} & {shot_by_copy[c] for c in picks})
        breaks = 0
        for c in chosen:
            s = shot_by_copy[c]
            ans = answers.get(s)
            ks = shown[s]
            pos = [rows[k]["copy"] for k in ks]
            if ans is not None and c in pos and judge.breaks_rules(judge.frame_flags(ans, pos.index(c) + 1)):
                breaks += 1
        return (f"{label:<26}{len(chosen):>6}{len(chosen & old):>13}{same_shot(old):>11}"
                + (f"{len(chosen & new):>11}{same_shot(new):>11}{len(chosen & both_times):>12}" if new else "")
                + f"{breaks:>13}")

    selected_and_not(out_root, "Studio", old)
    selected_and_not(out_root, "KeepSake", ks_list)
    for j in jobs:
        answers = {s: j.done[rid_of[s]] for s in shot_ids if rid_of[s] in j.done}
        claude = np.array([float(answers[s]["deliver_score"]) if s in answers else 5.0 for s in shot_ids])
        frame_of = {s: rows[k]["copy"] for s, k in frames_chosen(answers).items()}
        keep, why = {}, {}                                   # round 2: shot -> 0-100, a few words
        if j.model in scene_jobs:
            sj, scenes = scene_jobs[j.model]
            for gi, g in enumerate(groups):
                ans = sj.done.get(judge.scene_rid(scenes[gi][0], scenes[gi][1]))
                for item in (ans or {}).get("shots", []):
                    if 1 <= item.get("shot", 0) <= len(g):
                        keep[g[item["shot"] - 1]] = float(item.get("keep", 0))
                        why[g[item["shot"] - 1]] = item.get("why", "")
        scene = np.array([keep.get(s, claude[i] * 10) for i, s in enumerate(shot_ids)])
        # her rule: no blink, half blink, turned head or soft face in Selected. A shot whose chosen
        # frame breaks it (as Claude saw it) goes after every clean shot; it is only used if too few are clean.
        chosen_k = frames_chosen(answers)
        broken = np.array([s in answers and chosen_k[s] in shown[s] and bool(judge.breaks_rules(
            judge.frame_flags(answers[s], shown[s].index(chosen_k[s]) + 1))) for s in shot_ids])
        by_claude = sorted(range(len(shot_ids)), key=lambda i: (broken[i], -claude[i], -ks_shot[i]))[:n]
        by_scene = sorted(range(len(shot_ids)), key=lambda i: (broken[i], -scene[i], -claude[i], -ks_shot[i]))[:n]
        main = scene if keep else claude
        both = zscore(main) + zscore(ks_shot)
        by_both = sorted(range(len(shot_ids)), key=lambda i: (broken[i], -both[i]))[:n]
        lists = {"Claude round 1": {frame_of[shot_ids[i]] for i in by_claude},
                 "Claude selected": {frame_of[shot_ids[i]] for i in (by_scene if keep else by_claude)},
                 "KeepSake + Claude selected": {frame_of[shot_ids[i]] for i in by_both}}
        model_name = judge.NAMES[j.model]
        paid = judge.spent_so_far(j) + (judge.spent_so_far(scene_jobs[j.model][0]) if j.model in scene_jobs else 0)
        lines += [f"{model_name}: {len(answers)} of {len(shots)} shots answered"
                  + (f", {len(keep)} scored in round 2" if keep else "")
                  + f"; cost about ${paid:.2f} (this run ${spent[j.model]:.2f})",
                  head, row("KeepSake alone", ks_list, answers),
                  row("Claude, each shot alone", lists["Claude round 1"], answers)]
        if keep:
            lines.append(row("Claude, whole sequence", lists["Claude selected"], answers))
        lines += [row("KeepSake + Claude", lists["KeepSake + Claude selected"], answers), ""]

        folder_m = out_root / model_name
        selected_and_not(folder_m, "Claude", lists["Claude selected"])
        selected_and_not(folder_m, "KeepSake + Claude", lists["KeepSake + Claude selected"])
        with open(folder_m / "Claude notes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "shot", "claude score (round 1)", "claude best frame", "eyes", "head turned",
                        "sharp", "problems", "claude reason", "sequence score (round 2)", "sequence note",
                        "studio old", "studio new", "keepsake alone", "claude selected",
                        "keepsake + claude selected"])
            for s in shot_ids:
                ans = answers.get(s, {})
                pos = [rows[k]["copy"] for k in shown[s]]
                for k in members[s]:
                    c = rows[k]["copy"]
                    fl = judge.frame_flags(ans, pos.index(c) + 1) if c in pos else {}
                    w.writerow([c, s, ans.get("deliver_score", ""), int(frame_of[s] == c), fl.get("eyes", ""),
                                fl.get("head_turned", ""), fl.get("main_face_sharp", ""), fl.get("problems", ""),
                                ans.get("reason", "") if frame_of[s] == c else "",
                                keep.get(s, "") if frame_of[s] == c else "", why.get(s, "") if frame_of[s] == c else "",
                                int(c in old), int(c in new),
                                int(c in ks_list), int(c in lists["Claude selected"]),
                                int(c in lists["KeepSake + Claude selected"])])
    if new:
        lines.append(f"For comparison, the studio itself: its re-pick had {len(both_times)} of its {len(old)} old "
                     f"picks ({len(both_times) / max(len(old), 1):.0%}).")
    lines.append(f"Time {duration(time.perf_counter() - start)}")
    text = "\n".join(lines)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "Summary.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n\n")
    print("\n" + text)
    print(f"\nFolders: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
