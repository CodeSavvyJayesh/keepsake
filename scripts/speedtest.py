r"""Speed test for this laptop: the best number of workers for the slow reading steps.

    ks speedtest            about 5 minutes; uses photos KeepSake has already read

Measuring, faces and CLIP are each timed on the same photos with a few worker counts (more
workers use the processor's extra threads). The fastest is saved in data\speed.json and used
from then on by ks prepare, ks quality / faces / clip and the app. Nothing is written to any
wedding and no result changes: the same work is done, only shared out differently.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import now_text  # noqa: E402
from engine.clip import BATCH, clip_task  # noqa: E402
from engine.config import CACHE_DIR, SPEED_FILE, worker_count  # noqa: E402
from engine.faces import faces_task  # noqa: E402
from engine.ingest import Runner, cache_path, copies_here  # noqa: E402
from engine.quality import measure_task  # noqa: E402

PHOTOS = 480


def sample(n: int | None = None) -> tuple[Path, list[int]] | None:
    """n photos spread over the biggest wedding read so far (their analysis copies exist)."""
    n = n or PHOTOS
    best = None
    for db in CACHE_DIR.glob("*/catalog.db"):
        try:
            con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
            ids = [r[0] for r in con.execute("SELECT id FROM photos WHERE status = 'done' ORDER BY id")]
            con.close()
        except sqlite3.Error:
            continue
        here = copies_here(db.parent, "analysis")          # a catalog from another laptop has none
        ids = [pid for pid in ids if pid in here]
        if best is None or len(ids) > len(best[1]):
            best = (db.parent, ids)
    if best is None:
        return None
    cache, ids = best
    if len(ids) < n:
        return None
    step = len(ids) / n
    return cache, [ids[int(k * step)] for k in range(n)]


def rate(cache: Path, fn, tasks: list, workers: int, photos_per_task: int) -> float:
    """Photos per second once every worker is busy (the start-up of the workers is not counted)."""
    times = []
    with Runner(workers, cache, fn=fn) as runner:
        runner.run(tasks, lambda _r: times.append(time.perf_counter()))
    skip = min(len(times) // 4, workers)
    return (len(times) - 1 - skip) * photos_per_task / max(times[-1] - times[skip], 1e-6)


def main() -> int:
    found = sample()
    if not found:
        print(f"Read a wedding first (ks prepare): the speed test needs {PHOTOS} photos KeepSake has read.")
        return 1
    cache, ids = found
    logical = os.cpu_count() or 2
    physical = worker_count() + 1
    paths = [(pid, str(cache_path(cache, "analysis", pid))) for pid in ids]
    chunks = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
    steps = {     # step: (task function, tasks, photos per task, worker counts to try, default today)
        "quality": (measure_task, paths, 1, {physical - 1, (physical + logical) // 2 - 1, logical - 1}, physical - 1),
        "faces": (faces_task, paths, 1, {physical - 1, (physical + logical) // 2 - 1, logical - 2}, logical - 2),
        "clip": (clip_task, [(c[0][0], c) for c in chunks], BATCH,
                 {max(1, physical // 2), max(1, logical // 4 + 1), max(1, logical // 2 - 1)}, max(1, physical // 2)),
    }
    print(f"Speed test on {len(ids)} photos of {cache.name} ({physical} cores, {logical} threads). About 5 minutes...")
    saved = {"measured": now_text(), "cores": physical, "threads": logical}
    lines = []
    for step, (fn, tasks, per, options, today) in steps.items():
        results = {}
        for workers in sorted(w for w in options if w >= 1):
            results[workers] = rate(cache, fn, tasks, workers, per)
            print(f"  {step:<8} {workers:>2} workers   {results[workers]:6.1f} photos/s", flush=True)
        top = max(results.values())
        best = min(w for w, r in results.items() if r >= 0.97 * top)    # as fast, and lighter on the laptop
        saved[step] = best
        base = results.get(today)
        gain = f"{top / base - 1:+.0%} vs before" if base else ""
        lines.append(f"  {step:<8} best {best} workers, {results[best]:.1f} photos/s  {gain}")
    SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEED_FILE.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print("\nSaved (used from now on):\n" + "\n".join(lines) + f"\n  {SPEED_FILE}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
