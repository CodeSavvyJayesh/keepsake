r"""Phase 2: measure sharpness and exposure of every photo of an ingested shoot.

    ks quality "E:\Sharma Wedding\Raw"

Works from the analysis copies on this laptop, so the shoot drive need not be plugged in.
If the studio's choices are known (ks match), it also reports how well each measurement
matches them. Results go in the shoot's catalog (quality table).
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import measured_workers, worker_count  # noqa: E402
from engine.ingest import Progress, Runner, cache_path, duration, shoot_for  # noqa: E402
from engine.labels import attach_labels  # noqa: E402
from engine.quality import QUALITY_VERSION, auc, measure_task, relative_scores  # noqa: E402

MEASURES = [   # (column, what it means, higher is better?)
    ("sharp_global", "sharpness, whole frame", True),
    ("sharp_peak", "sharpness, sharpest part", True),
    ("sharp_rel", "sharpness vs same camera", True),
    ("luma_mean", "brightness", True),
    ("luma_rel", "brightness vs same event", True),
    ("clip_high", "blown highlights", False),
    ("clip_low", "crushed shadows", False),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ks quality",
                                 description="Measure sharpness and exposure of an ingested shoot.")
    ap.add_argument("folder", help="the shoot folder you ingested, in quotes")
    ap.add_argument("--workers", type=int, help="number of worker processes")
    ap.add_argument("--redo", action="store_true", help="measure every photo again")
    return ap.parse_args()


def label_report(cat: Catalog) -> list[str]:
    rows = cat.labelled_quality()
    if not rows:
        return []
    y = np.array([r["selected"] for r in rows])
    lines = ["", f"  How well each measurement picks the studio's choices "
                 f"({int(y.sum()):,} selected of {len(y):,}):",
             "  (0.50 = no better than chance, 1.00 = perfect)"]
    for col, name, higher in MEASURES:
        v = np.array([r[col] if r[col] is not None else np.nan for r in rows], dtype=float)
        ok = ~np.isnan(v)
        score = auc(v[ok] if higher else -v[ok], y[ok])
        lines.append(f"    {name:<28} {score:.2f}")
    rel = np.array([r["sharp_rel"] if r["sharp_rel"] is not None else 0 for r in rows], dtype=float)
    soft = rel <= np.percentile(rel, 5)
    lines.append(f"  The softest 5% of photos ({int(soft.sum()):,}): {1 - y[soft].mean():.0%} were "
                 f"rejected by the studio, against {1 - y.mean():.0%} of all photos.")
    return lines


def main() -> int:
    args = parse_args()
    root = clean_path(args.folder)
    shoot = shoot_for(root)
    if not shoot.db.exists():
        print(f"This shoot has not been ingested yet. Run first:  ks ingest \"{root}\"")
        return 1
    start = time.perf_counter()
    with Catalog(shoot.db) as cat:
        labelled = attach_labels(cat, root)
        if labelled:
            print(f"Studio choices: {labelled[0]:,} photos labelled from {labelled[1]}")
        if args.redo:
            cat.db.execute("DELETE FROM quality")
            cat.db.commit()
        todo = cat.quality_pending(QUALITY_VERSION)
        workers = args.workers or measured_workers("quality") or worker_count()
        if todo:
            print(f"Measuring {len(todo):,} photos with {workers} workers...")
            tasks = [(pid, str(cache_path(shoot.cache, "analysis", pid))) for pid, _ in todo]
            bar, batch = Progress(len(tasks), label="Measuring"), []

            def on_result(r: dict) -> None:
                batch.append(r)
                bar.add(None)
                if len(batch) >= 500:
                    cat.save_quality(batch, QUALITY_VERSION)
                    batch.clear()

            try:
                with Runner(workers, shoot.cache, fn=measure_task) as runner:
                    runner.run(tasks, on_result)
            finally:
                cat.save_quality(batch, QUALITY_VERSION)
                bar.finish()
        else:
            print("Every photo is already measured.")
        rows = cat.quality_rows()
        cat.save_relative(relative_scores(rows))
        failed = cat.db.execute("SELECT COUNT(*) FROM quality WHERE ok = 0").fetchone()[0]
        lines = [f"KeepSake quality report - {shoot.name} - {now_text()}",
                 f"  Photos measured     {len(rows):,}" + (f"   ({failed} could not be read)" if failed else ""),
                 f"  Time                {duration(time.perf_counter() - start)}"]
        lines += label_report(cat)
    text = "\n".join(lines)
    print("\n" + text)
    (shoot.cache / "quality_report.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
        sys.exit(2)
