r"""Phase 3: CLIP embedding for every photo of an ingested shoot (what each photo shows).

    ks clip "E:\Sharma Wedding\Raw"

Works from the analysis copies on this laptop (the shoot drive need not be plugged in).
Needs the CLIP model (ks models) and ONNX Runtime (pip install -r requirements.txt).
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.clip import BATCH, CLIP_VERSION, clip_task  # noqa: E402
from engine.config import measured_workers, worker_count  # noqa: E402
from engine.ingest import Progress, Runner, cache_path, duration, shoot_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="ks clip", description="CLIP embedding for every photo.")
    ap.add_argument("folder", help="the shoot folder you ingested, in quotes")
    ap.add_argument("--workers", type=int, help="number of worker processes (2 threads each)")
    args = ap.parse_args()
    root = clean_path(args.folder)
    shoot = shoot_for(root)
    if not shoot.db.exists():
        print(f"This shoot has not been ingested yet. Run first:  ks ingest \"{root}\"")
        return 1
    start = time.perf_counter()
    with Catalog(shoot.db) as cat:
        todo = cat.clip_pending(CLIP_VERSION)
        workers = args.workers or measured_workers("clip") or max(1, (worker_count() + 1) // 2)
        failed = 0
        if todo:
            print(f"CLIP for {len(todo):,} photos with {workers} workers...")
            chunks = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
            tasks = [(c[0], [(pid, str(cache_path(shoot.cache, "analysis", pid))) for pid in c])
                     for c in chunks]
            bar, batch = Progress(len(todo), label="CLIP"), []

            def on_result(r: dict) -> None:
                nonlocal failed
                items = r.get("items")
                if items is None:                  # the whole chunk crashed its worker
                    chunk = next(c for c in chunks if c[0] == r["id"])
                    items = [{"id": pid, "ok": False, "error": r.get("error")} for pid in chunk]
                batch.extend(items)
                for item in items:
                    failed += not item["ok"]
                    bar.add(None)
                if len(batch) >= 512:
                    cat.save_clip(batch, CLIP_VERSION)
                    batch.clear()

            try:
                with Runner(workers, shoot.cache, fn=clip_task) as runner:
                    runner.run(tasks, on_result)
            finally:
                cat.save_clip(batch, CLIP_VERSION)
                bar.finish()
        else:
            print("Every photo already has its CLIP embedding.")
        ids, _ = cat.clip_vectors()
        errors = [e for (e,) in cat.db.execute("SELECT error FROM clip WHERE ok = 0 LIMIT 3")]
    secs = time.perf_counter() - start
    print(f"\nKeepSake CLIP - {shoot.name} - {now_text()}")
    print(f"  Photos with an embedding  {len(ids):,}" + (f"   ({failed} failed)" if failed else ""))
    print(f"  Time                      {duration(secs)}"
          + (f" ({len(todo) / secs:.1f} photos/s)" if todo else ""))
    for e in errors:
        print(f"  Failed: {e}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
        sys.exit(2)
