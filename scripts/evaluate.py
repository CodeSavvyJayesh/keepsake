r"""Phase 3: how well KeepSake has learned the studio's taste, tested honestly.

    ks evaluate "E:\Sharma Wedding\Raw" ["F:\Mehta Wedding\Raw" ...]

Every shoot must be ingested and analysed (ks quality, ks faces, ks clip, ks groups) and have
the studio's choices (ks match). With several weddings, each is tested by a model trained only
on the others; with one wedding, each event is tested by a model trained on the other events.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import DATA_DIR  # noqa: E402
from engine.ingest import duration, shoot_for  # noqa: E402
from engine.labels import attach_labels  # noqa: E402
from engine.learn import GBM, Logistic, combine, leave_one_out, lightgbm_available, photo_features  # noqa: E402


def table(name: str, scores) -> list[str]:
    lines = [f"  {name}", f"    {'tested on':<44}{'AUC':>6}{'top 20%':>9}{'top 30%':>9}{'top 50%':>9}"
             f"{'for 90%':>9}{'for 95%':>9}"]
    for s in scores:
        lines.append(f"    {s.test[:43]:<44}{s.auc:6.2f}{s.caught[0.2]:9.0%}{s.caught[0.3]:9.0%}"
                     f"{s.caught[0.5]:9.0%}{s.needed[0.9]:9.0%}{s.needed[0.95]:9.0%}")
    if len(scores) > 1:
        m = lambda f: float(np.mean([f(s) for s in scores]))
        lines.append(f"    {'average':<44}{m(lambda s: s.auc):6.2f}{m(lambda s: s.caught[0.2]):9.0%}"
                     f"{m(lambda s: s.caught[0.3]):9.0%}{m(lambda s: s.caught[0.5]):9.0%}"
                     f"{m(lambda s: s.needed[0.9]):9.0%}{m(lambda s: s.needed[0.95]):9.0%}")
    return lines


def main() -> int:
    folders = [clean_path(a) for a in sys.argv[1:] if not a.startswith("--")]
    if not folders:
        print('Give one or more shoot folders, for example:  ks evaluate "E:\\Sharma Wedding\\Raw"')
        return 1
    start = time.perf_counter()
    datasets = []
    for root in folders:
        shoot = shoot_for(root)
        if not shoot.db.exists():
            print(f"Not ingested yet: {root}")
            return 1
        with Catalog(shoot.db) as cat:
            attach_labels(cat, root)
            ds = photo_features(cat, shoot.name)
        known = int((ds.y >= 0).sum())
        print(f"{shoot.name}: {len(ds.ids):,} photos, {known:,} with the studio's choice, "
              f"{int((ds.y == 1).sum()):,} picks" + ("" if ds.clip is not None else "  (no CLIP yet: run ks clip)"))
        if known:
            datasets.append(ds)
    if not datasets:
        print("No photos with the studio's choices. Run ks match on the wedding first.")
        return 1
    by_wedding = len(datasets) > 1
    data = combine(datasets, by_wedding)
    unit = "wedding" if by_wedding else "event"
    lines = [f"KeepSake accuracy - {now_text()}",
             f"  Each {unit} is tested by a model trained only on the other {unit}s.",
             f"  {len(data.names)} features per photo.",
             "  AUC: 0.50 = chance, 1.00 = perfect. 'top 20%': share of the studio's picks among the",
             "  20% of photos the model rates highest. 'for 95%': shortlist size needed to catch 95%.", ""]
    scores, _ = leave_one_out(data, Logistic)
    lines += table("Logistic regression (simple baseline)", scores)
    if lightgbm_available():
        scores, model = leave_one_out(data, GBM)
        lines += [""] + table("LightGBM (main model)", scores)
        if model is not None:
            lines += ["", "  What the model relies on most (share of its decisions):"]
            lines += [f"    {name:<36}{share:6.1%}" for name, share in model.importance(data.names)]
    else:
        lines += ["", "  LightGBM is not installed: run  .venv\\Scripts\\python.exe -m pip install -r requirements.txt"]
    lines += ["", f"  Time {duration(time.perf_counter() - start)}"]
    text = "\n".join(lines)
    print("\n" + text)
    out = DATA_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"accuracy_{time.strftime('%Y-%m-%d_%H%M')}.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
