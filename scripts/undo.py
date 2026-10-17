r"""Phase 4: undo ks apply - remove the copies it made in the output folder.

    ks undo "C:\KeepSake Output\Sharma"

Only files listed in the folder's KeepSake manifest are removed, and only when their checksum
still matches (a copy someone has edited since is kept). The originals were never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.cli import clean_path  # noqa: E402
from engine.deliver import OutputError, undo  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print('Give the output folder, for example:  ks undo "C:\\KeepSake Output\\Sharma"')
        return 1
    out = clean_path(sys.argv[1])
    try:
        report = undo(out)
    except OutputError as e:
        print(e)
        return 1
    print(f"Removed {report.removed:,} copies from {out}.")
    if report.missing:
        print(f"{report.missing:,} listed copies were already gone.")
    if report.kept_changed:
        print(f"Kept {len(report.kept_changed):,} copies that were changed after copying, e.g. {report.kept_changed[0]}")
    print("The original photos were never touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
