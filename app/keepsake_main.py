r"""The packaged KeepSake app (the Mac app; later a Windows installer) starts here.

    KeepSake                          the app: its window and the small server behind it
    KeepSake --run <step> [args...]   one step, exactly as `ks <step>` runs it (prepare, cull, apply...)

The packaged app has no separate Python, so the app starts each step as a copy of itself with
--run (engine.config.script_command), and the steps start their worker processes the same way
(multiprocessing.freeze_support, first thing below). Run from the code it works the same:

    python app/keepsake_main.py --run check_setup
"""
from __future__ import annotations

import importlib.util
import multiprocessing
import re
import runpy
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):                      # from the code: the project folder on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_step(name: str, args: list[str]) -> int:
    """One of the scripts in scripts/, as its own program: its exit code is the process's."""
    if not re.fullmatch(r"[a-z_]+", name) or importlib.util.find_spec(f"scripts.{name}") is None:
        print(f"Unknown step: {name}")
        return 2
    sys.argv = [f"{name}.py", *args]
    runpy.run_module(f"scripts.{name}", run_name="__main__", alter_sys=True)   # sys.exit() inside ends here
    return 0


def main() -> int:
    multiprocessing.freeze_support()       # a worker process of a step: does its task and ends here
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        return run_step(sys.argv[2], sys.argv[3:])
    import keepsake_app
    return keepsake_app.main()


if __name__ == "__main__":
    sys.exit(main())
