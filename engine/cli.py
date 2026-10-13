"""Small helpers shared by the command-line scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def path_arg(default: Path) -> Path:
    """The folder or file given on the command line, or `default`.

    Cleans up a Windows quirk: `ks list "E:\\"` arrives as `E:"`.
    """
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        return default
    text = sys.argv[1].strip().strip('"')
    if len(text) == 2 and text[1] == ":":
        text += "\\"
    return Path(text)
