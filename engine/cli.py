"""Small helpers shared by the command-line scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def clean_path(text: str) -> Path:
    """A folder typed on the Windows command line, cleaned up.

    Fixes a Windows quirk: `ks list "E:\\"` arrives as `E:"`, and a bare `E:` means the
    current folder on E:, not the drive.
    """
    text = text.strip().strip('"')
    if len(text) == 2 and text[1] == ":":
        text += "\\"
    return Path(text)


def path_arg(default: Path) -> Path:
    """The folder or file given as the first command-line argument, or `default`."""
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        return default
    return clean_path(sys.argv[1])
