"""Runs `ks ingest` the way the studio will, on a small fake shoot. Run: ks test"""
import os
import subprocess
import sys
from pathlib import Path

from makephotos import camera_jpeg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest.py"


def ks_ingest(tmp_path, *args):
    env = dict(os.environ, KEEPSAKE_DATA=str(tmp_path / "data"),
               KEEPSAKE_DOCS=str(tmp_path / "docs"), PYTHONUTF8="1")
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


def test_ingest_command_runs_resumes_and_reports(tmp_path):
    root = tmp_path / "Sharma Wedding" / "Raw"
    for i in range(3):
        camera_jpeg(root / "Haldi" / f"DSC_000{i}.JPG")
    camera_jpeg(root / "Haldi" / "L1000001.JPG", preview=None, make="LEICA CAMERA AG",
                model="LEICA Q (Typ 116)", taken="2015:01:01 03:55:19", serial=None)

    first = ks_ingest(tmp_path, root, "--workers", 1)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "4 new" in first.stdout and "done 4" in first.stdout
    assert "CHECK CLOCK" in first.stdout                     # the Leica's 2015 clock
    cache = tmp_path / "data" / "cache" / "sharma_wedding_raw"
    assert (cache / "catalog.db").exists() and (cache / "report.txt").exists()

    again = ks_ingest(tmp_path, root)
    assert again.returncode == 0 and "Nothing new to read" in again.stdout

    speed = ks_ingest(tmp_path, root, "--bench", "--bench-workers", "1", "--bench-size", "2")
    assert speed.returncode == 0, speed.stdout + speed.stderr
    assert "Best: 1 worker," in speed.stdout
    assert (tmp_path / "docs" / "phase1_speed.txt").exists()
    assert '"workers": 1' in (cache / "settings.json").read_text()


def test_wedding_folder_with_selected_copies_is_refused(tmp_path):
    wedding = tmp_path / "Sharma Wedding"
    camera_jpeg(wedding / "Raw image's" / "DSC_0001.JPG")
    camera_jpeg(wedding / "Selected image's" / "DSC_0001.JPG")
    r = ks_ingest(tmp_path, wedding)
    assert r.returncode == 1 and "selected copies" in r.stdout
    assert "Raw image's" in r.stdout                          # tells her the right command
