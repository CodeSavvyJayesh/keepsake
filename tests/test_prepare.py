"""Tests for many weddings at once: pairing photos with picks, ks match with two folders. Run: ks test"""
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_couple_names_pair_raw_and_selected_folders(tmp_path):
    from prepare import couple_key, pair_weddings
    assert couple_key("Vaibhav & Preeyam { Jaipur }") == couple_key("Vaibhav & Preeyam Jaipur") == ("vaibhav", "preeyam")
    assert couple_key("twinkle n jacky") == couple_key("Twinkle & Jacky (Dubai)")
    raw1, raw2, sel = tmp_path / "D" / "All Raw images", tmp_path / "E" / "All Raw images", tmp_path / "E" / "All Selected images"
    for parent, names in ((raw1, ["Ansh & Anjali  { Bikaner }", "twinkle n jacky", "Tej & Suriya"]),
                          (raw2, ["Shevaun & Brian Jaisalmer"]),
                          (sel, ["Ansh & Anjali ( Bikaner )", "Twinkle & Jacky (Dubai)", "Shevaun & Brian", "Soumya & Piyush (Goa)"])):
        for n in names:
            (parent / n).mkdir(parents=True)
    (sel / "._Tej & Suriya").write_bytes(b"mac")             # macOS leftovers are ignored
    pairs = {p.name: (s.name if s else None) for p, s in pair_weddings([raw1, raw2], sel)}
    assert pairs == {"Ansh & Anjali  { Bikaner }": "Ansh & Anjali ( Bikaner )",
                     "twinkle n jacky": "Twinkle & Jacky (Dubai)",
                     "Tej & Suriya": None,
                     "Shevaun & Brian Jaisalmer": "Shevaun & Brian"}


def test_a_wedding_prepared_under_another_folder_is_skipped(tmp_path):
    from prepare import prepared_elsewhere
    cache = tmp_path / "cache"
    (cache / "dinesh_ruchi_bali_raw_images").mkdir(parents=True)
    (cache / "dinesh_ruchi_bali_raw_images" / "catalog.db").write_bytes(b"")
    again = tmp_path / "E" / "All Raw images" / "Dinesh & Ruchi (Bali)"
    other = tmp_path / "E" / "All Raw images" / "Soumya & Piyush (Goa)"
    assert prepared_elsewhere(again, cache) == "dinesh_ruchi_bali_raw_images"
    assert prepared_elsewhere(other, cache) is None


def test_match_with_separate_raw_and_selected_folders(tmp_path):
    from engine.labels import load_label_csv
    raw = tmp_path / "D" / "All Raw images" / "Ansh & Anjali"
    sel = tmp_path / "E" / "All Selected images" / "Ansh & Anjali"
    for i in range(4):
        (raw / f"RBP_{i:04d}.JPG").parent.mkdir(parents=True, exist_ok=True)
        (raw / f"RBP_{i:04d}.JPG").write_bytes(os.urandom(100 + i))
        (raw / f"DSC_{i:04d}.JPG").write_bytes(os.urandom(100 + i))
    sel.mkdir(parents=True)
    (sel / "RBP_0001.jpg").write_bytes(b"edited export")
    (sel / "._RBP_0001.jpg").write_bytes(b"mac")
    env = dict(os.environ, KEEPSAKE_DATA=str(tmp_path / "data"), PYTHONUTF8="1")
    r = subprocess.run([sys.executable, str(SCRIPTS / "match_selected.py"), str(raw), str(sel)], env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Matched 1 of 1" in r.stdout and "no picks at all" not in r.stdout
    out = tmp_path / "data" / "labels" / "all_raw_images_ansh_anjali.csv"
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 8 and sum(int(x["selected"]) for x in rows) == 1
    assert load_label_csv(out, raw) == {f"{p}_{i:04d}.JPG": int(p == "RBP" and i == 1)
                                        for p in ("RBP", "DSC") for i in range(4)}
