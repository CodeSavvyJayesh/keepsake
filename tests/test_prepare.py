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
    assert couple_key("Om & Zoya { Jaipur }") == couple_key("Om & Zoya Jaipur") == ("om", "zoya")
    assert couple_key("ira n kian") == couple_key("Ira & Kian (Dubai)")
    raw1, raw2, sel = tmp_path / "D" / "All Raw images", tmp_path / "E" / "All Raw images", tmp_path / "E" / "All Selected images"
    for parent, names in ((raw1, ["Dev & Mira  { Bikaner }", "ira n kian", "Yash & Noor"]),
                          (raw2, ["Sana & Leo Jaisalmer"]),
                          (sel, ["Dev & Mira ( Bikaner )", "Ira & Kian (Dubai)", "Sana & Leo", "Isha & Kabir (Goa)"])):
        for n in names:
            (parent / n).mkdir(parents=True)
    (sel / "._Yash & Noor").write_bytes(b"mac")             # macOS leftovers are ignored
    pairs = {p.name: (s.name if s else None) for p, s in pair_weddings([raw1, raw2], sel)}
    assert pairs == {"Dev & Mira  { Bikaner }": "Dev & Mira ( Bikaner )",
                     "ira n kian": "Ira & Kian (Dubai)",
                     "Yash & Noor": None,
                     "Sana & Leo Jaisalmer": "Sana & Leo"}


def test_a_wedding_prepared_under_another_folder_is_skipped(tmp_path):
    from prepare import prepared_elsewhere
    cache = tmp_path / "cache"
    (cache / "neel_tara_bali_raw_images").mkdir(parents=True)
    (cache / "neel_tara_bali_raw_images" / "catalog.db").write_bytes(b"")
    again = tmp_path / "E" / "All Raw images" / "Neel & Tara (Bali)"
    other = tmp_path / "E" / "All Raw images" / "Isha & Kabir (Goa)"
    assert prepared_elsewhere(again, cache) == "neel_tara_bali_raw_images"
    assert prepared_elsewhere(other, cache) is None


def test_match_with_separate_raw_and_selected_folders(tmp_path):
    from engine.labels import load_label_csv
    raw = tmp_path / "D" / "All Raw images" / "Dev & Mira"
    sel = tmp_path / "E" / "All Selected images" / "Dev & Mira"
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
    out = tmp_path / "data" / "labels" / "all_raw_images_dev_mira.csv"
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 8 and sum(int(x["selected"]) for x in rows) == 1
    assert load_label_csv(out, raw) == {f"{p}_{i:04d}.JPG": int(p == "RBP" and i == 1)
                                        for p in ("RBP", "DSC") for i in range(4)}


def test_renamed_picks_matched_by_camera_and_capture_time():
    from engine.labels import match_by_capture
    originals = {f"CM2_{i}.JPG": ("068221000207", "Canon EOS R5", 1000 + i * 0.37) for i in range(50)}
    originals.update({f"TSV_{i}.JPG": ("7601113", "NIKON Z 9", 1000.01 + i * 0.05) for i in range(50)})  # 20 fps burst
    originals["DJI_1.JPG"] = (None, "DJI FC8282", 5000.0)
    selected = {"RBP_1.jpg": ("068221000207", "Canon EOS R5", 1000 + 7 * 0.37),       # renamed export
                "RBP_2.jpg": ("7601113", "NIKON Z 9", 1000.01 + 12 * 0.05),            # one frame of the burst
                "RBP_3.jpg": (None, "DJI FC8282", 5000.0),                          # no serial: camera name
                "RBP_4.jpg": ("7601113", "NIKON Z 9", 1001.0)}                      # second only: 20 frames fit
    found, how = match_by_capture(selected, originals)
    assert found == {"RBP_1.jpg": "CM2_7.JPG", "RBP_2.jpg": "TSV_12.JPG", "RBP_3.jpg": "DJI_1.JPG"}
    assert how["camera + capture time"] == 3


def test_a_camera_clock_corrected_in_lightroom_is_found():
    from engine.labels import match_by_capture
    originals = {f"DSC_{i}.JPG": ("7600802", "NIKON Z 9", 1000 + i * 7.13) for i in range(40)}
    selected = {f"RBP_{i}.jpg": ("7600802", "NIKON Z 9", 1000 + i * 7.13 + 3600) for i in range(0, 40, 4)}
    found, how = match_by_capture(selected, originals)
    assert found == {f"RBP_{i}.jpg": f"DSC_{i}.JPG" for i in range(0, 40, 4)}
    assert how["camera + capture time (clock corrected)"] == 10


def test_match_script_falls_back_to_capture_time(tmp_path, monkeypatch):
    import match_selected as script
    from engine.catalog import Catalog
    from engine.ingest import shoot_for
    raw = tmp_path / "D" / "All Raw images" / "Dev & Mira"
    sel = tmp_path / "E" / "All Selected images" / "Dev & Mira"
    raw.mkdir(parents=True)
    sel.mkdir(parents=True)
    for i in range(6):
        (raw / f"CM2_{i}.JPG").write_bytes(os.urandom(50))
    (sel / "RBP_1215.jpg").write_bytes(b"edited export")
    monkeypatch.setattr(script, "LABELS_DIR", tmp_path / "labels")
    shoot = shoot_for(raw, cache_dir=tmp_path / "cache")
    with Catalog(shoot.db) as cat:
        for i in range(6):
            cat.db.execute("INSERT INTO photos (path, folder, kind, bytes, mtime_ms, status, serial, camera, taken_s) "
                           "VALUES (?, '', 'jpeg', 50, 1, 'done', '068221000207', 'Canon EOS R5', ?)",
                           (f"CM2_{i}.JPG", 1000.0 + i * 0.4))
        cat.db.commit()
    import engine.ingest
    monkeypatch.setattr(engine.ingest, "shoot_for", lambda r, cache_dir=None: shoot)
    monkeypatch.setattr(script, "capture_of", lambda paths, label: {p: ("068221000207", "Canon EOS R5", 1001.2)
                                                                   for p in paths})
    monkeypatch.setattr(sys, "argv", ["match_selected.py", str(raw), str(sel)])
    assert script.main() == 0
    rows = list(csv.DictReader(open(tmp_path / "labels" / f"{shoot.name}.csv", encoding="utf-8")))
    assert [r["original"] for r in rows if r["selected"] == "1"] == ["Dev & Mira/CM2_3.JPG"]


def test_a_photo_twice_in_the_raw_folder_is_marked_twice():
    from engine.labels import match_by_capture
    originals = {"DSC_1735.JPG": ("7603823", "NIKON Z 8", 2000.86, 9000),
                 "untitled folder 2/DSC_1735.JPG": ("7603823", "NIKON Z 8", 2000.86, 9000),   # same file again
                 "L1000.JPG": ("5215384", "LEICA Q", 3000.0, 100), "L1001.JPG": ("5215384", "LEICA Q", 3000.0, 120)}
    found, _ = match_by_capture({"RBP_1718.jpg": ("7603823", "NIKON Z 8", 2000.86),
                                 "RBP_9.jpg": ("5215384", "LEICA Q", 3000.0)}, originals)
    assert found == {"RBP_1718.jpg": ("DSC_1735.JPG", "untitled folder 2/DSC_1735.JPG")}   # Leica: 2 frames, unsure


def test_repick_folder_is_one_stretch_with_a_hidden_key(tmp_path, monkeypatch):
    import testset as script
    from engine.catalog import Catalog
    from engine.ingest import shoot_for
    root = tmp_path / "D" / "All Raw images" / "Arjun & Riya"
    root.mkdir(parents=True)
    shoot = shoot_for(root, cache_dir=tmp_path / "cache")
    with Catalog(shoot.db) as cat:
        for i in range(1, 121):
            body = ["A", "B", "C"][i % 3]
            (root / f"IMG_{i:04d}.JPG").write_bytes(os.urandom(40))
            cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, body, taken_s) "
                           "VALUES (?, ?, '', 'jpeg', 40, 1, 'done', ?, ?)", (i, f"IMG_{i:04d}.JPG", body, i * 10.0))
            cat.db.execute("INSERT INTO labels (id, selected) VALUES (?, ?)", (i, int(i % 10 == 0)))
        cat.db.executemany("INSERT INTO clocks (body, offset, support) VALUES (?, 0, 9)", [("A",), ("B",), ("C",)])
        cat.db.commit()
    monkeypatch.setattr(script, "shoot_for", lambda r: shoot)
    monkeypatch.setattr(script, "attach_labels", lambda c, r: None)
    monkeypatch.setattr(script, "REPICK_DIR", tmp_path / "repick")
    out = tmp_path / "laptop" / "Re-pick test"
    monkeypatch.setattr(sys, "argv", ["testset.py", str(root), "50", str(out)])
    assert script.main() == 0
    copies = sorted(p.name for p in out.glob("*.JPG"))
    assert len(copies) == 50 and copies[0].startswith("0001_")
    key = list(csv.DictReader(open(tmp_path / "repick" / f"{shoot.name}.csv", encoding="utf-8")))
    assert len(key) == 50 and 3 <= sum(int(r["studio_selected"]) for r in key) <= 7
    assert not any("select" in p.name.lower() for p in out.iterdir())      # nothing hints at the old picks


def test_small_copies_for_whatsapp(tmp_path):
    from PIL import Image
    from testset import save_small
    src = tmp_path / "big.jpg"
    Image.new("RGB", (6000, 4000), (200, 100, 50)).save(src, quality=95)
    save_small(src, tmp_path / "small.jpg")
    assert Image.open(tmp_path / "small.jpg").size == (2048, 1365)


def test_blind_choice_on_the_test_photos(tmp_path, monkeypatch):
    import json
    import testcull as script
    from engine.catalog import Catalog
    from engine.ingest import shoot_for
    root = tmp_path / "D" / "Arjun & Riya"
    shoot = shoot_for(root, cache_dir=tmp_path / "cache")
    folder = tmp_path / "Re-pick test"
    folder.mkdir()
    repick = tmp_path / "repick"
    repick.mkdir()
    with Catalog(shoot.db) as cat, open(repick / f"{shoot.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["copy", "original", "photo_id", "studio_selected"])
        for i in range(1, 101):
            name = f"{i:04d}_IMG_{i}.JPG"
            (folder / name).write_bytes(b"x")
            w.writerow([name, f"IMG_{i}.JPG", i, int(i <= 10)])
            cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) VALUES (?, ?, '', 'jpeg', 1, 1, 'done')",
                           (i, f"IMG_{i}.JPG"))
        cat.save_cull([(i, "selected", 0, 100.0 - i, None, None) for i in range(1, 101)],
                      {"mode": "other weddings", "trained_on": ["another_wedding"]})
    (repick / f"{shoot.name}.json").write_text(json.dumps({"folder": str(folder), "photos": 100}))
    monkeypatch.setattr(script, "shoot_for", lambda r: shoot)
    monkeypatch.setattr(script, "REPICK_DIR", repick)
    monkeypatch.setattr(script, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(script, "labelled_shoots", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", ["testcull.py", str(root)])
    assert script.main() == 0
    out = tmp_path / "out" / "KeepSake choice - Arjun & Riya"
    assert len(list((out / "Selected").iterdir())) == 10 and len(list((out / "Removed").iterdir())) == 90
    assert "Chosen by both              10" in (repick / f"{shoot.name}_keepsake.txt").read_text()
