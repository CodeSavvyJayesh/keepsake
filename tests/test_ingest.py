"""Tests for Phase 1 ingest: catalog, resume, parallel workers, damaged files. Run: ks test"""
import os
from pathlib import Path

import pytest
from PIL import Image

from engine.catalog import Catalog
from engine.ingest import (Progress, Runner, bench, ingest, init_worker, is_inside, process_one,
                           shoot_files, shoot_for)
from engine.scan import scan_photos
from makephotos import camera_jpeg

QUIET = dict(progress=False)


def make_shoot(tmp_path: Path) -> Path:
    root = tmp_path / "Sharma & Verma (Goa)" / "Raw image's"
    camera_jpeg(root / "Day 1" / "DSC_0001.JPG")
    camera_jpeg(root / "Day 1" / "DSC_0002.JPG", orientation=8)
    camera_jpeg(root / "Day 2" / "L1000001.JPG", size=(6000, 4000), preview=None,
                make="LEICA CAMERA AG", model="LEICA Q (Typ 116)",
                taken="2015:01:01 03:55:19", subsec=None, serial=None)
    camera_jpeg(root / "Day 2" / "pair" / "DSC_0100.JPG")
    (root / "Day 2" / "pair" / "DSC_0100.NEF").write_bytes(b"raw sensor data")
    (root / "Day 2" / "empty.JPG").write_bytes(b"")
    (root / "Day 2" / "broken.JPG").write_bytes(b"not a photo" * 50)
    return root


def open_shoot(tmp_path: Path, root: Path):
    shoot = shoot_for(root, cache_dir=tmp_path / "cache")
    cat = Catalog(shoot.db)
    cat.sync(shoot_files(root, scan_photos(root)))
    return shoot, cat


def snapshot(root: Path) -> dict:
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


def test_ingest_makes_catalog_copies_and_thumbnails(tmp_path):
    root = make_shoot(tmp_path)
    before = snapshot(root)
    shoot, cat = open_shoot(tmp_path, root)
    assert shoot.name == "sharma_verma_goa_raw_images"

    run = ingest(shoot, cat, workers=0, **QUIET)
    assert (run.processed, run.done, run.failed) == (6, 4, 2)
    assert cat.counts() == {"done": 4, "failed": 1, "empty": 1}
    assert dict(cat.failures()) == {
        "Day 2/broken.JPG": "not a readable photo (damaged, or another kind of file)",
        "Day 2/empty.JPG": "empty.JPG is empty (0 bytes)"}

    rows = {r[0]: r[1:] for r in cat.db.execute(
        "SELECT path, companion, status, preview, body, folder FROM photos")}
    assert rows["Day 2/pair/DSC_0100.NEF"][:2] == ("Day 2/pair/DSC_0100.JPG", "done")  # one photo
    assert rows["Day 2/L1000001.JPG"][2:4] == ("decoded", "LEICA Q (Typ 116)")
    assert rows["Day 1/DSC_0001.JPG"][2:] == ("embedded", "NIKON Z 9 #7600802", "Day 1")

    for (pid,) in cat.db.execute("SELECT id FROM photos WHERE status = 'done'"):
        assert max(Image.open(shoot.analysis_path(pid)).size) == 1620
        assert max(Image.open(shoot.thumb_path(pid)).size) == 400
    pid, w, h = cat.db.execute(
        "SELECT id, width, height FROM photos WHERE path = 'Day 1/DSC_0002.JPG'").fetchone()
    assert (w, h) == (2000, 3000)
    assert Image.open(shoot.analysis_path(pid)).size == (1080, 1620)     # upright

    assert snapshot(root) == before            # the shoot folder was only read
    assert not is_inside(shoot.cache, root)
    cat.close()


def test_second_run_only_does_what_changed(tmp_path):
    root = make_shoot(tmp_path)
    shoot, cat = open_shoot(tmp_path, root)
    ingest(shoot, cat, workers=0, **QUIET)

    sync = cat.sync(shoot_files(root, scan_photos(root)))
    assert (sync.new, sync.changed, sync.unchanged, sync.missing) == (0, 0, 6, 0)
    assert [row[1] for row in cat.pending()] == ["Day 2/broken.JPG"]   # failed: retried

    camera_jpeg(root / "Day 1" / "DSC_0001.JPG", size=(3600, 2400))    # re-exported
    (root / "Day 1" / "DSC_0002.JPG").unlink()                          # deleted
    camera_jpeg(root / "Day 3" / "DSC_0300.JPG")                        # added
    sync = cat.sync(shoot_files(root, scan_photos(root)))
    assert (sync.new, sync.changed, sync.missing) == (1, 1, 1)

    run = ingest(shoot, cat, workers=0, **QUIET)
    assert run.processed == 3                     # changed + new + the broken one again
    width = cat.db.execute("SELECT width FROM photos WHERE path = 'Day 1/DSC_0001.JPG'").fetchone()
    assert width == (3600,)
    assert cat.counts()["missing"] == 1
    cat.close()


def test_limit_and_redo(tmp_path):
    root = make_shoot(tmp_path)
    shoot, cat = open_shoot(tmp_path, root)
    assert ingest(shoot, cat, workers=0, limit=2, **QUIET).processed == 2
    assert cat.counts()["new"] == 4
    ingest(shoot, cat, workers=0, **QUIET)
    cat.redo_all()
    assert ingest(shoot, cat, workers=0, **QUIET).processed == 6
    cat.close()


def test_parallel_workers(tmp_path):
    root = make_shoot(tmp_path)
    shoot, cat = open_shoot(tmp_path, root)
    run = ingest(shoot, cat, workers=2, **QUIET)
    assert (run.done, run.failed) == (4, 2)
    cat.close()


def crash_on_marked(task):
    """Stands in for a damaged RAW file that crashes LibRaw and kills its worker process."""
    if "CRASH" in task[1]:
        os._exit(3)
    return {"id": task[0], "status": "done"}


def test_a_file_that_crashes_its_worker_is_marked_failed(tmp_path):
    tasks = [(1, "a.JPG", None), (2, "CRASH.JPG", None), (3, "b.JPG", None), (4, "c.JPG", None)]
    results = {}
    with Runner(2, tmp_path, fn=crash_on_marked) as runner:
        runner.run(tasks, lambda r: results.__setitem__(r["id"], r["status"]))
    assert results == {1: "done", 2: "failed", 3: "done", 4: "done"}


def test_exiftool_fills_in_what_pillow_cannot_find(tmp_path):
    from engine.config import exiftool_path
    try:
        exiftool_path()
    except FileNotFoundError:
        pytest.skip("ExifTool not installed")
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
           b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" '
           b'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
           b'xmp:CreateDate="2026-08-27T16:41:34" tiff:Make="Canon" tiff:Model="Canon EOS R5"/>'
           b'</rdf:RDF></x:xmpmeta>')
    p = tmp_path / "exported.jpg"
    Image.new("RGB", (1200, 800), (90, 90, 90)).save(p, "JPEG", xmp=xmp)    # no EXIF at all
    init_worker(str(tmp_path / "cache"), True, ignore_ctrl_c=False)
    r = process_one((7, str(p), None))
    assert r["status"] == "done"
    assert r["camera"] == "Canon EOS R5" and r["taken"] == "2026-08-27 16:41:34"


def test_bodies_flag_a_camera_whose_clock_was_not_set(tmp_path):
    cat = Catalog(tmp_path / "c.db")
    rows = [(f"a{i}.JPG", "NIKON Z 9 #1", 1787848894.0 + i * 60, f"2026-08-27 16:{i:02d}:00")
            for i in range(10)]
    rows += [(f"l{i}.JPG", "LEICA Q", 1420070400.0 + i, f"2015-01-01 00:00:0{i}") for i in range(3)]
    day = 86400.0   # a second shooter who only covered the last evening, 2.5 days later
    rows += [(f"t{i}.JPG", "NIKON Z 9 #2", 1787848894.0 + 2.5 * day + i, "2026-08-30") for i in range(4)]
    rows += [(f"b{i}.JPG", "NIKON Z 9 #1", 1787848894.0 + d * day, "2026-08-2x") for i, d in
             enumerate([1, 1.5, 2, 2.4])]
    cat.db.executemany("INSERT INTO photos (path, folder, kind, bytes, mtime_ms, status, body, "
                       "taken_s, taken) VALUES (?, '', 'jpeg', 1, 1, 'done', ?, ?, ?)", rows)
    bodies = {b.name: b for b in cat.bodies()}
    assert bodies["LEICA Q"].clock_off
    assert not bodies["NIKON Z 9 #1"].clock_off and not bodies["NIKON Z 9 #2"].clock_off
    assert bodies["NIKON Z 9 #1"].photos == 14
    assert bodies["NIKON Z 9 #1"].first == "2026-08-27 16:00:00"
    cat.close()


def test_speed_test_leaves_nothing_behind(tmp_path):
    root = make_shoot(tmp_path)
    shoot = shoot_for(root, cache_dir=tmp_path / "cache")
    found = [f for f in scan_photos(root) if f.size > 0 and "broken" not in f.path.name]
    report = bench(shoot, found, [0], per_run=2, seed=1, out=lambda *a, **k: None)
    assert len(report.rows) == 1 and report.rows[0].photos == 2 and report.full is not None
    assert report.rows[0].failed == 0 and report.full.failed == 0
    assert not (tmp_path / "cache" / "_speed_test").exists()


def test_shoot_name_and_safety_checks(tmp_path):
    a = shoot_for(Path("D:/Weddings/Sharma Haldi"), cache_dir=tmp_path)
    b = shoot_for(Path("E:/Weddings/Sharma Haldi"), cache_dir=tmp_path)
    assert a.name == b.name == "weddings_sharma_haldi"      # same shoot on another drive letter
    assert is_inside(tmp_path / "a" / "b", tmp_path / "a")
    assert not is_inside(tmp_path / "ab", tmp_path / "a")


def test_progress_line():
    lines = []
    bar = Progress(10, every=0, out=lambda text, **k: lines.append(text))
    for _ in range(10):
        bar.add(0.5)
    bar.finish()
    assert "10 / 10" in lines[-1] and "photos/s" in lines[-1]
    assert sum("10 / 10" in line for line in lines) == 1        # the last line shown once


def test_unplugged_drive_stops_without_marking_photos_failed(tmp_path):
    root = make_shoot(tmp_path)
    shoot, cat = open_shoot(tmp_path, root)
    root.rename(tmp_path / "unplugged")                       # the drive disappears
    run = ingest(shoot, cat, workers=0, **QUIET)
    assert run.drive_gone and run.processed == 0
    assert cat.counts() == {"new": 6}                         # all still to do next time
    cat.close()


def test_full_laptop_disk_stops_the_run(tmp_path):
    root = make_shoot(tmp_path)
    shoot, cat = open_shoot(tmp_path, root)
    (shoot.cache / "analysis").write_text("in the way")      # writing copies now fails
    run = ingest(shoot, cat, workers=0, **QUIET)
    assert run.cache_error and run.done == 0
    assert cat.counts()["new"] >= 5
    cat.close()


def test_a_catalog_brought_from_another_computer_gets_its_copies_back(tmp_path):
    """A friend's PC gets the catalog without the copies: those photos are read again (copies
    made, nothing measured changes); a second run then has nothing to do."""
    import shutil
    from engine.ingest import copies_here, reread_missing_copies
    root = make_shoot(tmp_path)
    shoot, cat = open_shoot(tmp_path, root)
    ingest(shoot, cat, workers=0, **QUIET)
    done = sorted(r[0] for r in cat.db.execute("SELECT id FROM photos WHERE status = 'done'"))
    assert copies_here(shoot.cache, "analysis") == set(done) == copies_here(shoot.cache, "thumbs")
    assert reread_missing_copies(shoot, cat) == 0 and len(cat.pending()) == 1      # only the broken one

    before = list(cat.db.execute("SELECT id, path, width, height, taken_s, body FROM photos ORDER BY id"))
    shutil.rmtree(shoot.cache / "analysis")                  # what robocopy left behind
    shutil.rmtree(shoot.cache / "thumbs")
    assert copies_here(shoot.cache, "analysis") == set()
    assert reread_missing_copies(shoot, cat) == len(done)
    run = ingest(shoot, cat, workers=0, **QUIET)
    assert run.done == len(done)
    for pid in done:
        assert shoot.analysis_path(pid).exists() and shoot.thumb_path(pid).exists()
    assert list(cat.db.execute("SELECT id, path, width, height, taken_s, body FROM photos ORDER BY id")) == before
    assert reread_missing_copies(shoot, cat) == 0

    shoot.thumb_path(done[0]).unlink()                        # one copy short: only that photo again
    assert reread_missing_copies(shoot, cat) == 1
    cat.close()
