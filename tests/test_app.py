"""Tests for the desktop app's engine (app/keepsake_app.py): the local server and the job.
The real steps are replaced by small scripts that print what ks prepare / cull / apply print."""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

FAKE = {
    "prepare.py": '''import sys
for step in ("ingest", "quality", "faces", "clip", "groups"):
    print(f"  - {step}", flush=True)
    for k in (0, 50, 100):
        print(f"      Reading {k:>7,} / 100   {k:5.1f}%   10.0 photos/s   about 5 s left   ", end="\\r", flush=True)
    print()
''',
    "cull.py": '''import sys
assert "--keep" in sys.argv and sys.argv[sys.argv.index("--keep") + 1] == "7"
print("  Shots               80 (near-identical frames grouped)")
print("  Selected            12 of 100 (12%); 0 of them AI unsure")
print("  Not selected        88 (88%)")
''',
    "apply.py": '''import sys
from pathlib import Path
out = Path(sys.argv[2]); (out / "Selected").mkdir(parents=True, exist_ok=True); (out / "Not selected").mkdir(exist_ok=True)
print(f"Copying 12 Selected and 88 Not selected photos to {out}")
print("  copied 40 files (0.6 GB, 50 MB/s), 60 to go", flush=True)
''',
}


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import keepsake_app as app
    fake = tmp_path / "fake"
    fake.mkdir()
    for name, code in FAKE.items():
        (fake / name).write_text(code, encoding="utf-8")
    monkeypatch.setattr(app, "SCRIPTS", fake)
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(app, "BRAND_DIR", tmp_path / "branding")
    monkeypatch.setattr(app, "CACHE_DIR", tmp_path / "cache")
    app.JOB.reset()
    srv = app.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{srv.server_port}", app.Handler.token
    srv.shutdown()


def call(base, path, token, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"X-KeepSake": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_only_this_window_can_use_keepsake(server):
    app, base, token = server
    for path in ("/api/state", "/", "/ui/app.js"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + path, timeout=5)
        assert e.value.code == 403
    with urllib.request.urlopen(f"{base}/?t={token}", timeout=5) as r:
        assert b"KeepSake" in r.read() and "ks=" in r.headers["Set-Cookie"]
    with pytest.raises(urllib.error.HTTPError):                              # no way out of the ui folder
        call(base, "/ui/..%2F..%2Fengine%2Fconfig.py", token)


def test_a_wedding_from_folder_to_selected_and_not_selected(server, tmp_path):
    app, base, token = server
    folder = tmp_path / "Sharma Wedding" / "Raw"
    folder.mkdir(parents=True)
    check = call(base, "/api/check", token, {"folder": str(folder)})
    assert check["ok"] and check["name"] == "Sharma Wedding" and check["output"].endswith("Sharma Wedding")
    assert not check["prepared"] and check["not_selected_gb"] == 0
    assert not call(base, "/api/check", token, {"folder": str(tmp_path / "missing")})["ok"]
    call(base, "/api/start", token, {"folder": str(folder), "keep": "0.07", "copy_not_selected": True})
    for _ in range(100):
        st = call(base, "/api/state", token)
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert st["progress"] == 1 and all(s["state"] == "done" for s in st["stages"])
    r = st["result"]
    assert (r["selected"], r["not_selected"], r["total"], r["shots"]) == (12, 88, 100, 80)
    assert Path(r["selected_dir"]).is_dir() and Path(r["not_selected_dir"]).name == "Not selected"
    assert st["stages"][0]["total"] == 100 and st["stages"][-1]["done"] == 40


def test_progress_lines_of_the_real_steps_are_understood():
    import keepsake_app as app
    from engine.ingest import Progress
    bar = Progress(6284, label="Faces", out=lambda *a, **k: None)
    bar.done = 1571
    m = app.PROGRESS.search("    " + bar.line())
    assert m and m.group(1) == "1,571" and m.group(2) == "6,284" and "left" not in m.group(4)
    assert app.wedding_name(Path("E:/Sharma Wedding/Raw")) == "Sharma Wedding"
    assert app.wedding_name(Path("D:/All Raw images/Rishi & Aashi")) == "Rishi & Aashi"


def test_branding_stays_on_this_laptop(server, tmp_path):
    app, base, token = server
    assert call(base, "/api/home", token)["brand"]["logo"] == ""
    (tmp_path / "branding").mkdir()
    (tmp_path / "branding" / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / "branding" / "brand.json").write_text('{"name": "A Studio"}', encoding="utf-8")
    b = call(base, "/api/home", token)["brand"]
    assert b == {"name": "A Studio", "tagline": "", "logo": "/brand/logo.png"}
    (tmp_path / "branding" / "background.jpg").write_bytes(b"\xff\xd8")
    assert call(base, "/api/home", token)["brand"]["background"] == "/brand/background.jpg"


def test_copying_shows_the_time_left():
    import keepsake_app as app
    job = app.Job()
    job.on_apply_line("Copying 596 Selected and 5,688 Not selected photos to C:/out")
    job.on_apply_line("  copied 1,160 files (17.4 GB, 47 MB/s), 5,124 to go")
    st = job.stages["copy"]
    assert (st["done"], st["total"]) == (1160, 6284) and job.eta.startswith("27 min")


def test_a_person_can_move_a_photo_across_and_undo_it(server, tmp_path):
    app, base, token = server
    from engine import deliver
    from engine.catalog import Catalog
    from engine.ingest import cache_path
    root = tmp_path / "drive" / "Sharma Wedding"
    (root / "Haldi").mkdir(parents=True)
    names = ["a.jpg", "b.jpg", "c.jpg"]
    for n in names:
        (root / "Haldi" / n).write_bytes(n.encode() * 1000)
    shoot, db = "sharma", app.CACHE_DIR / "sharma" / "catalog.db"
    out = tmp_path / "out" / "Sharma Wedding"
    with Catalog(db) as cat:
        with cat.db:
            cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, taken_s) "
                               "VALUES (?, ?, 'Haldi', 'jpeg', 3000, 0, 'done', ?)",
                               [(i + 1, f"Haldi/{n}", i) for i, n in enumerate(names)])
            cat.db.execute("INSERT INTO labels (id, selected, source) VALUES (2, 1, 'test')")
        cat.save_cull([(1, "selected", 0, 90, "", None), (2, "removed", 0, 20, "a better frame of the same moment", 1),
                       (3, "removed", 0, 10, "eyes closed", None)], {})
        deliver.apply(root, out, deliver.jobs_from(cat, "list"), progress=lambda _line: None)   # Not selected: listed only
    for pid in (1, 2, 3):
        thumb = cache_path(app.CACHE_DIR / shoot, "thumbs", pid)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"\xff\xd8")
    app.JOB.status, app.JOB.shoot_name, app.JOB.folder, app.JOB.output = "done", shoot, root, out

    # Not selected -> Selected: it was only listed, so it is copied from the drive (read only)
    r = call(base, "/api/move", token, {"id": 2, "to": "selected"})
    assert (r["result"]["selected"], r["result"]["not_selected"], r["result"]["moved"]) == (2, 1, 1)
    assert r["result"]["studio_in_selected"] == 1
    assert (out / "Selected" / "Haldi" / "b.jpg").read_bytes() == (root / "Haldi" / "b.jpg").read_bytes()
    sel = call(base, "/api/gallery?set=selected", token)["photos"]
    assert [p["name"] for p in sel] == ["a.jpg", "b.jpg"] and sel[1]["moved"] and sel[1]["reason"] == "moved by you"
    # Selected -> Not selected: the copy moves across, nothing is deleted
    call(base, "/api/move", token, {"id": 1, "to": "not_selected"})
    assert not (out / "Selected" / "Haldi" / "a.jpg").exists() and (out / "Not selected" / "Haldi" / "a.jpg").exists()
    # Undo: back in Selected, with KeepSake's own verdict again
    r = call(base, "/api/move", token, {"id": 1, "to": "selected"})
    assert (out / "Selected" / "Haldi" / "a.jpg").exists() and r["result"]["moved"] == 1
    assert app.JOB.result["selected"] == 2
    rows = deliver.read_manifest(out)                       # ks undo still knows every copy
    assert rows["Haldi/a.jpg"]["copy"] == "Selected/Haldi/a.jpg" and rows["Haldi/b.jpg"]["copy"] == "Selected/Haldi/b.jpg"
    # a new cull of the same wedding keeps the person's choice
    with Catalog(db) as cat:
        cat.save_cull([(1, "selected", 0, 90, "", None), (2, "removed", 0, 20, "", 1), (3, "removed", 0, 10, "", None)], {})
        assert dict(cat.db.execute("SELECT id, decision FROM cull")) == {1: "selected", 2: "selected", 3: "removed"}
    # a photo that is only on the drive cannot come in while the drive is away
    root.rename(tmp_path / "drive" / "away")
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/move", token, {"id": 3, "to": "selected"})
    assert e.value.code == 400 and "Plug in the photo drive" in json.loads(e.value.read())["error"]
