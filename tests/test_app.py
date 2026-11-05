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
    assert app.wedding_name(Path("D:/All Raw images/Arjun & Riya")) == "Arjun & Riya"


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
    assert (r["photo"]["id"], r["photo"]["sel"], r["photo"]["moved"], r["photo"]["t"]) == (2, True, True, 1)
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


def test_swap_frame_puts_the_editors_frame_in_selected(server, tmp_path):
    """Big view: the frames of the photo's shot (and its moment) appear beside it, and one click puts
    another frame into Selected in place of KeepSake's; Undo is the same swap back."""
    app, base, token = server
    from engine import deliver
    from engine.catalog import Catalog
    from engine.ingest import cache_path
    root = tmp_path / "drive" / "Mehta Wedding"
    (root / "Haldi").mkdir(parents=True)
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    for n in names:
        (root / "Haldi" / n).write_bytes(n.encode() * 1000)
    shoot, db = "mehta", app.CACHE_DIR / "mehta" / "catalog.db"
    out = tmp_path / "out" / "Mehta Wedding"
    kept = "Same shot as a.jpg, which is kept"
    with Catalog(db) as cat:
        with cat.db:
            cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, taken_s) "
                               "VALUES (?, ?, 'Haldi', 'jpeg', 3000, 0, 'done', ?)",
                               [(i + 1, f"Haldi/{n}", i) for i, n in enumerate(names)])
            cat.db.executemany("INSERT INTO moments (id, moment, size, pos) VALUES (?, 1, 4, ?)", [(i, i - 1) for i in range(1, 5)])
        cat.save_cull([(1, "selected", 0, 90, "", None),
                       (2, "removed", 0, 80, kept, 1),
                       (3, "removed", 0, 70, kept + " - eyes closed (or lowered) on a main person in this one", 1),
                       (4, "removed", 0, 60, "Similar photo kept: a.jpg (same moment) - this one is softer", 1),
                       (5, "selected", 0, 95, "", None)],
                      {}, {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (4, 1), 5: (5, 1)})
        deliver.apply(root, out, deliver.jobs_from(cat, "list"), progress=lambda _line: None)
    for pid in range(1, 6):
        thumb = cache_path(app.CACHE_DIR / shoot, "thumbs", pid)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"\xff\xd8")
    app.JOB.status, app.JOB.shoot_name, app.JOB.folder, app.JOB.output = "done", shoot, root, out

    sel = call(base, "/api/gallery?set=selected", token)["photos"]
    assert [(p["name"], p["frames"]) for p in sel] == [("a.jpg", 2), ("e.jpg", 0)]
    old = app.CACHE_DIR / "old" / "catalog.db"           # a wedding culled by an older KeepSake (no shot columns yet)
    with Catalog(old) as cat, cat.db:
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) VALUES (1, 'x.jpg', '', 'jpeg', 1, 0, 'done')")
        cat.db.execute("INSERT INTO cull (id, decision, unsure) VALUES (1, 'selected', 0)")
        cat.db.execute("DROP INDEX cull_shot")
        cat.db.execute("ALTER TABLE cull DROP COLUMN shot")
        cat.db.execute("ALTER TABLE cull DROP COLUMN frame_rank")
    thumb = cache_path(app.CACHE_DIR / "old", "thumbs", 1)
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"\xff\xd8")
    assert [p["name"] for p in app.gallery("old", "selected")] == ["x.jpg"] and app.frames("old", 1)["shot"][0]["id"] == 1
    f = call(base, "/api/frames?id=1", token)
    assert [(p["id"], p["sel"], p["rank"]) for p in f["shot"]] == [(1, True, 1), (2, False, 2), (3, False, 3)]
    assert [p["id"] for p in f["moment"]] == [4]
    assert call(base, "/api/frames?id=3", token)["shot"] == f["shot"]           # the same strip from any frame

    # the editor prefers b.jpg: it goes into Selected (copied from the drive), a.jpg moves out
    r = call(base, "/api/swap", token, {"id": 2, "replace": 1})
    assert (r["result"]["selected"], r["result"]["not_selected"], r["result"]["moved"]) == (2, 3, 2)
    assert r["selected"]["id"] == 2 and r["selected"]["mine"] and r["selected"]["reason"] == "You chose this frame instead of a.jpg"
    assert r["not_selected"]["id"] == 1 and r["not_selected"]["reason"] == "You chose b.jpg instead"
    assert (out / "Selected" / "Haldi" / "b.jpg").read_bytes() == (root / "Haldi" / "b.jpg").read_bytes()
    assert (out / "Not selected" / "Haldi" / "a.jpg").exists() and not (out / "Selected" / "Haldi" / "a.jpg").exists()
    f = {p["id"]: p for p in call(base, "/api/frames?id=2", token)["shot"] + call(base, "/api/frames?id=2", token)["moment"]}
    assert f[3]["reason"] == "Same shot as b.jpg, which you chose - eyes closed (or lowered) on a main person in this one"
    assert f[4]["reason"] == "Similar photo kept: b.jpg (same moment)"         # 'softer' compared with a.jpg: dropped
    assert [p["name"] for p in call(base, "/api/gallery?set=selected", token)["photos"]] == ["b.jpg", "e.jpg"]

    # changes her mind again: c.jpg instead of b.jpg; b.jpg goes back to KeepSake's own verdict
    call(base, "/api/swap", token, {"id": 3, "replace": 2})
    f = {p["id"]: p for p in call(base, "/api/frames?id=1", token)["shot"]}
    assert [i for i in f if f[i]["sel"]] == [3] and not f[2]["moved"]
    assert f[2]["reason"] == "Same shot as c.jpg, which you chose" and f[1]["reason"] == "You chose c.jpg instead"
    # Undo twice: every photo is back exactly as KeepSake had it
    call(base, "/api/swap", token, {"id": 2, "replace": 3})
    r = call(base, "/api/swap", token, {"id": 1, "replace": 2})
    assert (r["result"]["selected"], r["result"]["moved"]) == (2, 0) and r["selected"]["reason"] == ""
    with Catalog(db) as cat:
        assert dict(cat.db.execute("SELECT id, reason FROM cull WHERE id IN (2, 3)"))[2] == kept
    assert (out / "Selected" / "Haldi" / "a.jpg").exists() and (out / "Not selected" / "Haldi" / "b.jpg").exists()
    rows = deliver.read_manifest(out)
    assert rows["Haldi/a.jpg"]["copy"] == "Selected/Haldi/a.jpg"

    # while the photos are still copying, the choice can be viewed but not changed yet
    app.JOB.status, app.JOB.result["ready"], app.JOB.eta = "running", True, "4 min"
    assert call(base, "/api/gallery?set=selected", token)["photos"]
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/swap", token, {"id": 2, "replace": 1})
    assert "as soon as copying finishes (about 4 min left)" in json.loads(e.value.read())["error"]
    app.JOB.status = "done"

    # refused cleanly: a frame already in Selected, a replaced photo not in Selected, the drive away
    for body, words in (({"id": 5, "replace": 1}, "already in Selected"), ({"id": 2, "replace": 3}, "not in Selected"),
                        ({"id": 1, "replace": 1}, "another frame")):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(base, "/api/swap", token, body)
        assert e.value.code == 400 and words in json.loads(e.value.read())["error"]
    root.rename(tmp_path / "drive" / "away")
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/swap", token, {"id": 4, "replace": 1})
    assert "Plug in the photo drive" in json.loads(e.value.read())["error"]
    assert call(base, "/api/state", token)["result"]["selected"] == 2
    with Catalog(db) as cat:
        assert dict(cat.db.execute("SELECT id, decision FROM cull WHERE id IN (1, 4)")) == {1: "selected", 4: "removed"}


def test_selected_maybe_and_not_selected(server, tmp_path):
    """Shortlist in three folders: the gallery shows each, a photo moves between any two (Undo puts
    KeepSake's own choice back), Swap frame keeps the folder, and the studio's picks are counted
    in Selected, with Maybe, and counting a near-identical frame."""
    app, base, token = server
    from engine import deliver
    from engine.catalog import Catalog
    from engine.ingest import cache_path
    root = tmp_path / "drive" / "Shah Wedding"
    (root / "Haldi").mkdir(parents=True)
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    for n in names:
        (root / "Haldi" / n).write_bytes(n.encode() * 1000)
    shoot, db = "shah", app.CACHE_DIR / "shah" / "catalog.db"
    out = tmp_path / "out" / "Shah Wedding"
    with Catalog(db) as cat:
        with cat.db:
            cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, taken_s) "
                               "VALUES (?, ?, 'Haldi', 'jpeg', 3000, 0, 'done', ?)",
                               [(i + 1, f"Haldi/{n}", i) for i, n in enumerate(names)])
            cat.db.executemany("INSERT INTO labels (id, selected, source) VALUES (?, ?, 'test')",
                               [(1, 1), (2, 0), (3, 1), (4, 1), (5, 0)])
        cat.save_cull([(1, "selected", 0, 90, "", None), (2, "selected", 0, 70, "", None),
                       (3, "removed", 0, 60, "Same shot as b.jpg, which is kept", 2),
                       (4, "removed", 0, 20, "Rated low", None), (5, "removed", 0, 10, "Rated low", None)],
                      {}, {1: (1, 1), 2: (2, 1), 3: (2, 2), 4: (4, 1), 5: (5, 1)}, best={1})
        deliver.apply(root, out, deliver.jobs_from(cat, "list"), progress=lambda _line: None)
    for pid in range(1, 6):
        thumb = cache_path(app.CACHE_DIR / shoot, "thumbs", pid)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"\\xff\\xd8")
    app.JOB.status, app.JOB.shoot_name, app.JOB.folder, app.JOB.output = "done", shoot, root, out
    assert (out / "Selected" / "Haldi" / "a.jpg").exists() and (out / "Maybe" / "Haldi" / "b.jpg").exists()

    g = {s: [p["name"] for p in call(base, f"/api/gallery?set={s}", token)["photos"]]
         for s in ("selected", "maybe", "not_selected")}
    assert g == {"selected": ["a.jpg"], "maybe": ["b.jpg"], "not_selected": ["c.jpg", "d.jpg", "e.jpg"]}
    c = app.counts(shoot)
    assert (c["selected"], c["maybe"], c["not_selected"]) == (1, 1, 3)
    assert (c["studio_picks"], c["studio_in_selected"], c["studio_in_kept"], c["studio_near"]) == (3, 1, 1, 2)

    # Maybe -> Selected -> Not selected, then back to Maybe: KeepSake's own choice again
    r = call(base, "/api/move", token, {"id": 2, "to": "selected"})
    assert r["photo"]["tier"] == "selected" and (r["result"]["selected"], r["result"]["maybe"]) == (2, 0)
    assert (out / "Selected" / "Haldi" / "b.jpg").exists() and not (out / "Maybe" / "Haldi" / "b.jpg").exists()
    call(base, "/api/move", token, {"id": 2, "to": "not_selected"})
    assert (out / "Not selected" / "Haldi" / "b.jpg").exists()
    r = call(base, "/api/move", token, {"id": 2, "to": "maybe"})
    assert r["photo"]["tier"] == "maybe" and not r["photo"]["moved"] and r["result"]["moved"] == 0
    assert (out / "Maybe" / "Haldi" / "b.jpg").exists()
    # a Not selected photo straight into Maybe (copied from the drive)
    r = call(base, "/api/move", token, {"id": 4, "to": "maybe"})
    assert r["photo"]["tier"] == "maybe" and (out / "Maybe" / "Haldi" / "d.jpg").read_bytes() == b"d.jpg" * 1000
    call(base, "/api/move", token, {"id": 4, "to": "not_selected"})

    # Swap frame in Maybe: the chosen frame goes to Maybe, not to Selected
    r = call(base, "/api/swap", token, {"id": 3, "replace": 2})
    assert r["selected"]["tier"] == "maybe" and r["not_selected"]["tier"] == "not_selected"
    assert (out / "Maybe" / "Haldi" / "c.jpg").exists() and (out / "Not selected" / "Haldi" / "b.jpg").exists()
    assert app.counts(shoot)["studio_in_kept"] == 2                  # c.jpg was a studio pick
    r = call(base, "/api/swap", token, {"id": 2, "replace": 3})       # Undo
    assert r["selected"]["id"] == 2 and r["selected"]["tier"] == "maybe" and r["result"]["moved"] == 0

    # a new cull keeps a person's choice of folder
    call(base, "/api/move", token, {"id": 5, "to": "maybe"})
    with Catalog(db) as cat:
        cat.save_cull([(i, "removed", 0, 1, "", None) for i in range(1, 6)], {}, best=set())
        assert cat.db.execute("SELECT decision, best FROM cull WHERE id = 5").fetchone() == ("selected", 0)


def test_the_opening_uses_a_wedding_whose_small_copies_are_here(server, tmp_path):
    """On a second computer the past weddings' catalogs come without their small copies:
    the opening must still find the one wedding that has them, every time."""
    import sqlite3
    app, base, token = server
    for k, name in enumerate(("a_copied_wedding", "b_copied_wedding", "c_read_here")):
        shoot = tmp_path / "cache" / name
        shoot.mkdir(parents=True)
        con = sqlite3.connect(shoot / "catalog.db")
        con.execute("CREATE TABLE info (key TEXT, value TEXT)")
        con.execute("CREATE TABLE photos (id INTEGER, status TEXT)")
        con.execute("INSERT INTO info VALUES ('root', ?)", (str(tmp_path / f"drive{k}" / name),))
        con.executemany("INSERT INTO photos VALUES (?, 'done')", [(i,) for i in range(1, 61)])
        con.commit()
        con.close()
    for i in range(1, 61):
        p = tmp_path / "cache" / "c_read_here" / "thumbs" / "000" / f"{i:06d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8")
    for _ in range(12):
        home = call(base, "/api/home", token)
        assert len(home["recent"]) == 3
        assert len(home["sample"]) == 48 and all(u.startswith("/photo/c_read_here/") for u in home["sample"])


def test_just_view_copies_nothing_and_a_move_only_changes_the_choice(server, tmp_path):
    """'Just view': read and choose, then stop; nothing is written anywhere. A person moves photos
    straight away (only the choice changes, even with the drive away), sees the studio's picks
    counted, and a later run with copying puts every photo where the person moved it."""
    app, base, token = server
    from engine import deliver
    from engine.catalog import Catalog
    from engine.ingest import cache_path
    folder = tmp_path / "Sharma Wedding" / "Raw"
    folder.mkdir(parents=True)
    call(base, "/api/start", token, {"folder": str(folder), "keep": "0.07", "copy": False,
                                      "output": str(tmp_path / "should not be made")})
    for _ in range(100):
        st = call(base, "/api/state", token)
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert st["progress"] == 1 and st["output"] == "" and st["result"]["copied"] is False
    assert {s["key"]: s["state"] for s in st["stages"]}["copy"] == "skipped"
    assert (st["result"]["selected"], st["result"]["not_selected"]) == (12, 88)
    assert not (tmp_path / "out").exists() and not (tmp_path / "should not be made").exists()
    assert not any("Copying" in line for line in st["log"])                         # apply.py never ran

    root = tmp_path / "drive" / "Shah Wedding"
    (root / "Haldi").mkdir(parents=True)
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    for n in names:
        (root / "Haldi" / n).write_bytes(n.encode() * 1000)
    shoot, db = "shah", app.CACHE_DIR / "shah" / "catalog.db"
    with Catalog(db) as cat:
        with cat.db:
            cat.db.executemany("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, taken_s) "
                               "VALUES (?, ?, 'Haldi', 'jpeg', 4000, 0, 'done', ?)",
                               [(i + 1, f"Haldi/{n}", i) for i, n in enumerate(names)])
            cat.db.executemany("INSERT INTO labels (id, selected, source) VALUES (?, ?, 'test')",
                               [(1, 1), (2, 0), (3, 1), (4, 0)])
        cat.save_cull([(1, "selected", 0, 90, "", None), (2, "selected", 0, 70, "", None),
                       (3, "removed", 0, 60, "Same shot as b.jpg, which is kept", 2), (4, "removed", 0, 10, "Rated low", None)],
                      {}, {1: (1, 1), 2: (2, 1), 3: (2, 2), 4: (4, 1)}, best={1})
    for pid in range(1, 5):
        thumb = cache_path(app.CACHE_DIR / shoot, "thumbs", pid)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"\xff\xd8")
    with app.JOB.lock:
        app.JOB.shoot_name, app.JOB.folder = shoot, root            # the finished 'just view' run, for this wedding
    root.rename(tmp_path / "drive" / "away")                        # works without the photo drive too

    c = app.counts(shoot)
    assert (c["selected"], c["maybe"], c["not_selected"], c["studio_in_selected"], c["studio_near"]) == (1, 1, 2, 1, 2)
    r = call(base, "/api/move", token, {"id": 3, "to": "selected"})
    assert r["photo"]["tier"] == "selected" and r["photo"]["moved"]
    assert (r["result"]["selected"], r["result"]["studio_in_selected"], r["result"]["moved"]) == (2, 2, 1)
    r = call(base, "/api/move", token, {"id": 2, "to": "not_selected"})
    assert (r["result"]["maybe"], r["result"]["not_selected"]) == (0, 2)
    r = call(base, "/api/move", token, {"id": 2, "to": "maybe"})              # Undo: KeepSake's own choice
    assert r["photo"]["tier"] == "maybe" and not r["photo"]["moved"] and r["result"]["moved"] == 1
    r = call(base, "/api/swap", token, {"id": 4, "replace": 1})               # Swap frame, also only the choice
    assert r["selected"]["tier"] == "selected" and r["not_selected"]["id"] == 1
    assert app.JOB.result["selected"] == 2
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob("**/Selected"))

    # later, a run with copying: KeepSake chooses again, the person's moves stay, the copies follow them
    (tmp_path / "drive" / "away").rename(root)
    out = tmp_path / "out" / "Shah Wedding"
    with Catalog(db) as cat:
        cat.save_cull([(1, "selected", 0, 90, "", None), (2, "selected", 0, 70, "", None),
                       (3, "removed", 0, 60, "", 2), (4, "removed", 0, 10, "", None)], {}, best={1})
        deliver.apply(root, out, deliver.jobs_from(cat, "list"), progress=lambda _line: None)
    assert sorted(p.name for p in (out / "Selected" / "Haldi").iterdir()) == ["c.jpg", "d.jpg"]
    assert [p.name for p in (out / "Maybe" / "Haldi").iterdir()] == ["b.jpg"]
