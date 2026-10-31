"""Tests for Phase 4: cull, apply, undo. Run: ks test"""
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from engine.catalog import Catalog
from engine.clip import to_blob
from engine.cull import cull_shoot, labelled_shoots, share_needed, with_variety
from engine.deliver import MANIFEST, OutputError, apply, check_output, jobs_from, undo
from engine.learn import Logistic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
RNG = np.random.default_rng(3)


def fake_wedding(db: Path, n=240, labelled=True, unread=0):
    """Studio keeps the smiling photos of the couple; 3-photo bursts; two events."""
    cat = Catalog(db)
    for pid in range(1, n + 1):
        event = "Haldi" if pid <= n // 2 else "Sangeet"
        smile = float(RNG.random())
        couple = pid % 2 == 0
        keep = int(couple and smile > 0.6)
        moment = (pid - 1) // 3 * 3 + 1
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status, body, taken_s) "
                       "VALUES (?, ?, ?, 'jpeg', 1, 1, 'done', 'A', ?)",
                       (pid, f"{event}/IMG_{pid:04d}.JPG", event, pid * 1.0))
        cat.db.execute("INSERT INTO quality (id, version, ok, sharp_global, sharp_peak, sharp_rel, luma_mean, luma_rel) "
                       "VALUES (?, 1, 1, ?, ?, ?, 120, ?)",
                       (pid, 100 + RNG.random(), 200 + RNG.random(), RNG.normal(), RNG.normal()))
        if labelled:
            cat.db.execute("INSERT INTO labels (id, selected) VALUES (?, ?)", (pid, keep))
        cat.db.execute("INSERT INTO moments (id, moment, size, pos) VALUES (?, ?, 3, ?)", (pid, moment, (pid - 1) % 3))
        cat.db.execute("INSERT INTO faces (photo_id, n, x, y, w, h, score, rel_w, sharp, contrast, subject, read, "
                       "smile, yaw, eyes, person) VALUES (?, 0, 500, 300, 200, 200, 0.9, 0.12, 300, 30, 1, 1, ?, 0, "
                       "'open', ?)", (pid, smile, 1 if couple else 7))
        cat.db.execute("INSERT INTO clip (id, version, ok, emb) VALUES (?, 1, 1, ?)", (pid, to_blob(RNG.normal(size=512))))
    for k in range(unread):
        cat.db.execute("INSERT INTO photos (id, path, folder, kind, bytes, mtime_ms, status) "
                       "VALUES (?, ?, 'Haldi', 'jpeg', 1, 1, 'failed')", (n + 1 + k, f"Haldi/BROKEN_{k}.JPG"))
    cat.db.commit()
    return cat


def test_share_needed_and_variety():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    s = np.arange(10, 0, -1).astype(float)          # best first
    assert share_needed(s, y, 0.95) == pytest.approx(0.8)
    adj = with_variety(np.array([3.0, 2.9, 2.8, 1.0]), np.array([1, 1, 1, 4]), strength=2)
    assert adj[3] > adj[2]                          # a single beats the 3rd frame of a burst


def test_cull_from_its_own_events(tmp_path):
    cat = fake_wedding(tmp_path / "w.db", unread=2)
    r = cull_shoot(cat, "w", [], make_model=Logistic)
    assert r.mode == "other events" and len(r.rows) == 242
    decisions = {pid: (d, u, reason) for pid, d, u, _, reason, _ in r.rows}
    assert decisions[241][0] == "selected" and decisions[241][1] == 1      # unreadable: kept, unsure
    removed = [x for x in decisions.values() if x[0] == "removed"]
    assert removed and all(x[2] for x in removed)                       # every removal has a reason
    got, picks = r.caught
    assert got / picks >= 0.85                                          # easy taste: nearly all picks kept
    assert len(removed) >= 60                                            # and a real share removed


def test_cull_learns_from_other_weddings(tmp_path):
    for name in ("past_one", "past_two"):
        (tmp_path / name).mkdir()
        fake_wedding(tmp_path / name / "catalog.db").close()
    (tmp_path / "new").mkdir()
    new = fake_wedding(tmp_path / "new" / "catalog.db", labelled=False)
    others = labelled_shoots(tmp_path, exclude=tmp_path / "new" / "catalog.db")
    assert [p.parent.name for p in others] == ["past_one", "past_two"]
    r = cull_shoot(new, "new", others, make_model=Logistic, style="safe")
    assert r.mode == "other weddings" and r.caught is None and len(r.rows) == 240
    assert 0.2 < r.keep_share < 1.0
    similar = [row for row in r.rows if row[4] and row[4].startswith("Similar photo kept")]
    assert all(row[5] for row in similar)


def test_new_wedding_gets_one_photo_per_shot_with_eyes_open(tmp_path):
    """Default style: like the studio. Each 3-frame burst is one shot (near-identical pictures);
    the middle frame of every burst has closed eyes and must never be the one kept."""
    for name in ("past_one", "past_two"):
        (tmp_path / name).mkdir()
        fake_wedding(tmp_path / name / "catalog.db").close()
    (tmp_path / "new").mkdir()
    new = fake_wedding(tmp_path / "new" / "catalog.db", labelled=False)
    for first in range(1, 241, 3):
        base = RNG.normal(size=512)
        for pid in range(first, first + 3):
            new.db.execute("UPDATE clip SET emb = ? WHERE id = ?", (to_blob(base + RNG.normal(0, 0.05, 512)), pid))
        new.db.execute("UPDATE faces SET eyes = 'closed' WHERE photo_id = ?", (first + 1,))
    new.db.commit()
    others = labelled_shoots(tmp_path, exclude=tmp_path / "new" / "catalog.db")
    r = cull_shoot(new, "new", others, make_model=Logistic)
    assert r.style == "studio" and r.shots == 80
    chosen = [pid for pid, d, *_ in r.rows if d == "selected"]
    assert abs(len(chosen) - round(r.keep_share * 240)) <= 1 and 0.1 < r.keep_share < 0.35
    assert len({(pid - 1) // 3 for pid in chosen}) == len(chosen)          # never two frames of one shot
    assert not any(pid % 3 == 2 for pid in chosen)                         # never the closed-eyes frame
    same = [row for row in r.rows if row[4] and row[4].startswith("Same shot as")]
    assert same and all(row[5] in chosen for row in same)
    assert any("eyes closed" in row[4] for row in same)
    # Swap frame needs each photo's shot and KeepSake's order of its frames: 1 = the frame kept
    assert set(r.frames) == {row[0] for row in r.rows} and len({s for s, _ in r.frames.values()}) == 80
    assert all(r.frames[pid][1] == 1 for pid in chosen)
    assert all(r.frames[pid][0] == r.frames[row[5]][0] for row in same for pid in [row[0]])
    new.save_cull(r.rows, {}, r.frames)
    assert new.db.execute("SELECT COUNT(*) FROM cull WHERE shot IS NOT NULL AND frame_rank >= 1").fetchone()[0] == 240


def test_shortlist_in_two_tiers_goes_to_three_folders(tmp_path):
    """Shortlist: Selected holds the best (about the studio's usual share), Maybe the rest of the
    shortlist; ks apply copies them into two folders and Not selected is listed."""
    for name in ("past_one", "past_two"):
        (tmp_path / name).mkdir()
        fake_wedding(tmp_path / name / "catalog.db").close()
    (tmp_path / "new").mkdir()
    new = fake_wedding(tmp_path / "new" / "catalog.db", labelled=False)
    others = labelled_shoots(tmp_path, exclude=tmp_path / "new" / "catalog.db")
    r = cull_shoot(new, "new", others, make_model=Logistic, shortlist_target=0.94)
    kept = {pid for pid, d, *_ in r.rows if d == "selected"}
    assert r.best and r.best < kept                                    # Best is part of the shortlist, not all of it
    usual = np.mean([v for o in others for (v,) in Catalog(o).db.execute("SELECT selected FROM labels")])
    assert abs(len(r.best) - usual * 240) <= max(3, 0.25 * usual * 240)
    new.save_cull(r.rows, {}, r.frames, r.best)
    tiers = dict(new.db.execute("SELECT (decision = 'selected') + best, COUNT(*) FROM cull GROUP BY 1"))
    assert tiers[2] == len(r.best) and tiers[1] == len(kept) - len(r.best)
    root = make_drive(tmp_path, new)
    out = tmp_path / "laptop" / "Out"
    apply(root, out, jobs_from(new, "list"), progress=lambda s: None)
    assert len(list((out / "Selected").rglob("*.JPG"))) == len(r.best)
    assert len(list((out / "Maybe").rglob("*.JPG"))) == len(kept) - len(r.best)
    assert not (out / "Not selected").exists()
    undo(out)
    assert not out.exists()


def test_past_weddings_are_read_once_and_kept(tmp_path):
    """The features of a past wedding are saved next to its catalog: the next cull loads exactly
    the same numbers in a moment, and a changed catalog is read again."""
    from engine.learn import FEATURES_FILE, past_wedding, photo_features
    from engine.shots import face_details
    (tmp_path / "past").mkdir()
    db = tmp_path / "past" / "catalog.db"
    fake_wedding(db).close()
    calls = []
    details = lambda cat, ids: calls.append(1) or face_details(cat, ids)      # noqa: E731
    first = past_wedding(db, details)
    assert (tmp_path / "past" / FEATURES_FILE).exists() and len(calls) == 1
    again = past_wedding(db, details)
    assert len(calls) == 1                                        # loaded, not computed
    for a, b in ((first.X, again.X), (first.y, again.y), (first.moment, again.moment), (first.clip, again.clip),
                 (first.meta["faces"], again.meta["faces"])):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert (first.ids, first.names, first.group, again.meta["wedding"]) == (again.ids, again.names, again.group, "past")
    with Catalog(db) as cat:
        fresh = photo_features(cat, "past")
        assert np.array_equal(fresh.X, again.X)
        cat.db.execute("UPDATE labels SET selected = 1 - selected WHERE id = 1")
        cat.db.commit()
    changed = past_wedding(db, details)
    assert len(calls) == 2 and changed.y[0] != first.y[0]       # the catalog changed: read again


def make_drive(tmp_path, cat: Catalog):
    root = tmp_path / "drive" / "Raw"
    for (path,) in cat.db.execute("SELECT path FROM photos"):
        f = root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(os.urandom(2000))
    return root


def test_apply_copies_checks_and_undoes(tmp_path):
    cat = fake_wedding(tmp_path / "w.db", n=30)
    r = cull_shoot(cat, "w", [], make_model=Logistic, keep_share=0.5)
    cat.save_cull(r.rows, {})
    root = make_drive(tmp_path, cat)
    before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob("*.JPG")}
    out = tmp_path / "laptop" / "Output"
    report = apply(root, out, jobs_from(cat), progress=lambda s: None)
    assert report.copied == 30 and set(report.folders) == {"Selected", "Not selected"}
    rows = list(csv.DictReader(open(out / MANIFEST, encoding="utf-8")))
    for row in rows:
        copy = out / row["copy"]
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == row["sha256"] == before[root / row["original"]][1]
        assert copy.stat().st_mtime_ns == before[root / row["original"]][0]
    assert not list(out.rglob("*.keepsake-part"))
    after = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob("*.JPG")}
    assert after == before                                          # originals untouched

    again = apply(root, out, jobs_from(cat), progress=lambda s: None)
    assert again.copied == 0 and again.skipped == 30                # carries on, nothing twice

    first_removed = next(pid for pid, d, *_ in r.rows if d == "removed")
    cat.move([first_removed], "selected")                           # a person moves one photo
    moved = apply(root, out, jobs_from(cat), progress=lambda s: None)
    assert moved.moved == 1 and moved.copied == 0
    name = cat.db.execute("SELECT path FROM photos WHERE id = ?", (first_removed,)).fetchone()[0]
    assert (out / "Selected" / name).exists() and not (out / "Not selected" / name).exists()

    edited = next((out / "Selected").rglob("*.JPG"))
    edited.write_bytes(b"edited by the photographer")
    u = undo(out)
    assert u.removed == 29 and len(u.kept_changed) == 1             # an edited copy is never deleted
    assert edited.exists() and after == {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                                         for p in root.rglob("*.JPG")}


def test_apply_list_only_removed(tmp_path):
    cat = fake_wedding(tmp_path / "w.db", n=30)
    r = cull_shoot(cat, "w", [], make_model=Logistic, keep_share=0.5)
    cat.save_cull(r.rows, {})
    root = make_drive(tmp_path, cat)
    out = tmp_path / "Output"
    report = apply(root, out, jobs_from(cat, removed="list"), progress=lambda s: None)
    n_removed = sum(1 for row in r.rows if row[1] == "removed")
    assert report.listed == n_removed and report.copied == 30 - n_removed
    assert not (out / "Not selected").exists()


def test_output_never_on_the_photo_drive(tmp_path):
    root = tmp_path / "drive" / "Raw"
    root.mkdir(parents=True)
    with pytest.raises(OutputError):
        check_output(root, root / "Selected")
    with pytest.raises(OutputError):
        check_output(root, root.parent)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "holiday.jpg").write_bytes(b"x")
    with pytest.raises(OutputError):
        check_output(root, busy)                                    # never mixes with other files
    check_output(root, tmp_path / "laptop" / "new")


def test_cull_and_apply_commands(tmp_path, monkeypatch):
    import apply as apply_script
    import cull as cull_script
    import undo as undo_script
    from engine import ingest
    root = tmp_path / "drive" / "Sharma Wedding" / "Raw"
    shoot = ingest.shoot_for(root, cache_dir=tmp_path / "cache")
    shoot.cache.mkdir(parents=True)
    cat = fake_wedding(shoot.db, n=60)
    make_drive(tmp_path, cat)
    real_root = tmp_path / "drive" / "Raw"
    cat.close()
    for script in (cull_script, apply_script):
        monkeypatch.setattr(script, "shoot_for", lambda r: shoot)
    monkeypatch.setattr(cull_script, "attach_labels", lambda c, r: None)
    monkeypatch.setattr(cull_script, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sys, "argv", ["cull.py", str(real_root), "--target", "90"])
    assert cull_script.main() == 0
    assert (shoot.cache / "review" / "cull_review.html").exists()
    changes = tmp_path / "keepsake_changes.csv"
    with Catalog(shoot.db) as c:
        removed = c.db.execute("SELECT p.path FROM cull JOIN photos p USING (id) WHERE decision = 'removed'").fetchone()[0]
    changes.write_text(f'path,decision\n"{removed}",selected\n', encoding="utf-8")
    out = tmp_path / "Output"
    monkeypatch.setattr(sys, "argv", ["apply.py", str(real_root), str(out), "--changes", str(changes)])
    assert apply_script.main() == 0
    assert (out / "Selected" / removed).exists() and (out / "KeepSake read me.txt").exists()
    monkeypatch.setattr(sys, "argv", ["undo.py", str(out)])
    assert undo_script.main() == 0
    assert not (out / "Selected").exists()
