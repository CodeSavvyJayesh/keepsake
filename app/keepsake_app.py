r"""KeepSake desktop app: choose a wedding folder, press Start, get Selected and Not selected.

    ks app

Opens KeepSake in its own window: on Windows a Microsoft Edge or Google Chrome app window (the
default browser if neither is found); in the Mac app its own window (WebKit, via pywebview). Everything stays on this laptop: a small web server, reachable only
from this laptop (127.0.0.1) and only with a secret key made at every start, shows the interface
(app/ui) and runs the same steps as the commands:

    ks prepare  ->  ks cull  ->  ks apply           (read, choose, copy)

so a stopped run carries on where it stopped. The photo drive is only ever read.
In the gallery a person can move any photo between Selected and Not selected; the file moves
with it, and a later cull of the same wedding keeps that person's choice.

The studio's own branding is kept on this laptop, never in Git:
    data\branding\logo.png     its logo (transparent background, for the dark window)
    data\branding\brand.json   {"name": "...", "tagline": "..."}
"""
from __future__ import annotations

import collections
import json
import mimetypes
import os
import platform
import random
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import deliver  # noqa: E402
from engine.catalog import Catalog  # noqa: E402
from engine.config import CACHE_DIR, DATA_DIR, FROZEN, script_command  # noqa: E402
from engine.config import ROOT as BUNDLE  # noqa: E402  (the packaged app's own files when FROZEN)
from engine.ingest import cache_path, duration, shoot_for  # noqa: E402

UI_DIR = (BUNDLE / "app" / "ui") if FROZEN else Path(__file__).resolve().parent / "ui"
SCRIPTS = ROOT / "scripts"
BRAND_DIR = DATA_DIR / "branding"
OUTPUT_ROOT = Path("C:/KeepSake Output") if os.name == "nt" else Path.home() / "KeepSake Output"
PYTHON = sys.executable
WINDOW = None             # the Mac app's own window (pywebview), once it is open

# The steps the window shows, with their share of the whole run (for the overall progress).
STAGES = [("read", "Reading the photos", 0.30), ("quality", "Sharpness and light", 0.08),
          ("faces", "Faces, eyes and smiles", 0.24), ("clip", "Understanding every photo", 0.20),
          ("groups", "Moments and bursts", 0.06), ("choose", "Choosing like your studio", 0.06),
          ("copy", "Copying into folders", 0.06)]
PREPARE_STEP = {"ingest": "read", "match": "read", "quality": "quality", "faces": "faces",
                "clip": "clip", "groups": "groups"}
PROGRESS = re.compile(r"([\d,]+)\s*/\s*([\d,]+)\s+([\d.]+)%.*?about (.+?) left")
COPIED = re.compile(r"copied ([\d,]+) files \(([\d,.]+) GB, ([\d,.]+) MB/s\), ([\d,]+) to go")
WIDE = 1.5                # "Wider": 1.5x the studio's usual share, so Selected holds more of the studio's own picks
COPY_MB_S = 45            # typical speed from a USB photo drive, for the estimate shown before starting
COPYING = re.compile(r"Copying ([\d,]+) Selected(?:, ([\d,]+) Maybe)?(?: and ([\d,]+) Not selected)?")
SELECTED_LINE = re.compile(r"^\s*Selected\s+([\d,]+) of ([\d,]+)")
NOT_SELECTED_LINE = re.compile(r"^\s*Not selected\s+([\d,]+)")
SHOTS_LINE = re.compile(r"^\s*Shots\s+([\d,]+)")
STUDIO_LINE = re.compile(r"studio picks in Selected\s+([\d,]+) of ([\d,]+)")


def number(text: str) -> int:
    return int(text.replace(",", ""))


def rough(seconds: float) -> str:
    """An estimate, rounded: '30 min', '1 h 20 min'."""
    m = max(1, round(seconds / 60))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def wedding_name(folder: Path) -> str:
    """'E:\\Sharma Wedding\\Raw' -> 'Sharma Wedding'."""
    return folder.parent.name if "raw" in folder.name.lower() and folder.parent.name else folder.name


# --- The job: prepare -> cull -> apply, one run at a time --------------------------------

class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.reset()

    def reset(self) -> None:
        self.status = "idle"            # idle / running / done / error / stopped
        self.folder = self.output = None
        self.name = ""
        self.stage = None
        self.stages = {k: {"key": k, "label": label, "state": "waiting", "done": 0, "total": 0}
                       for k, label, _ in STAGES}
        self.eta = ""
        self.log = collections.deque(maxlen=300)
        self.started = self.finished = None
        self.result: dict = {}
        self.error = ""
        self.stop_asked = False
        self.shoot_name = None
        self.copy = True                # False: just view (nothing is copied; moves only change the catalog)

    # state for the window
    def state(self) -> dict:
        with self.lock:
            progress = 0.0
            for k, _, w in STAGES:
                st = self.stages[k]
                if st["state"] == "done":
                    progress += w
                elif st["state"] == "active" and st["total"]:
                    progress += w * min(1.0, st["done"] / st["total"])
            if self.status == "done":
                progress = 1.0
            elapsed = (self.finished or time.time()) - self.started if self.started else 0
            return {"status": self.status, "folder": str(self.folder or ""), "output": str(self.output or ""),
                    "name": self.name, "stage": self.stage, "stages": list(self.stages.values()), "copy": self.copy,
                    "progress": round(progress, 4), "eta": self.eta, "elapsed": round(elapsed),
                    "elapsed_text": duration(elapsed) if self.started else "",
                    "log": list(self.log)[-12:], "result": self.result, "error": self.error}

    def start(self, folder: Path, output: Path | None, keep: float | str | None, copy_not_selected: bool,
              copy: bool = True) -> None:
        with self.lock:
            if self.status == "running":
                raise RuntimeError("KeepSake is already working on a wedding.")
        self.reset()
        self.copy = copy
        self.folder, self.output, self.name = folder, (output if copy else None), wedding_name(folder)
        self.shoot_name = shoot_for(folder).name
        self.status, self.started = "running", time.time()
        self.thread = threading.Thread(target=self.run, args=(keep, copy_not_selected), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_asked = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def enter(self, key: str) -> None:
        with self.lock:
            order = [k for k, _, _ in STAGES]
            for k in order[:order.index(key)]:
                self.stages[k]["state"] = "done"
            if self.stages[key]["state"] != "done":
                self.stages[key]["state"] = "active"
            self.stage, self.eta = key, ""

    def run(self, keep: float | str | None, copy_not_selected: bool) -> None:
        try:
            self.enter("read")
            if self.command(["prepare.py", str(self.folder)], self.on_prepare_line) != 0:
                raise StepFailed("Reading the photos stopped")
            self.enter("choose")
            args = ["cull.py", str(self.folder)] + (["--shortlist"] if keep == "shortlist" else
                                                    ["--wide", f"{WIDE:g}"] if keep == "wide" else
                                                    ["--keep", f"{keep * 100:g}"] if keep else [])
            if self.command(args, self.on_cull_line) != 0:
                raise StepFailed("Choosing stopped")
            with self.lock:                     # the choice is made: its photos can be viewed while they copy
                self.result.update(counts(self.shoot_name))
                self.result["ready"] = True
            if not self.copy:                   # just view: nothing is copied (run again with copying, later)
                with self.lock:
                    for k, _, _ in STAGES:
                        self.stages[k]["state"] = "done" if k != "copy" else "skipped"
                    self.result.update({"copied": False, "output": ""})
                    self.status, self.finished = "done", time.time()
                    self.result["time"] = duration(self.finished - self.started)
                return
            self.enter("copy")
            args = ["apply.py", str(self.folder), str(self.output)]
            if not copy_not_selected:
                args += ["--removed", "list"]
            if self.command(args, self.on_apply_line) != 0:
                raise StepFailed("Copying stopped")
            with self.lock:
                for st in self.stages.values():
                    st["state"] = "done"
                self.result.update({"output": str(self.output),
                                    "selected_dir": str(self.output / "Selected"),
                                    "maybe_dir": str(self.output / "Maybe"),
                                    "not_selected_dir": str(self.output / "Not selected"),
                                    "copied": True, "copied_not_selected": copy_not_selected})
                self.result.update(counts(self.shoot_name))      # includes photos moved by hand earlier
                self.status, self.finished = "done", time.time()
                self.result["time"] = duration(self.finished - self.started)
        except StepFailed as e:
            with self.lock:
                self.finished = time.time()
                if self.stop_asked:
                    self.status, self.error = "stopped", "Stopped. Start again any time: KeepSake carries on where it stopped."
                else:
                    last = next((line for line in reversed(self.log) if line.strip()), "")
                    self.status, self.error = "error", f"{e}: {last}"
        except Exception as e:                      # never leave the window waiting
            with self.lock:
                self.status, self.error, self.finished = "error", f"Something went wrong: {e}", time.time()

    def command(self, args: list[str], on_line) -> int:
        if self.stop_asked:
            return 1
        cmd = script_command(args[0], *args[1:]) if FROZEN else [PYTHON, str(SCRIPTS / args[0])] + args[1:]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
        for line in self.proc.stdout:            # text mode: '\r' progress updates arrive as lines
            line = line.rstrip()
            if not line:
                continue
            with self.lock:
                self.log.append(line.strip())
            on_line(line)
        return self.proc.wait()

    def progress(self, line: str) -> None:
        m = PROGRESS.search(line)
        if m and self.stage:
            with self.lock:
                st = self.stages[self.stage]
                st["done"], st["total"] = number(m.group(1)), number(m.group(2))
                self.eta = m.group(4).strip()

    def on_prepare_line(self, line: str) -> None:
        m = re.match(r"^\s{2}- (\w+)\s*$", line)          # prepare.py: "  - quality"
        if m and m.group(1) in PREPARE_STEP:
            self.enter(PREPARE_STEP[m.group(1)])
            return
        self.progress(line)

    def on_cull_line(self, line: str) -> None:
        with self.lock:
            for pattern, keys in ((SELECTED_LINE, ("selected", "total")), (NOT_SELECTED_LINE, ("not_selected",)),
                                  (SHOTS_LINE, ("shots",)), (STUDIO_LINE, ("studio_in_selected", "studio_picks"))):
                m = pattern.search(line)
                if m:
                    for k, g in zip(keys, m.groups()):
                        self.result[k] = number(g)

    def on_apply_line(self, line: str) -> None:
        m = COPYING.search(line)
        with self.lock:
            st = self.stages["copy"]
            if m:
                st["total"] = sum(number(g) for g in m.groups() if g)
            m = COPIED.search(line)
            if m:
                done, gb, rate, left = number(m.group(1)), float(m.group(2).replace(",", "")), \
                    float(m.group(3).replace(",", "")), number(m.group(4))
                st["done"], st["total"] = done, max(st["total"], done + left)
                if done and rate:
                    self.eta = duration(left * gb * 1000 / done / rate)


class StepFailed(Exception):
    pass


JOB = Job()


# --- Photos for the window -----------------------------------------------------------------

def catalogs() -> list[Path]:
    return sorted(CACHE_DIR.glob("*/catalog.db"))


def read_db(db: Path, sql: str, args=()) -> list:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def recent_weddings() -> list[dict]:
    out = []
    for db in catalogs():
        try:
            root = (read_db(db, "SELECT value FROM info WHERE key = 'root'") or [[None]])[0][0]
            n = read_db(db, "SELECT COUNT(*) FROM photos WHERE status = 'done'")[0][0]
        except sqlite3.Error:
            continue
        if root and n:
            out.append({"name": wedding_name(Path(root)), "folder": root, "photos": n,
                        "ready": Path(root).is_dir()})
    return sorted(out, key=lambda r: r["name"].lower())


def has_thumbs(shoot_dir: Path) -> bool:
    """True when at least one small copy of this wedding is on this computer (a catalog copied
    from another laptop comes without them)."""
    return next((shoot_dir / "thumbs").glob("*/*.jpg"), None) is not None


def thumbs(shoot: str, which: str, n: int) -> list[str]:
    """URLs of up to n thumbnails of one prepared wedding: any, selected or not selected."""
    db = CACHE_DIR / shoot / "catalog.db"
    if not db.exists():
        return []
    try:
        if which in ("selected", "not_selected"):
            want = "selected" if which == "selected" else "removed"
            ids = [r[0] for r in read_db(db, "SELECT id FROM cull WHERE decision = ?", (want,))]
        else:
            ids = [r[0] for r in read_db(db, "SELECT id FROM photos WHERE status = 'done'")]
    except sqlite3.Error:
        return []
    random.Random(len(ids)).shuffle(ids)
    out = []
    for pid in ids:
        if cache_path(CACHE_DIR / shoot, "thumbs", pid).exists():
            out.append(f"/photo/{shoot}/{pid}")
            if len(out) >= n:
                break
    return out


def gallery(shoot: str, which: str) -> list[dict]:
    """Every photo of one folder of the current wedding (selected, maybe or not_selected), in time
    order, with its file name and KeepSake's reason, for the gallery and the big view."""
    db = CACHE_DIR / shoot / "catalog.db"
    where = {"selected": "c.decision = 'selected' AND c.best = 1", "maybe": "c.decision = 'selected' AND c.best = 0"
             }.get(which, "c.decision = 'removed'")
    try:
        return entries(db, shoot, f"WHERE {where} ORDER BY p.taken_s, p.path", frames_of=True) if db.exists() else []
    except sqlite3.Error:
        return []


def tier_of(decision: str, best) -> str:
    """'selected' (the Selected folder), 'maybe' (the rest of a Shortlist) or 'not_selected'."""
    return "not_selected" if decision != "selected" else "selected" if best else "maybe"


def _in(ids) -> str:
    return "(" + ",".join("?" * len(ids)) + ")"


def _chunks(ids: list, n: int = 500):
    for k in range(0, len(ids), n):
        yield ids[k:k + n]


UPDATED: set[Path] = set()


def up_to_date(db: Path) -> None:
    """A wedding culled by an older KeepSake gets the catalog columns added since (once per run)."""
    if db not in UPDATED:
        with Catalog(db):
            pass
        UPDATED.add(db)


def entries(db: Path, shoot: str, where: str, args=(), frames_of: bool = False) -> list[dict]:
    """Photos of the wedding for the gallery and the big view: {id, u (thumbnail), name, reason,
    sel (kept: in Selected or Maybe), tier (selected / maybe / not_selected), moved (by a person), mine (a frame the person chose), rank (KeepSake's order
    of the frames of its shot, 1 = its best), t (time taken), frames (other frames of the shot,
    with frames_of)}."""
    up_to_date(db)
    rows = read_db(db, "SELECT c.id, p.path, c.decision, COALESCE(c.reason, ''), c.changed, c.similar, c.shot, "
                       "c.frame_rank, COALESCE(p.taken_s, 0), c.best FROM cull c JOIN photos p ON p.id = c.id " + where, args)
    rows = [r for r in rows if cache_path(CACHE_DIR / shoot, "thumbs", r[0]).exists()]
    reasons = live_reasons(db, [(pid, reason, changed, similar) for pid, _, _, reason, changed, similar, *_ in rows])
    size = dict(read_db(db, "SELECT shot, COUNT(*) FROM cull WHERE shot IS NOT NULL GROUP BY shot")) if frames_of else {}
    return [{"id": pid, "u": f"/photo/{shoot}/{pid}", "name": Path(path).name, "reason": reasons[pid],
             "sel": decision == "selected", "moved": bool(changed), "mine": bool(changed) and reasons[pid].startswith(MINE),
             "tier": tier_of(decision, best), "rank": rank, "t": taken,
             **({"frames": size.get(shot, 1) - 1} if frames_of else {})}
            for pid, path, decision, _, changed, _, shot, rank, taken, best in rows]


# --- Swap frame: the other frames of a photo's shot, and choosing one of them instead -------------

SHOT_MAX = 240          # frames of one shot shown in the big view (a long scene: the nearest ones)
MOMENT_MAX = 24         # then up to this many other photos of the same moment, nearest in time
MINE = "You chose"      # reasons of the frames a person swapped in and out
RELATIVE = ("softer", "less smiling")      # reason endings that compare with the frame KeepSake kept


def _names_kept(reason: str, changed: int) -> bool:
    if changed:
        return reason.startswith(MINE + " ") and not reason.startswith(MINE + " this frame")
    return reason.startswith(("Same shot as ", "Similar photo kept: "))


def live_reasons(db: Path, rows: list[tuple]) -> dict[int, str]:
    """{id: reason} for rows (id, reason, changed, similar). Some reasons name another photo: KeepSake's
    'Same shot as X, which is kept' and 'Similar photo kept: X (same moment)', and 'You chose X
    instead' on a frame a person swapped out. When X has since been swapped for another frame,
    the reason names the frame now in Selected instead."""
    out = {pid: reason for pid, reason, _, _ in rows}
    targets = sorted({sim for _, reason, changed, sim in rows if sim is not None and _names_kept(reason, changed)})
    info = {}
    for part in _chunks(targets):
        for pid, path, decision, shot, taken in read_db(
                db, "SELECT c.id, p.path, c.decision, c.shot, p.taken_s FROM cull c JOIN photos p ON p.id = c.id "
                    "WHERE c.id IN " + _in(part), part):
            if decision != "selected":
                info[pid] = (Path(path).name, shot, taken or 0)
    if not info:
        return out
    shots = sorted({shot for _, shot, _ in info.values() if shot is not None})
    now_in = collections.defaultdict(list)
    for part in _chunks(shots):
        for path, shot, taken in read_db(db, "SELECT p.path, c.shot, p.taken_s FROM cull c JOIN photos p ON p.id = c.id "
                                             "WHERE c.decision = 'selected' AND c.shot IN " + _in(part), part):
            now_in[shot].append((Path(path).name, taken or 0))
    for pid, reason, changed, sim in rows:
        if sim not in info or not _names_kept(reason, changed):
            continue
        name, shot, taken = info[sim]
        for head, kept_now, gone in (
                (f"Same shot as {name}, which is kept", "Same shot as {}, which you chose",
                 f"Same shot as {name}, which you moved to Not selected"),
                (f"Similar photo kept: {name} (same moment)", "Similar photo kept: {} (same moment)",
                 f"Similar photo {name} (same moment) - you moved it to Not selected"),
                (f"{MINE} {name} instead", MINE + " {} instead", f"{MINE} {name} instead, then moved it to Not selected")):
            if reason.startswith(head):
                tail = reason[len(head):]
                tail = "" if any(w in tail for w in RELATIVE) else tail
                kept = now_in.get(shot)
                out[pid] = (kept_now.format(min(kept, key=lambda k: abs(k[1] - taken))[0]) if kept else gone) + tail
                break
    return out


def frames(shoot: str, pid: int) -> dict:
    """The frames of this photo's shot (near-identical frames, in time order) and then other photos
    of its moment, for the big view's strip: the person can choose another frame instead."""
    db = CACHE_DIR / shoot / "catalog.db"
    out = {"id": pid, "shot": [], "moment": []}
    if not db.exists():
        return out
    try:
        up_to_date(db)
        me = read_db(db, "SELECT c.shot, m.moment FROM cull c LEFT JOIN moments m ON m.id = c.id WHERE c.id = ?", (pid,))
        if not me:
            return out
        shot, moment = me[0]
        order = "ORDER BY p.taken_s, p.path"
        in_moment = [r[0] for r in read_db(db, "SELECT m.id FROM moments m JOIN photos p ON p.id = m.id "
                                               "WHERE m.moment = ? " + order, (moment,))] if moment is not None else []
        in_shot = [r[0] for r in read_db(db, "SELECT c.id FROM cull c JOIN photos p ON p.id = c.id WHERE c.shot = ? "
                                             + order, (shot,))] if shot is not None else in_moment or [pid]
        if len(in_shot) > SHOT_MAX:                       # a long scene: the frames around this one
            at = in_shot.index(pid) if pid in in_shot else 0
            start = min(max(0, at - SHOT_MAX // 2), len(in_shot) - SHOT_MAX)
            in_shot = in_shot[start:start + SHOT_MAX]
        taken = dict(read_db(db, "SELECT id, COALESCE(taken_s, 0) FROM photos WHERE id = ?", (pid,)))
        rest = [i for i in in_moment if i not in set(in_shot)]
        if len(rest) > MOMENT_MAX:
            t = {i: s or 0 for part in _chunks(rest) for i, s in read_db(
                db, "SELECT id, taken_s FROM photos WHERE id IN " + _in(part), part)}
            near = set(sorted(rest, key=lambda i: abs(t[i] - taken.get(pid, 0)))[:MOMENT_MAX])
            rest = [i for i in rest if i in near]
        found = {}
        for part in _chunks(in_shot + rest):
            found.update({e["id"]: e for e in entries(db, shoot, "WHERE c.id IN " + _in(part), part)})
    except sqlite3.Error:
        return out
    out["shot"] = [found[i] for i in in_shot if i in found]
    out["moment"] = [found[i] for i in rest if i in found]
    return out


def swap_frame(new: int, old: int) -> dict:
    """Swap frame: `new` (Not selected) goes into Selected (or Maybe) in place of `old`, as one
    change: both copies move (as with Move) and Undo is the same swap the other way round."""
    with JOB.lock:
        shoot = JOB.shoot_name or ""
    db = CACHE_DIR / shoot / "catalog.db"
    if new == old or not db.exists():
        raise MoveError("Choose another frame.")
    state = {i: tier_of(d, b) for i, d, b in read_db(db, "SELECT id, decision, best FROM cull WHERE id IN (?, ?)", (new, old))}
    if state.get(new) != "not_selected":
        raise MoveError(f"That frame is already in {TIER_NAME[state[new]]}." if new in state
                        else "That photo is not part of this wedding.")
    if state.get(old) not in ("selected", "maybe"):
        raise MoveError("The photo it replaces is not in Selected or Maybe any more.")
    move_photo(new, state[old])                           # into the same folder as the frame it replaces
    try:
        move_photo(old, "not_selected")
    except MoveError:
        move_photo(new, "not_selected")                   # as it was
        raise
    names = {i: Path(path).name for i, path in read_db(db, "SELECT id, path FROM photos WHERE id IN (?, ?)", (new, old))}
    with MOVE_LOCK, Catalog(db) as cat, cat.db:           # the person's choice (an undo puts KeepSake's own back)
        cat.db.execute("UPDATE cull SET reason = ? WHERE id = ? AND changed = 1",
                       (f"{MINE} this frame instead of {names[old]}", new))
        cat.db.execute("UPDATE cull SET reason = ?, similar = ? WHERE id = ? AND changed = 1",
                       (f"{MINE} {names[new]} instead", new, old))
    got = {e["id"]: e for e in entries(db, shoot, "WHERE c.id IN (?, ?)", (new, old), frames_of=True)}
    return {"ok": True, "result": counts(shoot), "selected": got.get(new), "not_selected": got.get(old)}


def counts(shoot: str | None) -> dict:
    """Selected / Maybe / Not selected / moved by hand, and the studio's picks found when known:
    in Selected, in Selected or Maybe, and counting a near-identical frame of the same shot."""
    db = CACHE_DIR / (shoot or "") / "catalog.db"
    if not shoot or not db.exists():
        return {}
    try:
        up_to_date(db)
        sel, maybe, total, moved = read_db(
            db, "SELECT COALESCE(SUM(decision = 'selected' AND best = 1), 0), COALESCE(SUM(decision = 'selected' "
                "AND best = 0), 0), COUNT(*), COALESCE(SUM(changed), 0) FROM cull")[0]
        picks, found, kept = read_db(
            db, "SELECT COUNT(*), COALESCE(SUM(c.decision = 'selected' AND c.best = 1), 0), "
                "COALESCE(SUM(c.decision = 'selected'), 0) FROM labels l JOIN cull c ON c.id = l.id WHERE l.selected = 1")[0]
        near = read_db(db, "SELECT COUNT(*) FROM labels l JOIN cull c ON c.id = l.id WHERE l.selected = 1 AND "
                           "(c.decision = 'selected' OR c.shot IN (SELECT shot FROM cull WHERE decision = 'selected' "
                           "AND shot IS NOT NULL))")[0][0] if picks else 0
    except sqlite3.Error:
        return {}
    out = {"selected": sel, "maybe": maybe, "not_selected": total - sel - maybe, "total": total, "moved": moved}
    if picks:
        out.update(studio_picks=picks, studio_in_selected=found, studio_in_kept=kept, studio_near=near)
    return out


# --- A person's change: move one photo between Selected, Maybe and Not selected -------------

class MoveError(Exception):
    """Why a photo could not be moved, in plain words."""


MOVE_LOCK = threading.Lock()
BEFORE_MOVE: dict[tuple[str, int], tuple] = {}      # KeepSake's own verdict, so Undo puts it back exactly


TIER_NAME = {"selected": "Selected", "maybe": "Maybe", "not_selected": "Not selected"}


def move_photo(pid: int, to: str) -> dict:
    """Move one photo (and its RAW/JPEG companion) of the finished wedding to Selected, Maybe or Not
    selected. Its copy moves between the folders; a photo that was only listed is copied from the
    drive (read only, checksum checked, as in ks apply). Nothing is ever deleted."""
    if to not in TIER_NAME:
        raise MoveError("Choose Selected, Maybe or Not selected.")
    with JOB.lock:
        if JOB.status == "running" and JOB.result.get("ready"):
            raise MoveError("You can move photos as soon as copying finishes"
                            + (f" (about {JOB.eta} left)." if JOB.eta else "."))
        if JOB.status != "done" or not JOB.shoot_name:
            raise MoveError("Finish culling this wedding first.")
        shoot, root = JOB.shoot_name, Path(JOB.folder)
        out = Path(JOB.output) if JOB.copy and JOB.output else None     # None: just viewing, no folders yet
    decision = "removed" if to == "not_selected" else "selected"
    best = int(to == "selected")
    folder = deliver.folder_for(decision, best)
    with MOVE_LOCK, Catalog(CACHE_DIR / shoot / "catalog.db") as cat:
        row = cat.db.execute("SELECT p.path, p.companion, c.decision, c.best, c.unsure, c.reason, c.changed "
                             "FROM cull c JOIN photos p ON p.id = c.id WHERE c.id = ?", (pid,)).fetchone()
        if row is None:
            raise MoveError("That photo is not part of this wedding.")
        path, companion, now, now_best, unsure, reason, changed = row
        if tier_of(now, now_best) != to:
            before = BEFORE_MOVE.setdefault((shoot, pid), (now, now_best, unsure, reason, changed))
            back = tier_of(before[0], before[1]) == to and not before[4]   # Undo: KeepSake's own verdict again
            if out is not None:                 # the folders exist: the copy moves with the photo
                earlier = deliver.read_manifest(out).get(path, {})
                has_copy = bool(earlier.get("copy")) and (out / earlier["copy"]).exists()
                if folder != deliver.NOT_SELECTED and not has_copy and not root.is_dir():
                    raise MoveError(f"Plug in the photo drive first: this photo is copied from it into {folder}.")
                job = deliver.Job(pid, folder, before[2] if back else 0, (before[3] or "") if back else "moved by you",
                                  path, folder != deliver.NOT_SELECTED)
                try:
                    deliver.apply(root, out, [(job, companion)], progress=lambda _line: None)
                except (deliver.OutputError, OSError) as e:
                    raise MoveError(f"Could not move {Path(path).name}: {e}") from e
            if back:
                with cat.db:
                    cat.db.execute("UPDATE cull SET decision = ?, best = ?, unsure = ?, reason = ?, changed = 0 "
                                   "WHERE id = ?", (decision, before[1], before[2], before[3], pid))
                BEFORE_MOVE.pop((shoot, pid), None)
            else:
                cat.move([pid], decision, best=bool(best))
    result = counts(shoot)
    with JOB.lock:
        JOB.result.update(result)
    return {"ok": True, "id": pid, "to": to, "result": result}


def brand() -> dict:
    info = {"name": "", "tagline": "", "logo": ""}
    try:
        info.update(json.loads((BRAND_DIR / "brand.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    for name in ("logo.png", "logo.jpg", "logo.svg"):
        if (BRAND_DIR / name).exists():
            info["logo"] = f"/brand/{name}"
            break
    for name in ("background.jpg", "background.png"):          # optional: a real mandap photo behind the scene
        if (BRAND_DIR / name).exists():
            info["background"] = f"/brand/{name}"
            break
    return info


def native_folder_dialog(start: str = "") -> str:
    """The Mac's own folder dialog, over KeepSake's window (pywebview runs it on the main thread)."""
    import webview
    kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
    if kind is None:
        kind = webview.FOLDER_DIALOG
    near = Path(start) if start else None
    while near is not None and not near.is_dir():           # the dialog opens where the folder would be
        near = near.parent if near.parent != near else None
    try:
        picked = WINDOW.create_file_dialog(kind, directory=str(near or ""))
    except Exception:  # noqa: BLE001 - a dialog that cannot open simply picks nothing
        return ""
    if not picked:
        return ""
    return str(picked[0] if isinstance(picked, (list, tuple)) else picked)


def choose_folder(title: str, start: str = "") -> str:
    """The system's folder dialog (in its own small process, so the server never blocks)."""
    if WINDOW is not None:
        return native_folder_dialog(start)
    if FROZEN:
        return ""                       # the packaged app has no separate Python: the path can be typed instead
    code = ("import sys, tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw()\n"
            "r.attributes('-topmost', True)\n"
            "p = filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2] or None, mustexist=False)\n"
            "print(p or '')\n")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run([PYTHON, "-c", code, title, start], capture_output=True, text=True, timeout=600,
                           creationflags=flags)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    path = r.stdout.strip()
    return str(Path(path)) if path else ""


def open_folder(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))                                  # noqa: S606 (our own output folder)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# --- The local web server --------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "KeepSake"
    token = ""

    def log_message(self, *args) -> None:        # quiet console
        pass

    def allowed(self) -> bool:
        if self.headers.get("X-KeepSake") == self.token:
            return True
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return "ks" in jar and jar["ks"].value == self.token

    def send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if kind.startswith("application/json") else "max-age=3600")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, data, code: int = 200) -> None:
        self.send(code, json.dumps(data).encode(), "application/json; charset=utf-8")

    def file(self, path: Path) -> None:
        if not path.is_file():
            return self.send(404, b"not found", "text/plain")
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".otf", ".ttf"):
            kind = "font/otf" if path.suffix == ".otf" else "font/ttf"
        elif path.suffix == ".js":
            kind = "text/javascript"                 # Windows can say text/plain, which a module refuses
        self.send(200, path.read_bytes(), kind)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            if q.get("t", [""])[0] != self.token and not self.allowed():
                return self.send(403, b"Open KeepSake with:  ks app", "text/plain")
            body = (UI_DIR / "index.html").read_bytes()
            return self.send(200, body, "text/html; charset=utf-8",
                             {"Set-Cookie": f"ks={self.token}; Path=/; HttpOnly; SameSite=Strict"})
        if not self.allowed():
            return self.send(403, b"forbidden", "text/plain")
        if url.path.startswith("/ui/"):
            target = (UI_DIR / unquote(url.path[4:])).resolve()
            if UI_DIR.resolve() in target.parents:
                return self.file(target)
            return self.send(404, b"not found", "text/plain")
        if url.path.startswith("/brand/"):
            name = Path(unquote(url.path[7:])).name
            return self.file(BRAND_DIR / name)
        m = re.match(r"^/photo/([\w\-]+)/(\d+)(?:/(big))?$", url.path)
        if m:
            kind = "analysis" if m.group(3) else "thumbs"
            return self.file(cache_path(CACHE_DIR / m.group(1), kind, int(m.group(2))))
        if url.path == "/api/state":
            return self.json(JOB.state())
        if url.path == "/api/home":
            weddings = recent_weddings()
            sample = []
            if weddings:                      # the opening's photos: a wedding whose small copies are here
                dbs = [db for db in catalogs() if has_thumbs(db.parent)] or catalogs()
                sample = thumbs(random.choice(dbs).parent.name, "any", 48)
            return self.json({"brand": brand(), "recent": weddings, "sample": sample,
                              "output_root": str(OUTPUT_ROOT), "computer": platform.node()})
        if url.path == "/api/gallery":
            shoot = JOB.shoot_name or ""
            return self.json({"photos": gallery(shoot, q.get("set", ["selected"])[0]) if shoot else []})
        if url.path == "/api/frames":
            shoot = JOB.shoot_name or ""
            try:
                pid = int(q.get("id", ["-1"])[0])
            except ValueError:
                pid = -1
            return self.json(frames(shoot, pid) if shoot else {"id": pid, "shot": [], "moment": []})
        if url.path == "/api/thumbs":
            shoot = JOB.shoot_name or ""
            which = q.get("set", ["any"])[0]
            n = min(int(q.get("n", ["40"])[0]), 400)
            return self.json({"photos": thumbs(shoot, which, n) if shoot else []})
        return self.send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self.allowed():
            return self.send(403, b"forbidden", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.json({"error": "bad request"}, 400)
        path = urlparse(self.path).path
        if path == "/api/choose":
            title = "Choose the wedding's photo folder" if data.get("kind") != "output" else "Choose where to save"
            return self.json({"path": choose_folder(title, data.get("start", ""))})
        if path == "/api/check":
            folder = Path(data.get("folder", "").strip().strip('"'))
            ok = bool(data.get("folder")) and folder.is_dir()
            name = wedding_name(folder) if ok else ""
            out = OUTPUT_ROOT / name if ok else None
            free = shutil.disk_usage(out.anchor if out else OUTPUT_ROOT.anchor).free if ok else 0
            prepared = ok and shoot_for(folder).db.exists()
            photos, size = 0, 0
            if prepared:
                try:
                    photos, size = read_db(shoot_for(folder).db, "SELECT COUNT(*), COALESCE(SUM(bytes), 0) "
                                                                 "FROM photos WHERE status = 'done'")[0]
                except sqlite3.Error:
                    pass
            not_gb = size * 0.9 / 1e9                                   # about 9 in 10 photos are not selected
            return self.json({"ok": ok, "name": name, "output": str(out or ""), "prepared": prepared,
                              "free_gb": round(free / 1e9, 1), "photos": photos, "total_gb": round(size / 1e9, 1),
                              "not_selected_gb": round(not_gb, 1),
                              "not_selected_time": rough(not_gb * 1000 / COPY_MB_S) if size else "",
                              "error": "" if ok else "Folder not found. Is the drive connected?"})
        if path == "/api/start":
            folder = Path(data.get("folder", "").strip().strip('"'))
            if not folder.is_dir():
                return self.json({"error": "Folder not found. Is the drive connected?"}, 400)
            copy = data.get("copy", True) is not False          # False: just view, nothing is copied
            output = Path(data.get("output") or OUTPUT_ROOT / wedding_name(folder)) if copy else None
            keep = data.get("keep")
            keep = keep if keep in ("wide", "shortlist") else float(keep) if keep not in (None, "", "studio") else None
            try:
                JOB.start(folder, output, keep, bool(data.get("copy_not_selected", True)), copy=copy)
            except RuntimeError as e:
                return self.json({"error": str(e)}, 409)
            return self.json(JOB.state())
        if path == "/api/stop":
            JOB.stop()
            return self.json(JOB.state())
        if path == "/api/reset":
            if JOB.status != "running":
                JOB.reset()
            return self.json(JOB.state())
        if path == "/api/move":
            try:
                moved = move_photo(int(data.get("id", -1)), str(data.get("to", "")))
                db = CACHE_DIR / (JOB.shoot_name or "") / "catalog.db"
                moved["photo"] = next(iter(entries(db, JOB.shoot_name or "", "WHERE c.id = ?", (moved["id"],),
                                                   frames_of=True)), None)
                return self.json(moved)
            except (MoveError, ValueError, TypeError) as e:
                return self.json({"error": str(e) or "bad request"}, 400)
        if path == "/api/swap":
            try:
                return self.json(swap_frame(int(data.get("id", -1)), int(data.get("replace", -1))))
            except (MoveError, ValueError, TypeError) as e:
                return self.json({"error": str(e) or "bad request"}, 400)
        if path == "/api/open":
            which = data.get("which")
            target = {"selected": JOB.result.get("selected_dir"), "maybe": JOB.result.get("maybe_dir"),
                      "not_selected": JOB.result.get("not_selected_dir"),
                      "output": JOB.result.get("output")}.get(which)
            if target and Path(target).is_dir():
                open_folder(Path(target))
                return self.json({"ok": True})
            return self.json({"error": "That folder is not there yet."}, 404)
        return self.json({"error": "unknown"}, 404)


def make_server(port: int = 0) -> ThreadingHTTPServer:
    Handler.token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def app_window(url: str) -> subprocess.Popen | None:
    """KeepSake in its own window: Edge or Chrome in app mode (no address bar)."""
    profile = DATA_DIR / "app_window"
    candidates = []
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
            candidates += [Path(base) / "Microsoft/Edge/Application/msedge.exe",
                           Path(base) / "Google/Chrome/Application/chrome.exe"]
    elif sys.platform == "darwin":
        candidates += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                       Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")]
    else:
        candidates += [Path(p) for p in (shutil.which("chromium") or "", shutil.which("google-chrome") or "") if p]
    for exe in candidates:
        if exe.is_file():
            return subprocess.Popen([str(exe), f"--app={url}", f"--user-data-dir={profile}",
                                     "--window-size=1440,900", "--no-first-run", "--no-default-browser-check",
                                     "--autoplay-policy=no-user-gesture-required"])     # the opening's sounds
    webbrowser.open(url)
    return None


def native_window(url: str) -> bool:
    """The Mac app's own window (WebKit, via pywebview): its Dock icon and menu, the system's
    folder dialog, and its settings kept between runs. Returns when the window is closed; False
    when pywebview is not installed."""
    global WINDOW
    try:
        import webview
    except ImportError:
        return False
    WINDOW = webview.create_window("KeepSake", url, width=1440, height=900, min_size=(1100, 700),
                                   background_color="#0b0907", text_select=False)
    storage = DATA_DIR / "app_window"
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(private_mode=False, storage_path=str(storage))
    return True


def finish_then_stop() -> None:
    while JOB.status == "running":          # the window was closed while working: finish first
        print("The window was closed; KeepSake finishes the current wedding first...", end="\r")
        time.sleep(2)


def main() -> int:
    server = make_server(int(os.environ.get("KEEPSAKE_APP_PORT", "0")))
    url = f"http://127.0.0.1:{server.server_port}/?t={Handler.token}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("KeepSake is open in its own window.")
    print("Keep this window open while KeepSake works; close the KeepSake window when you are done.")
    print(f"(If the window did not open, paste this into Edge or Chrome:  {url} )", flush=True)
    own_window = sys.platform == "darwin" and (FROZEN or os.environ.get("KEEPSAKE_WINDOW") == "native")
    if "--no-window" not in sys.argv and own_window:
        try:
            if native_window(url):
                finish_then_stop()
                server.shutdown()
                return 0
        except KeyboardInterrupt:
            JOB.stop()
            server.shutdown()
            return 0
    window = None if "--no-window" in sys.argv else app_window(url)
    try:
        if window is not None:
            window.wait()
            finish_then_stop()
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        JOB.stop()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
