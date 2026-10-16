"""The shoot catalog: one SQLite database per shoot, on this laptop.

One row per photo, holding where it is, its camera details and how far the engine got with it.
The status column is what makes ingest resumable: a crash, a Ctrl+C or an unplugged drive
loses at most the last couple of seconds of work.

    new      found, not read yet (or changed since it was read)
    done     analysis copy and thumbnail made
    failed   could not be read; the error column says why (retried on the next run)
    empty    0-byte file
    missing  was in the shoot folder before, is not there now
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS info (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,   -- main file, relative to the shoot folder, '/' separators
    companion   TEXT,                   -- JPEG/HEIF saved together with a RAW: same photo
    folder      TEXT NOT NULL,          -- sub-folder, e.g. '27.08.2026 Welcome & Bhajan/Shashank'
    kind        TEXT NOT NULL,          -- raw / jpeg / heif
    bytes       INTEGER NOT NULL,       -- file size (both files for a RAW+JPEG pair)
    mtime_ms    INTEGER NOT NULL,       -- last modified: a changed file is read again
    status      TEXT NOT NULL DEFAULT 'new',
    error       TEXT,
    camera      TEXT,                   -- 'NIKON Z 9'
    serial      TEXT,                   -- camera body serial number
    body        TEXT,                   -- one physical camera: 'NIKON Z 9 #7600802'
    lens        TEXT,
    taken       TEXT,                   -- camera clock, 'YYYY-MM-DD HH:MM:SS.ss'
    taken_s     REAL,                   -- the same in seconds, for sorting and time gaps
    tz          TEXT,                   -- the camera's time-zone setting, '+05:30'
    width       INTEGER,                -- full photo, upright
    height      INTEGER,
    orientation INTEGER,                -- EXIF orientation of the original (cache is upright)
    iso         INTEGER,
    shutter     REAL,                   -- seconds
    aperture    REAL,                   -- f-number
    focal       REAL,                   -- mm
    flash       INTEGER,                -- EXIF flash code, bit 0 = fired
    preview     TEXT,                   -- where the analysis copy came from (see imaging.py)
    preview_w   INTEGER,
    preview_h   INTEGER,
    read_mb     REAL,                   -- read from the drive for this photo
    ms          REAL                    -- time taken for this photo
);
CREATE INDEX IF NOT EXISTS photos_status ON photos(status);
CREATE INDEX IF NOT EXISTS photos_taken ON photos(taken_s);

-- Phase 2: picture-quality measurements (see engine/quality.py), one row per photo
CREATE TABLE IF NOT EXISTS quality (
    id            INTEGER PRIMARY KEY REFERENCES photos(id),
    version       INTEGER NOT NULL,
    ok            INTEGER NOT NULL,       -- 0 = the analysis copy could not be read
    error         TEXT,
    sharp_global  REAL, sharp_peak REAL, peak_contrast REAL, noise REAL, motion REAL,
    luma_mean     REAL, luma_p01 REAL, luma_p99 REAL, clip_low REAL, clip_high REAL,
    sharp_rel     REAL,                   -- sharpness against the same camera body (robust z)
    luma_rel      REAL                    -- brightness against the same event (robust z)
);

-- Phase 2b: faces (see engine/faces.py). face_scan has one row per photo looked at,
-- faces one row per face found (n = 0 is the largest face in the photo).
CREATE TABLE IF NOT EXISTS face_scan (
    id         INTEGER PRIMARY KEY REFERENCES photos(id),
    version    INTEGER NOT NULL,
    ok         INTEGER NOT NULL,
    error      TEXT,
    faces      INTEGER,                -- faces found
    eyes_read  INTEGER                 -- 1 when the eye reader (MediaPipe) was available
);
CREATE TABLE IF NOT EXISTS faces (
    photo_id   INTEGER NOT NULL REFERENCES photos(id),
    n          INTEGER NOT NULL,
    x REAL, y REAL, w REAL, h REAL,    -- box in the analysis copy, pixels
    score      REAL,                   -- how sure YuNet is that this is a face
    rel_w      REAL,                   -- face width / photo's long side
    sharp      REAL, contrast REAL,    -- sharpness and detail of the middle of the face
    subject    INTEGER,                -- 1 = one of the photo's main people (large, in focus)
    read       INTEGER,                -- 1 = eyes and expression were read
    blink_l REAL, blink_r REAL,        -- 0 = eye wide open, 1 = shut (the person's left/right)
    squint REAL, look_down REAL, smile REAL, jaw_open REAL,
    yaw REAL, pitch REAL, roll REAL,   -- head turn, nod, tilt in degrees
    eyes       TEXT,                   -- open / closed / unsure / down / side (see faces.py)
    PRIMARY KEY (photo_id, n)
);

-- The studio's own choices, when known (from ks match): 1 = selected, 0 = not selected
CREATE TABLE IF NOT EXISTS labels (
    id        INTEGER PRIMARY KEY REFERENCES photos(id),
    selected  INTEGER NOT NULL,
    source    TEXT
);
"""

QUALITY_COLUMNS = ("sharp_global", "sharp_peak", "peak_contrast", "noise", "motion",
                   "luma_mean", "luma_p01", "luma_p99", "clip_low", "clip_high")

RESULT_COLUMNS = ("status", "error", "camera", "serial", "body", "lens", "taken", "taken_s",
                  "tz", "width", "height", "orientation", "iso", "shutter", "aperture", "focal",
                  "flash", "preview", "preview_w", "preview_h", "read_mb", "ms")


@dataclass(frozen=True)
class ShootFile:
    """One photo found on the drive, ready to go in the catalog."""
    path: str              # relative, '/' separators
    companion: str | None
    folder: str
    kind: str
    bytes: int
    mtime_ms: int


@dataclass
class SyncCounts:
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    missing: int = 0
    back: int = 0          # missing before, found again


@dataclass
class Body:
    name: str
    photos: int
    first: str | None
    last: str | None
    clock_off: bool        # capture times far from every other camera: clock probably not set


class Catalog:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.db = sqlite3.connect(str(db_path))
        self.db.execute("PRAGMA journal_mode=WAL")      # crash-safe, readers never block
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._add_missing_columns()

    def close(self) -> None:
        self.db.close()

    def _add_missing_columns(self) -> None:
        """Catalogs made by an older KeepSake get the columns added since (data is kept)."""
        added = {"faces": [("subject", "INTEGER"), ("eyes", "TEXT")]}
        for table, columns in added.items():
            have = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns:
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
        self.db.commit()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- info --------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM info WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key: str, value) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)",
                            (key, None if value is None else str(value)))

    # --- syncing with the drive ----------------------------------------------------------
    def sync(self, files: list[ShootFile]) -> SyncCounts:
        """Bring the catalog in line with what is on the drive now."""
        counts = SyncCounts()
        known = {row[0]: row[1:] for row in self.db.execute(
            "SELECT path, id, bytes, mtime_ms, companion, status FROM photos")}
        seen = set()
        inserts, resets, backs = [], [], []
        for f in files:
            seen.add(f.path)
            old = known.get(f.path)
            if old is None:
                inserts.append((f.path, f.companion, f.folder, f.kind, f.bytes, f.mtime_ms))
                counts.new += 1
                continue
            photo_id, size, mtime, companion, status = old
            if (size, mtime, companion) != (f.bytes, f.mtime_ms, f.companion):
                resets.append((f.companion, f.folder, f.kind, f.bytes, f.mtime_ms, photo_id))
                counts.changed += 1
            elif status == "missing":
                backs.append((photo_id,))
                counts.back += 1
            else:
                counts.unchanged += 1
        gone = [(row[0],) for path, row in known.items() if path not in seen and row[4] != "missing"]
        counts.missing = len(gone)
        with self.db:
            self.db.executemany(
                "INSERT INTO photos (path, companion, folder, kind, bytes, mtime_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)", inserts)
            self.db.executemany(
                "UPDATE photos SET companion = ?, folder = ?, kind = ?, bytes = ?, mtime_ms = ?, "
                "status = 'new', error = NULL WHERE id = ?", resets)
            self.db.executemany("UPDATE photos SET status = 'new' WHERE id = ?", backs)
            self.db.executemany("UPDATE photos SET status = 'missing' WHERE id = ?", gone)
        return counts

    def redo_all(self) -> None:
        with self.db:
            self.db.execute("UPDATE photos SET status = 'new', error = NULL "
                            "WHERE status != 'missing'")

    def pending(self, limit: int | None = None) -> list[tuple[int, str, str | None, str, int]]:
        """(id, path, companion, kind, bytes) still to do, in folder order (kind to the drive)."""
        sql = ("SELECT id, path, companion, kind, bytes FROM photos "
               "WHERE status IN ('new', 'failed') ORDER BY path")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return list(self.db.execute(sql))

    def save(self, results: list[dict]) -> None:
        """Store worker results: one dict per photo with 'id' and some of RESULT_COLUMNS."""
        if not results:
            return
        with self.db:
            for r in results:
                cols = [c for c in RESULT_COLUMNS if c in r]
                sets = ", ".join(f"{c} = ?" for c in cols)
                self.db.execute(f"UPDATE photos SET {sets} WHERE id = ?",
                                [r[c] for c in cols] + [r["id"]])

    # --- Phase 2: quality ---------------------------------------------------------------
    def quality_pending(self, version: int) -> list[tuple[int, str]]:
        """(id, event) of read photos with no quality row of this version yet."""
        return list(self.db.execute(
            "SELECT p.id, p.folder FROM photos p LEFT JOIN quality q ON q.id = p.id "
            "WHERE p.status = 'done' AND (q.id IS NULL OR q.version < ?) ORDER BY p.id",
            (version,)))

    def save_quality(self, results: list[dict], version: int) -> None:
        cols = ("id", "version", "ok", "error") + QUALITY_COLUMNS
        with self.db:
            self.db.executemany(
                f"INSERT OR REPLACE INTO quality ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})",
                [[r["id"], version, int(r.get("ok", False)), r.get("error")]
                 + [r.get(c) for c in QUALITY_COLUMNS] for r in results])

    def quality_rows(self) -> list[dict]:
        """Every measured photo with its body and event, for the relative scores."""
        cur = self.db.execute(
            "SELECT q.id, COALESCE(p.body, p.camera, ''), p.folder, "
            + ", ".join(f"q.{c}" for c in QUALITY_COLUMNS)
            + " FROM quality q JOIN photos p ON p.id = q.id WHERE q.ok = 1 AND p.status = 'done'")
        rows = []
        for r in cur:
            d = dict(zip(QUALITY_COLUMNS, r[3:]))
            d.update(id=r[0], body=r[1], event=(r[2] or "").split("/")[0])
            rows.append(d)
        return rows

    def save_relative(self, scores: dict[int, dict]) -> None:
        with self.db:
            self.db.executemany("UPDATE quality SET sharp_rel = ?, luma_rel = ? WHERE id = ?",
                                [(s.get("sharp_rel"), s.get("luma_rel"), pid)
                                 for pid, s in scores.items()])

    def faces_pending(self, version: int, eyes_available: bool) -> list[int]:
        """Read photos not yet scanned for faces with this version (or scanned before the eye
        reader was installed, when it is now)."""
        sql = ("SELECT p.id FROM photos p LEFT JOIN face_scan s ON s.id = p.id "
               "WHERE p.status = 'done' AND (s.id IS NULL OR s.version < ? OR s.ok = 0"
               + (" OR s.eyes_read = 0" if eyes_available else "") + ") ORDER BY p.id")
        return [r[0] for r in self.db.execute(sql, (version,))]

    def save_faces(self, results: list[dict], version: int) -> None:
        from .faces import FACE_COLUMNS
        cols = ", ".join(FACE_COLUMNS)
        marks = ", ".join("?" for _ in FACE_COLUMNS)
        with self.db:
            for r in results:
                self.db.execute("DELETE FROM faces WHERE photo_id = ?", (r["id"],))
                self.db.executemany(
                    f"INSERT INTO faces (photo_id, n, {cols}) VALUES (?, ?, {marks})",
                    [[r["id"], n] + [f.get(c) for c in FACE_COLUMNS]
                     for n, f in enumerate(r.get("faces", []))])
                self.db.execute(
                    "INSERT OR REPLACE INTO face_scan (id, version, ok, error, faces, eyes_read) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (r["id"], version, int(r.get("ok", False)), r.get("error"),
                     len(r.get("faces", [])), int(bool(r.get("eyes")))))

    def refresh_eyes(self, rule) -> int:
        """Re-judge every read face with the current eye rule, from the stored scores (no need
        to read the photos again). Returns how many verdicts changed."""
        cols = ("read", "yaw", "pitch", "look_down", "blink_l", "blink_r", "smile", "squint")
        changes = []
        for row in self.db.execute(f"SELECT photo_id, n, eyes, {', '.join(cols)} FROM faces"):
            verdict = rule(dict(zip(cols, row[3:])))
            if verdict != row[2]:
                changes.append((verdict, row[0], row[1]))
        with self.db:
            self.db.executemany("UPDATE faces SET eyes = ? WHERE photo_id = ? AND n = ?", changes)
        return len(changes)

    def save_labels(self, selected_by_path: dict[str, int], source: str) -> int:
        """Store the studio's choices against catalog photos; returns how many matched."""
        ids = dict(self.db.execute("SELECT path, id FROM photos"))
        rows = [(ids[p], s, source) for p, s in selected_by_path.items() if p in ids]
        with self.db:
            self.db.execute("DELETE FROM labels")
            self.db.executemany("INSERT INTO labels (id, selected, source) VALUES (?, ?, ?)", rows)
        return len(rows)

    def labelled_quality(self) -> list[dict]:
        cur = self.db.execute(
            "SELECT l.selected, q.sharp_rel, q.luma_rel, " + ", ".join(f"q.{c}" for c in QUALITY_COLUMNS)
            + " FROM labels l JOIN quality q ON q.id = l.id WHERE q.ok = 1")
        names = ("selected", "sharp_rel", "luma_rel") + QUALITY_COLUMNS
        return [dict(zip(names, r)) for r in cur]

    # --- reports -------------------------------------------------------------------
    def counts(self) -> Counter:
        return Counter(dict(self.db.execute("SELECT status, COUNT(*) FROM photos GROUP BY status")))

    def total(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM photos WHERE status != 'missing'").fetchone()[0]

    def failures(self) -> list[tuple[str, str]]:
        return list(self.db.execute(
            "SELECT path, error FROM photos WHERE status IN ('failed', 'empty') ORDER BY path"))

    def preview_sources(self) -> Counter:
        return Counter(dict(self.db.execute(
            "SELECT preview, COUNT(*) FROM photos WHERE status = 'done' GROUP BY preview")))

    def read_totals(self) -> tuple[float, float]:
        """(MB read from the drive, MB the files hold) for done photos with a known read size."""
        row = self.db.execute("SELECT SUM(read_mb), SUM(bytes) / 1048576.0 FROM photos "
                              "WHERE status = 'done' AND read_mb IS NOT NULL").fetchone()
        return (row[0] or 0.0, row[1] or 0.0)

    def no_time(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM photos WHERE status = 'done' "
                               "AND taken_s IS NULL").fetchone()[0]

    def bodies(self, gap_days: float = 7.0) -> list[Body]:
        """Each camera body with its photo count and first/last capture time.

        All capture times are split wherever nothing was shot for more than `gap_days`; the part
        with the most photos is the shoot. A body whose typical time falls outside it is flagged:
        its clock was probably never set (e.g. 2015-01-01). A camera used on only one day of a
        three-day wedding is inside it, so it is not flagged.
        """
        rows = list(self.db.execute(
            "SELECT COALESCE(body, '(unknown camera)'), taken_s, taken FROM photos "
            "WHERE status = 'done' ORDER BY taken_s"))
        times = [t for _, t, _ in rows if t is not None]          # sorted
        main = None
        if times:
            parts, first = [], 0
            for i in range(1, len(times) + 1):
                if i == len(times) or times[i] - times[i - 1] > gap_days * 86400:
                    parts.append((i - first, times[first], times[i - 1]))
                    first = i
            _, lo, hi = max(parts)
            main = (lo, hi)
        per: dict[str, list] = {}
        for name, t, text in rows:
            per.setdefault(name, []).append((t, text))
        bodies = []
        for name, items in per.items():
            ts = [(t, text) for t, text in items if t is not None]
            off = False
            if ts and main is not None:
                typical = ts[len(ts) // 2][0]
                off = not main[0] <= typical <= main[1]
            bodies.append(Body(name, len(items), ts[0][1] if ts else None,
                               ts[-1][1] if ts else None, off))
        bodies.sort(key=lambda b: -b.photos)
        return bodies


def now_text() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
