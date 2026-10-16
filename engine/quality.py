"""Phase 2: picture-quality measurements, made from the analysis copies (no AI model).

Everything is measured on the upright 1620 px copy Phase 1 saved on the laptop, so the shoot
drive is not needed. The numbers are raw measurements; what counts as "too blurry" or "too
dark" is decided relative to the rest of the shoot (see relative_scores), because a dim
sangeet and a sunny haldi, or a 24 MP and a 45 MP camera, cannot share one fixed threshold.

Sharpness (Laplacian variance, the standard focus measure)
  sharp_global   over the whole frame
  sharp_peak     in the sharpest part of the frame: the image is cut into a 12 x 8 grid of
                 tiles and the 95th-percentile tile is taken. A portrait at f/1.8 has a soft
                 background but a sharp subject, so this is the number that tells a missed
                 focus or a shaken photo from a deliberately soft background.
  peak_contrast  how much detail those sharpest tiles hold (grey-level spread)
  noise          sensor noise (Immerkaer's method) in the flattest tiles: high-ISO grain
                 raises Laplacian variance, so sharpness is read against it
  motion         0-1: how much the edges all run one way (structure-tensor coherence);
                 camera shake smears the picture in one direction
Exposure (luma, 0-255)
  luma_mean, luma_p01, luma_p99   brightness and the dark and bright ends
  clip_low, clip_high             share of pixels crushed to black / blown to white
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

QUALITY_VERSION = 1
GRID = (8, 12)                     # tile rows x columns for a landscape photo

FEATURES = ("sharp_global", "sharp_peak", "peak_contrast", "noise", "motion",
            "luma_mean", "luma_p01", "luma_p99", "clip_low", "clip_high")

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def read_gray(path: Path) -> np.ndarray:
    """The analysis copy as 8-bit grey (luma), read without decoding colour.

    np.fromfile + imdecode instead of cv2.imread, which fails on Windows for folder names
    with non-English characters.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OSError(f"cannot decode {path.name}")
    return gray


def _tiles(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """View `a` as rows x cols tiles: shape (rows, cols, tile_h, tile_w)."""
    th, tw = a.shape[0] // rows, a.shape[1] // cols
    a = a[:th * rows, :tw * cols]
    return a.reshape(rows, th, cols, tw).swapaxes(1, 2)


def measure_gray(gray: np.ndarray) -> dict:
    """All quality numbers for one grey image."""
    rows, cols = GRID if gray.shape[1] >= gray.shape[0] else GRID[::-1]
    g = gray.astype(np.float32)

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    tile_sharp = _tiles(lap, rows, cols).var(axis=(2, 3)).ravel()
    tile_spread = _tiles(g, rows, cols).std(axis=(2, 3)).ravel()
    order = np.argsort(tile_sharp)
    peak = float(np.percentile(tile_sharp, 95))
    top = order[-max(3, len(order) // 20):]                 # the sharpest ~5% of tiles

    # Immerkaer noise: sqrt(pi/2) / 6 * mean |I * M|, read in the flattest tiles
    residual = np.abs(cv2.filter2D(g, cv2.CV_32F, _IMMERKAER))
    tile_noise = _tiles(residual, rows, cols).mean(axis=(2, 3)).ravel() * math.sqrt(math.pi / 2) / 6
    flat = np.argsort(tile_spread)[:max(3, len(order) // 10)]

    # structure tensor on a half-size copy: coherence 1 = every edge runs the same way
    small = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    spread = math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    motion = spread / (jxx + jyy) if jxx + jyy > 0 else 0.0

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / hist.sum()
    return {
        "sharp_global": float(lap.var()),
        "sharp_peak": peak,
        "peak_contrast": float(np.median(tile_spread[top])),
        "noise": float(np.median(tile_noise[flat])),
        "motion": float(motion),
        "luma_mean": float(np.dot(hist, np.arange(256)) / hist.sum()),
        "luma_p01": float(np.searchsorted(cdf, 0.01)),
        "luma_p99": float(np.searchsorted(cdf, 0.99)),
        "clip_low": float(cdf[5]),
        "clip_high": float(1 - cdf[249]),
    }


def measure_task(task: tuple) -> dict:
    """Worker job: (photo id, analysis copy path) -> {'id', 'ok', features...}. Never raises."""
    photo_id, path = task
    try:
        result = measure_gray(read_gray(Path(path)))
        result.update(id=photo_id, ok=True)
        return result
    except Exception as e:  # a damaged cache file: record it, carry on
        return {"id": photo_id, "ok": False, "error": f"{type(e).__name__}: {e}"}


# --- Relative scores ---------------------------------------------------------------

def robust_z(values: np.ndarray) -> np.ndarray:
    """How many 'typical spreads' each value sits from the group's middle (median / MAD).

    Unlike mean and standard deviation, a few extreme photos cannot drag these around.
    """
    middle = np.median(values)
    mad = np.median(np.abs(values - middle)) * 1.4826
    if not mad > 0:
        return np.zeros_like(values)
    return (values - middle) / mad


def relative_scores(rows: list[dict], min_group: int = 30) -> dict[int, dict]:
    """Per photo, sharpness and brightness relative to comparable photos of the same shoot.

    sharp_rel   log sharpness against the same camera body (sensor, lens kit and the
                photographer's habits); -3 means far softer than that camera usually is
    luma_rel    brightness against the same event (folder), since a dark reception and a
                bright haldi are each fine for what they are
    Groups smaller than min_group are compared with the whole shoot instead.
    """
    out: dict[int, dict] = {r["id"]: {} for r in rows}
    if not rows:
        return out

    def by(key, value, target):
        groups: dict = {}
        for r in rows:
            groups.setdefault(r.get(key) or "", []).append(r)
        everyone = np.array([value(r) for r in rows], dtype=np.float64)
        everyone_z = robust_z(everyone)
        for i, r in enumerate(rows):
            out[r["id"]][target] = float(everyone_z[i])
        for members in groups.values():
            if len(members) < min_group:
                continue
            z = robust_z(np.array([value(r) for r in members], dtype=np.float64))
            for r, v in zip(members, z):
                out[r["id"]][target] = float(v)

    by("body", lambda r: math.log(max(r["sharp_peak"], 1e-3)), "sharp_rel")
    by("event", lambda r: r["luma_mean"], "luma_rel")
    return out


# --- How well a measurement matches the studio's choices ----------------------------

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Chance that a random selected photo scores higher than a random rejected one.

    0.5 = no better than a coin, 1.0 = perfect. (Mann-Whitney U, ties counted half.)
    """
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):                       # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
