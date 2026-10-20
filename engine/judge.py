"""Phase 7: Claude as a second editor (needs the internet and an Anthropic API key; paid).

KeepSake's own models cannot see a half blink or a head turning away reliably. Claude can look at
the photos. For each SHOT (near-identical frames, engine/shots.py) Claude gets small copies of
its frames (IMAGE_PX on the long side, about 925 image tokens each) and the editor's rules
(engine/judge_prompt.md), and answers in a fixed JSON form (SCHEMA):
    per frame: eyes (all open / half closed / closed / looking down / no faces), head turned,
               main face sharp, other problems
    best_frame, deliver_score 1-10, a one-line reason.

Money: every request is priced before anything is sent (estimate), the caller asks the user, and
answers are saved (data/judge/<shoot>/<model>-<prompt version>.jsonl) so a second run never pays
for the same shot twice. By default requests go through the Message Batches API: half price,
usually done within minutes, at most 24 hours. --now sends them one by one at full price.

The key is read from data/anthropic_key.txt (or ANTHROPIC_API_KEY); it never goes in Git or logs.
Only the small copies leave the laptop, never the originals.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR

API = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5",
          "opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}
NAMES = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}
PRICE = {"haiku": (0.10, 0.50), "sonnet": (2.0, 10.0), "opus": (4.0, 20.0), "fable": (10.0, 50.0)}  # $/M tokens
IMAGE_PX = 1024            # long side of the copy Claude sees
MAX_FRAMES = 8             # frames of one shot in one request (the rest are not shown)
OUTPUT_GUESS = 400         # output tokens per request incl. some thinking, plus OUTPUT_PER_FRAME per frame
OUTPUT_PER_FRAME = 60      # (only for the estimate shown before anything is sent)
MAX_TOKENS = 3000          # thinking + answer
BATCH_MB = 150             # split bigger jobs into several batches (the API allows 256 MB)
POLL_S = 20
KEY_FILE = DATA_DIR / "anthropic_key.txt"
JUDGE_DIR = DATA_DIR / "judge"
PROMPT_FILE = Path(__file__).with_name("judge_prompt.md")

# Round 2, the whole sequence: Claude compares ~SCENE_SIZE consecutive shots (their best frames)
# and scores each 0-100 the way the studio chooses (engine/judge_scene_prompt.md): one photo per
# beat of the story, the best expression, family and groups, spread over the event.
SCENE_PROMPT_FILE = Path(__file__).with_name("judge_scene_prompt.md")
SCENE_PX = 768             # smaller copies: blinks were checked in round 1
SCENE_SIZE = 40            # shots per request
SCENE_MAX_TOKENS = 12000
SCENE_OUTPUT = (1500, 30)  # output tokens per request + per shot (for the estimate)

EYES = ["all open", "some half closed", "some closed", "looking down", "no faces"]
SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "frame": {"type": "integer", "description": "frame number, 1 = first"},
                "eyes": {"type": "string", "enum": EYES},
                "head_turned": {"type": "boolean"},
                "main_face_sharp": {"type": "boolean"},
                "problems": {"type": "string", "description": "a few words, empty if none"}},
            "required": ["frame", "eyes", "head_turned", "main_face_sharp", "problems"],
            "additionalProperties": False}},
        "best_frame": {"type": "integer", "description": "the frame to deliver, 1 = first"},
        "deliver_score": {"type": "integer", "description": "1 to 10"},
        "reason": {"type": "string", "description": "one short sentence"}},
    "required": ["frames", "best_frame", "deliver_score", "reason"],
    "additionalProperties": False}


SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "shot": {"type": "integer", "description": "shot number, 1 = first"},
                "keep": {"type": "integer", "description": "0 to 100"},
                "why": {"type": "string", "description": "a few words"}},
            "required": ["shot", "keep", "why"],
            "additionalProperties": False}}},
    "required": ["shots"],
    "additionalProperties": False}


class JudgeError(RuntimeError):
    pass


# --- Key -----------------------------------------------------------------------------

def load_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8-sig").strip().strip('"')
    if not key:
        raise JudgeError(f"No API key found. Save it (one line, starts with sk-ant-) in Notepad as:  {KEY_FILE}")
    if not key.startswith("sk-ant-") or any(c.isspace() for c in key):
        raise JudgeError(f"The key in {KEY_FILE} does not look right: it should be one line starting with sk-ant-")
    return key


# --- Requests ------------------------------------------------------------------------

def small_jpeg(path: Path, px: int = IMAGE_PX) -> tuple[bytes, int, int]:
    """The copy Claude sees: upright, px on the long side, JPEG (the file itself is only read)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((px, px), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue(), im.width, im.height


def image_tokens(w: int, h: int) -> int:
    return math.ceil(w / 28) * math.ceil(h / 28)


def prompt_text() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def prompt_version() -> str:
    """Changes whenever the rules, the answer form or the image size change (new answers needed)."""
    text = prompt_text() + json.dumps(SCHEMA, sort_keys=True) + str(IMAGE_PX)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def request_id(names: list[str]) -> str:
    return "s" + hashlib.sha1("|".join(names).encode()).hexdigest()[:24]


def build_params(model: str, images: list[bytes], effort: str = "low") -> dict:
    n = len(images)
    content = []
    for k, data in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {k}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(data).decode()}})
    content.append({"type": "text", "text": (
        f"This shot has {n} frame{'s' if n > 1 else ''}. Check every frame, choose the best frame and "
        "score the shot, following the rules.")})
    return {"model": MODELS[model], "max_tokens": MAX_TOKENS, "system": prompt_text(),
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}


def dollars(model: str, input_tokens: int, output_tokens: int, batch: bool) -> float:
    pin, pout = PRICE[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6 * (0.5 if batch else 1.0)


# --- Jobs ----------------------------------------------------------------------------

@dataclass
class Job:
    model: str
    cache: Path
    names: dict[str, list[str]]                         # request id -> frame file names
    todo: dict[str, dict] = field(default_factory=dict)  # request id -> params, not answered yet
    done: dict[str, dict] = field(default_factory=dict)  # request id -> answer (from earlier runs)
    input_tokens: int = 0                               # estimate for `todo`
    output_tokens: int = 0

    def estimate(self, batch: bool = True) -> float:
        return dollars(self.model, self.input_tokens, self.output_tokens, batch)


def read_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def prepare(shots: dict[str, list[Path]], model: str, cache_dir: Path, effort: str = "low") -> Job:
    """Requests for every shot not answered before by this model with these rules.
    shots: any key -> the frames' files, in time order (at most MAX_FRAMES are shown)."""
    if model not in MODELS:
        raise JudgeError(f"Unknown model {model!r}: use {', '.join(MODELS)}")
    cache = cache_dir / f"{model}-{prompt_version()}.jsonl"
    old = read_cache(cache)
    job = Job(model, cache, {})
    system_tokens = len(prompt_text()) // 4 + 40
    for files in shots.values():
        files = list(files)[:MAX_FRAMES]
        names = [Path(f).name for f in files]
        rid = request_id(names)
        job.names[rid] = names
        if rid in old:
            job.done[rid] = old[rid]["answer"]
            continue
        images = [small_jpeg(Path(f)) for f in files]
        job.todo[rid] = build_params(model, [b for b, _, _ in images], effort)
        job.input_tokens += system_tokens + 30 * len(files) + sum(image_tokens(w, h) for _, w, h in images)
        job.output_tokens += OUTPUT_GUESS + OUTPUT_PER_FRAME * len(files)
    return job


def spent_so_far(job: Job) -> float:
    """Money already paid for this job's shots (this run and earlier runs), from the saved answers."""
    return sum(r.get("dollars", 0.0) for rid, r in read_cache(job.cache).items() if rid in job.names)


def scene_version() -> str:
    text = SCENE_PROMPT_FILE.read_text(encoding="utf-8") + json.dumps(SCENE_SCHEMA, sort_keys=True) + str(SCENE_PX)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def build_scene_params(model: str, images: list[bytes], keep: int, effort: str = "medium") -> dict:
    n = len(images)
    content = []
    for k, data in enumerate(images, 1):
        content.append({"type": "text", "text": f"Shot {k}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(data).decode()}})
    content.append({"type": "text", "text": (
        f"This sequence has {n} shots. From a sequence like this the studio usually delivers about {keep}. "
        f"Score every shot, Shot 1 to Shot {n}.")})
    return {"model": MODELS[model], "max_tokens": SCENE_MAX_TOKENS,
            "system": SCENE_PROMPT_FILE.read_text(encoding="utf-8"),
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCENE_SCHEMA}}}


def scene_estimate(model: str, shots: int, scenes: int, batch: bool = True) -> float:
    """Price of round 2 before round 1 has answered (which frames are sent is not known yet)."""
    system = len(SCENE_PROMPT_FILE.read_text(encoding="utf-8")) // 4 + 40
    tin = scenes * (system + 60) + shots * (10 + image_tokens(SCENE_PX, SCENE_PX * 2 // 3))
    tout = scenes * SCENE_OUTPUT[0] + shots * SCENE_OUTPUT[1]
    return dollars(model, tin, tout, batch)


def prepare_scenes(scenes: dict, model: str, cache_dir: Path, effort: str = "medium") -> Job:
    """Round 2 requests: scenes = any key -> (best frames of consecutive shots, about how many the
    studio keeps). Answers are kept like round 1's (data/judge/<shoot>/<model>-scene-<version>.jsonl)."""
    cache = cache_dir / f"{model}-scene-{scene_version()}.jsonl"
    old = read_cache(cache)
    job = Job(model, cache, {})
    system = len(SCENE_PROMPT_FILE.read_text(encoding="utf-8")) // 4 + 40
    for files, keep in scenes.values():
        names = [Path(f).name for f in files]
        rid = request_id(names + [f"keep {keep}"])
        job.names[rid] = names
        if rid in old:
            job.done[rid] = old[rid]["answer"]
            continue
        images = [small_jpeg(Path(f), SCENE_PX) for f in files]
        job.todo[rid] = build_scene_params(model, [b for b, _, _ in images], keep, effort)
        job.input_tokens += system + 60 + sum(10 + image_tokens(w, h) for _, w, h in images)
        job.output_tokens += SCENE_OUTPUT[0] + SCENE_OUTPUT[1] * len(files)
    return job


def scene_rid(files: list, keep: int) -> str:
    return request_id([Path(f).name for f in files] + [f"keep {keep}"])


def answer_of(message: dict) -> dict | None:
    """The JSON answer of one reply, or None (refused, cut off, or not JSON)."""
    if message.get("stop_reason") in ("refusal", "max_tokens"):
        return None
    for block in message.get("content", []):
        if block.get("type") == "text":
            try:
                ans = json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                return None
            if isinstance(ans, dict) and (("best_frame" in ans and "deliver_score" in ans)
                                          or isinstance(ans.get("shots"), list)):
                return ans
    return None


# --- Sending -------------------------------------------------------------------------

def http(method: str, url: str, key: str, body: dict | None = None, raw: bool = False):
    """One call to the API (replaced in the tests)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                payload = r.read()
            return payload if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            if e.code == 401:
                raise JudgeError("The API key was refused (401). Check data\\anthropic_key.txt.") from None
            if e.code == 400 and "credit" in detail.lower():
                raise JudgeError("Not enough API credit. Add credit on platform.claude.com (Settings > Billing).") from None
            raise JudgeError(f"API error {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            if attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            raise JudgeError(f"Cannot reach the API (internet?): {e.reason}") from None


def _save(job: Job, rid: str, message: dict, batch: bool) -> tuple[dict | None, float]:
    usage = message.get("usage", {})
    tin = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) \
        + usage.get("cache_read_input_tokens", 0)
    cost = dollars(job.model, tin, usage.get("output_tokens", 0), batch)
    ans = answer_of(message)
    if ans is not None:
        job.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(job.cache, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rid, "frames": job.names[rid], "answer": ans, "usage": usage,
                                "dollars": round(cost, 5)}) + "\n")
        job.done[rid] = ans
    return ans, cost


def chunks(todo: dict[str, dict], mb: float = BATCH_MB) -> list[dict[str, dict]]:
    out, cur, size = [], {}, 0
    for rid, params in todo.items():
        s = len(json.dumps(params))
        if cur and size + s > mb * 1e6:
            out.append(cur)
            cur, size = {}, 0
        cur[rid] = params
        size += s
    if cur:
        out.append(cur)
    return out


def submit(job: Job, key: str, log=print) -> list[str]:
    """Send the job's open requests as message batches; returns the batch ids (also saved, so an
    interrupted run picks them up again instead of paying twice)."""
    state = job.cache.with_suffix(".batches")
    if state.exists():
        ids = json.loads(state.read_text(encoding="utf-8"))
        if ids:
            log(f"  {NAMES[job.model]}: continuing the batch sent earlier")
            return ids
    ids = []
    for part in chunks(job.todo):
        b = http("POST", f"{API}/messages/batches", key,
                 {"requests": [{"custom_id": rid, "params": p} for rid, p in part.items()]})
        ids.append(b["id"])
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(ids), encoding="utf-8")
    log(f"  {NAMES[job.model]}: sent {len(job.todo)} request{'s' if len(job.todo) > 1 else ''} "
        f"({len(ids)} batch{'es' if len(ids) > 1 else ''})")
    return ids


def collect(jobs: list[Job], batch_ids: dict[str, list[str]], key: str, log=print, poll: float = POLL_S) -> dict[str, float]:
    """Waits for the batches, saves every answer; returns the money spent per model."""
    spent = {j.model: 0.0 for j in jobs}
    open_ids = {(j.model, b) for j in jobs for b in batch_ids[j.model]}
    by_model = {j.model: j for j in jobs}
    start = time.time()
    while open_ids:
        line = []
        for model, bid in sorted(open_ids):
            b = http("GET", f"{API}/messages/batches/{bid}", key)
            c = b.get("request_counts", {})
            line.append(f"{NAMES[model]} {c.get('succeeded', 0) + c.get('errored', 0)}/"
                        f"{sum(c.values()) if c else '?'}")
            if b.get("processing_status") == "ended":
                job = by_model[model]
                text = http("GET", b["results_url"], key, raw=True).decode("utf-8")
                for row in text.splitlines():
                    if not row.strip():
                        continue
                    r = json.loads(row)
                    if r["custom_id"] in job.names and r["result"]["type"] == "succeeded":
                        spent[model] += _save(job, r["custom_id"], r["result"]["message"], batch=True)[1]
                open_ids.discard((model, bid))
        if open_ids:
            log(f"  waiting for Claude ({time.time() - start:.0f} s): " + ", ".join(line))
            time.sleep(poll)
    for j in jobs:
        j.cache.with_suffix(".batches").unlink(missing_ok=True)
    return spent


def send_now(job: Job, key: str, workers: int = 4, log=print) -> float:
    """Full price, no waiting: requests one by one (a few at a time)."""
    def one(item):
        rid, params = item
        return rid, http("POST", f"{API}/messages", key, params)
    spent = 0.0
    with ThreadPoolExecutor(workers) as pool:
        for k, (rid, message) in enumerate(pool.map(one, job.todo.items()), 1):
            spent += _save(job, rid, message, batch=False)[1]
            if k % 20 == 0:
                log(f"  {NAMES[job.model]}: {k}/{len(job.todo)}")
    return spent


# --- Using the answers -----------------------------------------------------------------

def frame_flags(ans: dict, k: int) -> dict:
    """Claude's check of frame k (1-based) of a shot, or {} if it did not answer for that frame."""
    for f in ans.get("frames", []):
        if f.get("frame") == k:
            return f
    return {}


def breaks_rules(flags: dict) -> list[str]:
    out = []
    eyes = str(flags.get("eyes", "")).lower()
    if eyes in ("some half closed", "some closed"):
        out.append("eyes " + eyes.replace("some ", ""))
    if flags.get("head_turned"):
        out.append("head turned")
    if flags.get("main_face_sharp") is False:
        out.append("not sharp")
    return out


def best_frame(ans: dict, n_frames: int) -> int:
    """Claude's best frame (0-based); if Claude's pick breaks a rule and another frame does not,
    the first clean frame Claude rated is used instead."""
    k = int(ans.get("best_frame", 1))
    k = min(max(k, 1), n_frames)
    if breaks_rules(frame_flags(ans, k)):
        clean = [f["frame"] for f in ans.get("frames", [])
                 if 1 <= f.get("frame", 0) <= n_frames and not breaks_rules(f)]
        if clean:
            k = clean[0]
    return k - 1
