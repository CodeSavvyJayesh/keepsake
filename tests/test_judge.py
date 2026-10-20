"""Tests for Claude as a second editor, with a fake API (nothing is sent, nothing is paid)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine import judge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class FakeAPI:
    """Answers like Claude would: frame 1 of every shot is a blink, the last frame is the best."""

    def __init__(self):
        self.sent = []          # request ids sent

    def reply(self, params):
        n = sum(1 for b in params["messages"][0]["content"] if b["type"] == "image")
        if "shots" in params["output_config"]["format"]["schema"]["properties"]:     # round 2: later shots win
            ans = {"shots": [{"shot": k, "keep": 10 + 3 * k, "why": "test"} for k in range(1, n + 1)]}
            return {"type": "message", "content": [{"type": "text", "text": json.dumps(ans)}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 500 * n, "output_tokens": 30 * n}}
        frames = [{"frame": k, "eyes": "some closed" if k == 1 and n > 1 else "all open", "head_turned": False,
                   "main_face_sharp": True, "problems": ""} for k in range(1, n + 1)]
        ans = {"frames": frames, "best_frame": 1, "deliver_score": 3 + n, "reason": "test"}
        return {"type": "message", "content": [{"type": "thinking", "thinking": "..."},
                                               {"type": "text", "text": json.dumps(ans)}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1000 * n, "output_tokens": 200}}

    def __call__(self, method, url, key, body=None, raw=False):
        assert key.startswith("sk-ant-")
        if url.endswith("/messages/batches") and method == "POST":
            self.batch = body["requests"]
            self.sent += [r["custom_id"] for r in body["requests"]]
            return {"id": "msgbatch_1"}
        if url.endswith("/messages/batches/msgbatch_1"):
            return {"processing_status": "ended", "request_counts": {"succeeded": len(self.batch)},
                    "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"}
        if url.endswith("/results"):
            return "\n".join(json.dumps({"custom_id": r["custom_id"], "result": {
                "type": "succeeded", "message": self.reply(r["params"])}}) for r in self.batch).encode()
        if url.endswith("/messages") and method == "POST":
            self.sent.append("now")
            return self.reply(body)
        raise AssertionError(url)


def photos(folder: Path, n: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = folder / f"{i + 1:04d}_IMG_{i}.JPG"
        Image.new("RGB", (2048, 1365), (i * 20 % 255, 90, 40)).save(p)
        out.append(p)
    return out


def test_request_has_every_frame_and_the_answer_form(tmp_path):
    files = photos(tmp_path, 3)
    data = [judge.small_jpeg(f) for f in files]
    assert data[0][1:] == (1024, 683) and judge.image_tokens(1024, 683) == 925
    p = judge.build_params("opus", [d[0] for d in data])
    assert p["model"] == "claude-opus-5-5" and "blink" in p["system"]
    assert sum(b["type"] == "image" for b in p["messages"][0]["content"]) == 3
    fmt = p["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert p["output_config"]["effort"] == "low"


def test_key_is_read_from_the_laptop_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(judge, "KEY_FILE", tmp_path / "anthropic_key.txt")
    with pytest.raises(judge.JudgeError, match="No API key"):
        judge.load_key()
    (tmp_path / "anthropic_key.txt").write_text("﻿sk-ant-api03-abc\n", encoding="utf-8")
    assert judge.load_key() == "sk-ant-api03-abc"
    (tmp_path / "anthropic_key.txt").write_text("my key is sk-ant-x", encoding="utf-8")
    with pytest.raises(judge.JudgeError, match="does not look right"):
        judge.load_key()


def test_priced_first_and_never_paid_twice(tmp_path, monkeypatch):
    files = photos(tmp_path / "test", 5)
    shots = {1: files[:3], 2: files[3:4], 3: files[4:]}
    fake = FakeAPI()
    monkeypatch.setattr(judge, "http", fake)
    job = judge.prepare(shots, "sonnet", tmp_path / "judge")
    assert len(job.todo) == 3 and not fake.sent
    assert 0.002 < job.estimate() < 0.05                       # 5 images on Sonnet, batch price
    ids = judge.submit(job, "sk-ant-test", log=lambda *a: None)
    spent = judge.collect([job], {"sonnet": ids}, "sk-ant-test", log=lambda *a: None, poll=0)
    assert len(job.done) == 3 and spent["sonnet"] == pytest.approx((5000 * 2 + 600 * 10) / 1e6 / 2)
    again = judge.prepare(shots, "sonnet", tmp_path / "judge")
    assert not again.todo and len(again.done) == 3 and len(fake.sent) == 3
    now = judge.prepare(shots, "haiku", tmp_path / "judge")      # another model: its own answers
    judge.send_now(now, "sk-ant-test", log=lambda *a: None)
    assert len(now.done) == 3 and fake.sent.count("now") == 3


def test_a_blink_is_never_the_chosen_frame():
    ans = {"best_frame": 1, "frames": [
        {"frame": 1, "eyes": "some half closed", "head_turned": False, "main_face_sharp": True},
        {"frame": 2, "eyes": "all open", "head_turned": True, "main_face_sharp": True},
        {"frame": 3, "eyes": "looking down", "head_turned": False, "main_face_sharp": True}]}
    assert judge.breaks_rules(ans["frames"][0]) == ["eyes half closed"]
    assert judge.breaks_rules(ans["frames"][2]) == []            # looking down on purpose is fine
    assert judge.best_frame(ans, 3) == 2
    assert judge.best_frame({"best_frame": 9, "frames": []}, 3) == 2      # out of range: clamped
    assert judge.answer_of({"stop_reason": "refusal", "content": []}) is None


def test_claude_test_on_the_repick_photos(tmp_path, monkeypatch):
    import judge as script
    from engine.catalog import Catalog
    from engine.clip import to_blob
    from engine.cull import cull_shoot, labelled_shoots
    from engine.ingest import shoot_for
    from engine.learn import Logistic
    from test_cull import fake_wedding
    rng = np.random.default_rng(1)
    for name in ("past_one", "past_two"):
        (tmp_path / name).mkdir()
        fake_wedding(tmp_path / name / "catalog.db").close()
    root = tmp_path / "D" / "Rishi & Aashi"
    shoot = shoot_for(root, cache_dir=tmp_path / "cache")
    shoot.cache.mkdir(parents=True)
    cat = fake_wedding(shoot.db, n=60, labelled=False)
    for first in range(1, 61, 3):                                 # 3-frame bursts: one shot each
        base = rng.normal(size=512)
        for pid in range(first, first + 3):
            cat.db.execute("UPDATE clip SET emb = ? WHERE id = ?", (to_blob(base + rng.normal(0, 0.05, 512)), pid))
    cat.db.commit()
    r = cull_shoot(cat, shoot.name, labelled_shoots(tmp_path), make_model=Logistic)
    cat.save_cull(r.rows, {"mode": r.mode, "trained_on": r.trained_on, "style": r.style})
    cat.close()
    folder = tmp_path / "Re-pick test"
    files = photos(folder, 60)
    repick = tmp_path / "repick"
    repick.mkdir()
    with open(repick / f"{shoot.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["copy", "original", "photo_id", "studio_selected"])
        for i, p in enumerate(files, 1):
            w.writerow([p.name, f"IMG_{i}.JPG", i, int(i % 6 == 0)])
    (repick / f"{shoot.name}.json").write_text(json.dumps({"folder": str(folder), "photos": 60}))
    (repick / f"{shoot.name}_new.txt").write_text("\n".join(p.name for p in files[5::6][:7]))
    fake = FakeAPI()
    monkeypatch.setattr(judge, "http", fake)
    monkeypatch.setattr(judge, "JUDGE_DIR", tmp_path / "judge")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(script, "shoot_for", lambda r: shoot)
    monkeypatch.setattr(script, "REPICK_DIR", repick)
    monkeypatch.setattr(script, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["judge.py", str(root), "--test", "--model", "haiku", "opus", "--yes"])
    assert script.main() == 0
    assert len(fake.sent) == 42                     # 20 shots + 1 sequence, x 2 models, once each
    out = tmp_path / "out" / "Claude test - Rishi & Aashi"
    text = (out / "Summary.txt").read_text(encoding="utf-8")
    assert "KeepSake alone" in text and "Claude, whole sequence" in text and "20 of 20 shots answered" in text
    assert "20 scored in round 2" in text
    chosen = list(next((out / "Opus 5.5").glob("Claude selected (*)")).iterdir())
    rest = list(next((out / "Opus 5.5").glob("Claude not selected (*)")).iterdir())
    assert len(chosen) + len(rest) == 60 and not {p.name for p in chosen} & {p.name for p in rest}
    ks = len(list(next(out.glob("KeepSake selected (*)")).iterdir()))
    assert ks == len(chosen) and ks + len(list(next(out.glob("KeepSake not selected (*)")).iterdir())) == 60
    assert chosen and not any(p.name[:4] in {f"{k:04d}" for k in range(1, 61, 3)} for p in chosen)  # no blinks
    assert len({(int(p.name[:4]) - 1) // 3 for p in chosen}) == len(chosen)                        # one per shot
    notes = list(csv.DictReader(open(out / "Opus 5.5" / "Claude notes.csv", encoding="utf-8")))
    assert len(notes) == 60
    assert script.main() == 0 and len(fake.sent) == 42            # second run: nothing paid again


def test_claude_test_on_any_past_wedding_without_the_drive(tmp_path, monkeypatch):
    import judge as script
    from engine import ingest
    from test_cull import fake_wedding
    cache = tmp_path / "cache"
    for name in ("past_one", "past_two", "all_raw_images_shevaun_brian_jaisalmer"):
        (cache / name).mkdir(parents=True)
        cat = fake_wedding(cache / name / "catalog.db", n=90)
        cat.db.execute("INSERT INTO clocks (body, offset, support) VALUES ('A', 0, 9)")
        cat.db.commit()
        if name.startswith("all_raw"):
            for (pid,) in cat.db.execute("SELECT id FROM photos"):
                p = ingest.cache_path(cache / name, "analysis", pid)
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (1620, 1080), (pid, 80, 80)).save(p)
        cat.close()
    fake = FakeAPI()
    monkeypatch.setattr(judge, "http", fake)
    monkeypatch.setattr(judge, "JUDGE_DIR", tmp_path / "judge")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(script, "CACHE_DIR", cache)
    monkeypatch.setattr(script, "REPICK_DIR", tmp_path / "repick")
    monkeypatch.setattr(script, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["judge.py", "Shevaun", "--test", "--photos", "30", "--model", "haiku", "--yes"])
    assert script.main() == 0
    sent = len(fake.sent)
    assert 0 < sent <= 31                                          # shots + 1 sequence
    key = list(csv.DictReader(open(tmp_path / "judge" / "all_raw_images_shevaun_brian_jaisalmer" / "test_photos.csv",
                                   encoding="utf-8")))
    assert len(key) == 30 and {r["keepsake"] for r in key} <= {"selected", "removed"}
    out = tmp_path / "out" / "Claude test - Shevaun"
    assert len(list(next(out.glob("Test photos (*)")).iterdir())) == 30
    studio = next(out.glob("Studio selected (*)"))
    assert len(list(studio.iterdir())) == sum(r["studio_selected"] == "1" for r in key)
    assert len(list(next(out.glob("Studio not selected (*)")).iterdir())) == 30 - len(list(studio.iterdir()))
    summary = (out / "Summary.txt").read_text(encoding="utf-8")
    assert "KeepSake alone" in summary and "30 of 30 shots answered, 30 scored in round 2" in summary
    assert script.main() == 0 and len(fake.sent) == sent            # same photos, nothing paid again
