r"""Phase 2 exit test: a blind check of the eye reader by a person.

    ks eyecheck "E:\Sharma Wedding\Raw"

Makes review\eye_check.html in the shoot's cache folder: 50 main faces, half read by the
AI as eyes closed and half as open, shuffled, with the AI's verdict hidden. You mark each
one Open / Closed / Can't tell; the page then shows how often the AI was right and what share
of all closed eyes it catches. Answers are kept in the browser if the page is reloaded, and
can be downloaded as a file.
"""
from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.catalog import Catalog  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.faces import square_crop  # noqa: E402
from engine.ingest import cache_path, shoot_for  # noqa: E402

PER_VERDICT = 25
SEED = 7                      # the same 50 faces every time the page is made


def sample(cat: Catalog, verdict: str, n: int) -> tuple[list[tuple], int]:
    rows = list(cat.db.execute(
        "SELECT f.photo_id, f.n, f.x, f.y, f.w, f.h, f.eyes FROM faces f "
        "WHERE f.subject = 1 AND f.eyes = ? ORDER BY f.photo_id, f.n", (verdict,)))
    return random.Random(f"{SEED}-{verdict}").sample(rows, min(n, len(rows))), len(rows)


def write_images(shoot, faces: list[tuple], folder: Path) -> None:
    """For each face: a 240 px cut-out and the whole photo (480 px) with the face boxed."""
    folder.mkdir(parents=True, exist_ok=True)
    for pid, k, x, y, w, h, _ in faces:
        bgr = cv2.imdecode(np.fromfile(str(cache_path(shoot.cache, "analysis", pid)), np.uint8),
                           cv2.IMREAD_COLOR)
        cut, _ = square_crop(bgr, (x, y, w, h), margin=1.6, size=240)
        cv2.imencode(".jpg", cut)[1].tofile(str(folder / f"{pid:06d}_{k}_face.jpg"))
        scene = bgr.copy()
        t = max(2, int(max(scene.shape[:2]) / 300))
        cv2.rectangle(scene, (int(x), int(y)), (int(x + w), int(y + h)), (0, 220, 255), t)
        scale = 480 / max(scene.shape[:2])
        scene = cv2.resize(scene, (round(scene.shape[1] * scale), round(scene.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(
            str(folder / f"{pid:06d}_{k}_photo.jpg"))


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KeepSake eye check</title>
<style>
:root {{ --ink:#1d1d1f; --quiet:#6e6e73; --line:#d9d9de; --bg:#f6f6f7; --card:#fff; --accent:#2f6fde; }}
body {{ margin:0; font:15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink); background:var(--bg); }}
header {{ position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line); padding:12px 24px; z-index:2;
          display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
h1 {{ font-size:18px; margin:0; }}
#count {{ color:var(--quiet); }}
button {{ font:inherit; border:1px solid var(--line); background:var(--card); border-radius:8px; padding:6px 12px; cursor:pointer; }}
button.on {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
main {{ padding:16px 24px 48px; max-width:1100px; margin:auto; }}
.intro {{ color:var(--quiet); max-width:760px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; margin:12px 0;
         display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
.card.done {{ opacity:.55; }}
.card img.face {{ width:240px; height:240px; border-radius:8px; }}
.card img.photo {{ max-width:min(480px, 100%); border-radius:8px; }}
.num {{ font-weight:600; width:28px; }}
.choices {{ display:flex; flex-direction:column; gap:8px; }}
#result {{ background:var(--card); border:2px solid var(--accent); border-radius:12px; padding:16px; margin:16px 0; display:none; }}
#result pre {{ white-space:pre-wrap; background:var(--bg); padding:12px; border-radius:8px; }}
</style>
<header><h1>Eye check: {shoot}</h1><span id="count"></span>
<button id="show">Show result</button><button id="save">Download answers</button></header>
<main>
<p class="intro">Look at each face (the yellow box shows who it is in the photo) and say whether
<b>their eyes are open or closed</b>. Choose "Can't tell" if it is too small, blurred or hidden.
The AI's own answer is hidden until you press "Show result". Takes about 15 minutes.</p>
<div id="result"></div>
{cards}
</main>
<script>
const FACES = {faces};
const TOTALS = {totals};
const KEY = "keepsake-eyecheck-{key}";
let answers = {{}};
try {{ answers = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ answers = {{}}; }}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(answers)); }} catch (e) {{}} }}
function paint() {{
  document.querySelectorAll(".card").forEach(card => {{
    const a = answers[card.dataset.id];
    card.classList.toggle("done", !!a);
    card.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === a));
  }});
  document.getElementById("count").textContent = Object.keys(answers).length + " of " + FACES.length + " answered";
}}
document.querySelectorAll(".card button").forEach(b => b.addEventListener("click", () => {{
  answers[b.closest(".card").dataset.id] = b.dataset.v; save(); paint();
}}));
function pct(a, b) {{ return b ? Math.round(100 * a / b) + "%" : "-"; }}
document.getElementById("show").addEventListener("click", () => {{
  let c = {{closed: 0, closedRight: 0, open: 0, openRight: 0, missed: 0, cant: 0}};
  for (const f of FACES) {{
    const a = answers[f.id];
    if (!a) continue;
    if (a === "cant") {{ c.cant++; continue; }}
    if (f.ai === "closed") {{ c.closed++; if (a === "closed") c.closedRight++; }}
    else {{ c.open++; if (a === "open") c.openRight++; else c.missed++; }}
  }}
  // share of all closed eyes the AI catches, scaling the two samples to the whole shoot
  const caught = c.closed ? TOTALS.closed * c.closedRight / c.closed : 0;
  const missed = c.open ? TOTALS.open * c.missed / c.open : 0;
  const catchRate = caught + missed ? caught / (caught + missed) : 0;
  const answered = Object.keys(answers).length;
  const text = `KeepSake eye check ({shoot}), ${{answered}} of ${{FACES.length}} answered, ${{c.cant}} can't tell
AI said CLOSED: you agreed ${{c.closedRight}} of ${{c.closed}} (${{pct(c.closedRight, c.closed)}})
AI said OPEN:   you agreed ${{c.openRight}} of ${{c.open}} (${{pct(c.openRight, c.open)}})
Overall agreement: ${{pct(c.closedRight + c.openRight, c.closed + c.open)}}
Estimated share of all closed eyes (main people, faces it could judge) the AI catches: ${{Math.round(100 * catchRate)}}%`;
  const box = document.getElementById("result");
  box.style.display = "block";
  box.innerHTML = "<b>Result</b> (copy this and send it):<pre></pre>";
  box.querySelector("pre").textContent = text;
  box.scrollIntoView({{behavior: "smooth"}});
}});
document.getElementById("save").addEventListener("click", () => {{
  const blob = new Blob([JSON.stringify({{shoot: "{shoot}", faces: FACES, answers}}, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "eye_check_answers.json"; a.click();
}});
paint();
</script></html>
"""


def main() -> int:
    if len(sys.argv) < 2:
        print('Give the shoot folder, for example:  ks eyecheck "E:\\Sharma Wedding\\Raw"')
        return 1
    root = clean_path(sys.argv[1])
    shoot = shoot_for(root)
    if not shoot.db.exists():
        print(f"This shoot has not been ingested yet. Run first:  ks ingest \"{root}\"")
        return 1
    with Catalog(shoot.db) as cat:
        closed, n_closed = sample(cat, "closed", PER_VERDICT)
        opened, n_open = sample(cat, "open", PER_VERDICT)
    if not closed or not opened:
        print("No eye readings yet. Run first:  ks faces \"<shoot folder>\"")
        return 1
    faces = closed + opened
    random.Random(SEED).shuffle(faces)
    folder = shoot.cache / "review" / "eyecheck"
    write_images(shoot, faces, folder)
    cards, data = [], []
    for i, (pid, k, *_rest, verdict) in enumerate(faces, 1):
        fid = f"{pid}-{k}"
        data.append({"id": fid, "ai": verdict})
        cards.append(
            f'<div class="card" data-id="{fid}"><span class="num">{i}</span>'
            f'<img class="face" src="eyecheck/{pid:06d}_{k}_face.jpg" alt="face {i}">'
            f'<img class="photo" src="eyecheck/{pid:06d}_{k}_photo.jpg" alt="photo {i}">'
            '<div class="choices"><button data-v="open">Eyes open</button>'
            '<button data-v="closed">Eyes closed</button><button data-v="cant">Can\'t tell</button></div></div>')
    page = shoot.cache / "review" / "eye_check.html"
    page.write_text(PAGE.format(shoot=html.escape(shoot.name), key=shoot.name, cards="\n".join(cards),
                                faces=json.dumps(data),
                                totals=json.dumps({"closed": n_closed, "open": n_open})),
                    encoding="utf-8")
    print(f"Eye check ready: {len(faces)} faces ({len(closed)} the AI reads as closed, "
          f"{len(opened)} as open, shuffled).")
    print(f"Main faces in this shoot read as closed: {n_closed:,}; as open: {n_open:,}.")
    print(f"\nOpen this page in your browser, answer all {len(faces)}, then press 'Show result':")
    print(f"  {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
