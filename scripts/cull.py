r"""Phase 4: KeepSake's own Selected / Removed for a shoot, with a reason for every removal.

    ks cull "E:\Sharma Wedding\Raw"              like the studio: its usual share, ONE best photo per shot
    ks cull "E:\Sharma Wedding\Raw" --keep 15    the same, but keep 15% of the photos
    ks cull "E:\Sharma Wedding\Raw" --safe       a big safe shortlist: 95% of what the studio would pick
    ks cull "E:\Sharma Wedding\Raw" --target 90  safe shortlist catching 90% (smaller)

The shoot must be prepared first (ks prepare). Nothing is copied or moved yet: open the review
page, change anything you disagree with, then run ks apply. See engine/cull.py for the method.
"""
from __future__ import annotations

import html
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.catalog import Catalog, now_text  # noqa: E402
from engine.cli import clean_path  # noqa: E402
from engine.config import CACHE_DIR  # noqa: E402
from engine.cull import TARGET, UNSURE_BAND, cull_shoot, labelled_shoots  # noqa: E402
from engine.ingest import cache_path, duration, shoot_for  # noqa: E402
from engine.labels import attach_labels  # noqa: E402


def option(name: str) -> float | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return float(sys.argv[i + 1].strip("%")) / 100
    return None


def review_page(shoot, cat: Catalog) -> Path:
    """One page with every photo, Selected first; click a photo to move it; save the changes."""
    rows = list(cat.db.execute(
        "SELECT p.id, p.path, p.folder, c.decision, c.unsure, COALESCE(c.score, -1), COALESCE(c.reason, '') "
        "FROM cull c JOIN photos p ON p.id = c.id ORDER BY p.folder, p.taken_s, p.id"))
    items = []
    for pid, path, folder, decision, unsure, score, reason in rows:
        thumb = cache_path(shoot.cache, "thumbs", pid)
        items.append({"p": path, "e": (folder or "").split("/")[0], "d": decision[0], "u": unsure,
                      "s": round(score), "r": reason,
                      "t": "../" + thumb.relative_to(shoot.cache).as_posix() if thumb.exists() else ""})
    page = shoot.cache / "review" / "cull_review.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page.write_text(f"""<!doctype html><meta charset="utf-8"><title>KeepSake cull review</title>
<style>
body{{font:14px system-ui;margin:0;background:#f4f4f2;color:#222}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:2}}
h1{{font-size:18px;margin:0 0 6px}} button{{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}}
button.on{{background:#222;color:#fff;border-color:#222}} #save{{background:#2f6fde;color:#fff;border-color:#2f6fde}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;padding:16px}}
.c{{background:#fff;border-radius:8px;overflow:hidden;cursor:pointer;border:3px solid transparent}}
.c.s{{border-color:#2e9e5b}} .c.r{{border-color:#d0d0d0;opacity:.75}} .c.u{{border-color:#e0a100}} .c.x{{outline:3px dashed #2f6fde}}
.c img{{width:100%;height:150px;object-fit:cover;display:block;background:#ddd}}
.c div{{padding:6px 8px;font-size:12px;line-height:1.35}} .c b{{font-size:12px}}
#more{{margin:0 16px 24px}}
</style>
<header><h1>{html.escape(shoot.name)} - KeepSake cull</h1>
<span id=tabs></span> &nbsp; <span id=info></span> &nbsp;
<button id=save onclick="save()">Save my changes</button>
<p style="margin:6px 0 0;color:#666">Green = Selected, yellow = AI unsure (kept), grey = Removed. Click a photo to move it.
Then press <b>Save my changes</b> and run:<br><code>ks apply "photo folder" "output folder" --changes "path of the saved file"</code></p></header>
<main id=grid></main><button id=more onclick="draw(false)">Show more</button>
<script>
const items = {data};
const changed = new Map();
let view = 'all', shown = 0;
const tabs = {{all:'All', s:'Selected', u:'AI unsure', r:'Removed', x:'My changes'}};
function state(it){{ const d = changed.get(it.p) || it.d; return d; }}
function match(it){{ const d = state(it);
  return view==='all' || (view==='s'&&d==='s') || (view==='r'&&d==='r') || (view==='u'&&it.u&&d==='s') || (view==='x'&&changed.has(it.p)); }}
function card(it){{ const d = state(it), el = document.createElement('div');
  el.className = 'c ' + (d==='s' ? (it.u&&!changed.has(it.p) ? 'u' : 's') : 'r') + (changed.has(it.p) ? ' x' : '');
  const name = it.p.split('/').pop();
  el.innerHTML = (it.t ? `<img loading=lazy src="${{it.t}}">` : '<img>') +
    `<div><b>${{d==='s' ? 'Selected' : 'Removed'}}</b> ${{it.s>=0 ? '· score '+it.s : ''}}<br>${{name}} · ${{it.e}}<br>${{changed.has(it.p) ? 'moved by you' : it.r}}</div>`;
  el.onclick = () => {{ const now = state(it) === 's' ? 'r' : 's';
    if (now === it.d) changed.delete(it.p); else changed.set(it.p, now);
    el.replaceWith(card(it)); counts(); }};
  return el; }}
function draw(reset){{ const g = document.getElementById('grid');
  if (reset){{ g.innerHTML=''; shown=0; }}
  const list = items.filter(match), next = list.slice(shown, shown + 400);
  next.forEach(it => g.appendChild(card(it))); shown += next.length;
  document.getElementById('more').style.display = shown < list.length ? '' : 'none'; }}
function counts(){{ let s=0, r=0; items.forEach(it => state(it)==='s' ? s++ : r++);
  document.getElementById('info').textContent = `Selected ${{s.toLocaleString()}} · Removed ${{r.toLocaleString()}} · your changes ${{changed.size}}`; }}
function setView(v){{ view = v; document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.v===v)); draw(true); }}
function save(){{ const lines = ['path,decision'];
  changed.forEach((d, p) => lines.push('"' + p.replaceAll('"','""') + '",' + (d==='s' ? 'selected' : 'removed')));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\\n') + '\\n'], {{type:'text/csv'}}));
  a.download = 'keepsake_changes.csv'; a.click(); }}
document.getElementById('tabs').innerHTML = Object.entries(tabs).map(([v,t]) => `<button data-v=${{v}} onclick="setView('${{v}}')">${{t}}</button>`).join(' ');
setView('all'); counts();
</script>""", encoding="utf-8")
    return page


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name in ("--target", "--keep"):
        if name in sys.argv:
            i = sys.argv.index(name)
            if i + 1 < len(sys.argv) and sys.argv[i + 1] in args:
                args.remove(sys.argv[i + 1])
    if not args:
        print('Give the shoot folder, for example:  ks cull "E:\\Sharma Wedding\\Raw"')
        return 1
    root = clean_path(args[0])
    shoot = shoot_for(root)
    if not shoot.db.exists():
        print(f'This shoot has not been prepared yet. Run first:  ks prepare "{root}"')
        return 1
    target = option("--target") or TARGET
    keep = option("--keep")
    style = "safe" if ("--safe" in sys.argv or "--target" in sys.argv) else "studio"
    start = time.perf_counter()
    others = labelled_shoots(CACHE_DIR, exclude=shoot.db)
    with Catalog(shoot.db) as cat:
        attach_labels(cat, root)
        if not cat.db.execute("SELECT COUNT(*) FROM moments").fetchone()[0]:
            print(f'Moments are missing. Run first:  ks prepare "{root}"')
            return 1
        print(f"Learning from {len(others)} other wedding(s) and deciding every photo...")
        try:
            result = cull_shoot(cat, shoot.name, others, target=target, keep_share=keep, style=style)
        except RuntimeError as e:
            print(e)
            return 1
        settings = {"when": now_text(), "style": result.style, "target": target, "keep_share": result.keep_share,
                    "unsure_band": UNSURE_BAND, "mode": result.mode, "model": result.model,
                    "trained_on": result.trained_on}
        cat.save_cull(result.rows, settings)
        page = review_page(shoot, cat)

    total = len(result.rows)
    d = Counter((r[1], r[2]) for r in result.rows)
    selected = d[("selected", 0)] + d[("selected", 1)]
    removed = d[("removed", 0)]
    if result.mode == "other weddings":
        learned = (f"{len(result.trained_on)} past wedding(s), {result.train_photos:,} photos, "
                   f"{result.train_picks:,} studio picks")
    else:
        learned = ("this wedding's own studio choices: each event decided by a model trained only on "
                   f"the other events ({result.train_picks:,} picks)")
    if result.style == "studio":
        how = [f"  Model               {result.model}; chooses like the studio, one photo per shot",
               f"  Shots               {result.shots:,} (near-identical frames grouped)",
               f"  How many            {result.keep_share:.1%} of photos (this studio's usual share)",
               "  Best frame          the frame the studio would keep; not closed / half-closed eyes or turned heads"]
    else:
        how = [f"  Model               {result.model}; aim: keep {target:.0%} of what the studio would pick",
               f"  Shortlist size      {result.keep_share:.0%} of photos, plus {UNSURE_BAND:.0%} 'AI unsure' kept for you to check"]
    lines = [f"KeepSake cull - {shoot.name} - {now_text()}", "",
             f"  Learned from        {learned}", *how, "",
             f"  Selected            {selected:,} of {total:,} ({selected / max(total, 1):.0%}); "
             f"{d[('selected', 1)]:,} of them AI unsure",
             f"  Removed             {removed:,} ({removed / max(total, 1):.0%})"]
    for kind, n in result.reasons.most_common():
        lines.append(f"    {kind:<44}{n:>7,}")
    if result.unread:
        lines.append(f"  Could not be read   {result.unread:,} (kept in Selected for you to check)")
    if result.caught:
        got, picks = result.caught
        lines += ["", f"  Check against the studio's own choices (not used to decide this shoot):",
                  f"    studio picks in Selected  {got:,} of {picks:,} ({got / max(picks, 1):.0%})",
                  f"    studio picks removed      {picks - got:,}"]
    lines += ["", f"  Time                {duration(time.perf_counter() - start)}"]
    text = "\n".join(lines)
    print("\n" + text)
    (shoot.cache / "cull_report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nReview page (open it in your browser): {page}")
    print(f'Next:  ks apply "{root}" "C:\\KeepSake Output\\{shoot.name}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
