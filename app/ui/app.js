import { Scene } from '/ui/scene.js';
import { Sound } from '/ui/sound.js';

const $ = s => document.querySelector(s);
const fmt = n => (n ?? 0).toLocaleString('en-IN');
const wait = ms => new Promise(r => setTimeout(r, ms));
const scene = new Scene($('#scene'));
const sound = new Sound();
function cue(kind, card) {                 // a sound for the scene (and a note of it when a demo is filmed)
  const opts = card ? { pan: card.pan, seed: card.i } : {};
  sound.play(kind, opts);
  if (window.__ksRecord) (window.__ksEvents = window.__ksEvents || []).push({ kind, at: performance.now() / 1000, ...opts });
}
const SPEAKER = on => `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/>${on ? '<path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>' : '<path d="M16 9.5l5 5M21 9.5l-5 5"/>'}</svg>`;
function showSound() { const b = $('#soundBtn'); b.innerHTML = SPEAKER(sound.on); b.title = sound.on ? 'Sound on (click to turn off)' : 'Sound off (click to turn on)'; }
$('#soundBtn').onclick = () => { sound.setOn(!sound.on); showSound(); };
showSound();

async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || 'KeepSake did not answer');
  return j;
}
function toast(text, ms = 6000, action) {
  const t = $('#toast'); t.textContent = text; t.hidden = false;
  if (action) {
    const b = document.createElement('button'); b.textContent = action.label;
    b.onclick = () => { t.hidden = true; action.fn(); }; t.appendChild(b);
  }
  clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.hidden = true), ms);
}
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === id));
  scene.setMode(id);
  $('#status').classList.toggle('busy', id === 'working');
  $('#statusText').textContent = id === 'working' ? 'Working · everything stays on this laptop' : 'Offline · everything stays on this laptop';
}
function ready() { document.body.classList.remove('intro'); document.body.classList.add('ready'); }

// ---------- opening: the mandap draws itself, photos are dealt into a ring, the camera drops inside ----------
const pick = { folder: '', output: '', keep: 'shortlist', after: 'copy' };
let checked = null;                     // the chosen folder's details (/api/check)
let home = {};

async function boot() {
  home = await api('/api/home').catch(() => ({ brand: {}, recent: [], sample: [] }));
  const b = home.brand || {};
  if (b.logo) { for (const img of [$('#brandLogo'), $('#introLogo')]) { img.src = b.logo; img.hidden = false; } $('#brandRule').hidden = false; }
  if (b.background) { $('#photoBg').style.backgroundImage = `url("${b.background}")`; document.body.classList.add('has-photo-bg'); }
  if (b.name) { $('#kicker').innerHTML = `${b.name}<span class="sep"></span>Wedding photo culling`; document.title = `KeepSake · ${b.name}`; }
  renderRecent(home.recent || []);
  $('#kit').hidden = (home.recent || []).length > 0;           // a new computer: no past weddings yet
  scene.setPhotos(home.sample || []);

  const st = await api('/api/state').catch(() => ({ status: 'idle' }));
  if (st.status === 'running' || st.status === 'done') {          // the window was reopened: no opening scene
    scene.start(); scene.open(); ready();
    if (st.status === 'running') return follow(st);
    finish(st, false); return;
  }
  // scene time, with a wall-clock safety net so the app always opens
  const either = (p, ms) => Promise.race([p, wait(ms)]);
  await wait(450);
  scene.onEvent = cue; cue('start');                  // the music (sound.js) follows the scene's moments
  await either(scene.start(), 9000);                  // the photos scatter up into a tiny tilted ring
  await either(scene.wait(0.7), 2500);                // a beat to see the ring
  scene.open(); document.body.classList.add('drop');  // the camera drops inside
  await either(scene.wait(1.5), 4000);
  scene.onEvent = null;
  ready(); show('welcome'); cue('end');
}

function renderRecent(list) {
  const box = $('#recent');
  if (!list.length) return;
  const clean = s => s.replace(/[\uE000-\uF8FF]/g, '').replace(/\s+/g, ' ').trim();
  list = [...list].sort((p, q) => (q.ready - p.ready));            // connected drives first
  const SHOW = 4;
  box.innerHTML = '<b>On this laptop</b>';
  list.forEach((w, k) => {
    const btn = document.createElement('button');
    btn.textContent = clean(w.name); btn.className = (w.ready ? '' : 'off') + (k >= SHOW ? ' more' : '');
    btn.title = w.ready ? `${w.folder} · ${fmt(w.photos)} photos` : `${w.folder} (drive not connected)`;
    btn.onclick = () => (w.ready ? useFolder(w.folder) : toast(`Connect the drive with “${clean(w.name)}” first.`));
    if (k) box.insertAdjacentHTML('beforeend', `<span class="dot${k >= SHOW ? ' more' : ''}">·</span>`);
    box.appendChild(btn);
  });
  if (list.length > SHOW) {
    const all = document.createElement('button');
    all.className = 'all'; all.textContent = `+${list.length - SHOW} more`;
    all.onclick = () => { box.classList.add('open'); all.remove(); };
    box.appendChild(all);
  }
  box.hidden = false;
}

// ---------- 1. choose ----------
async function useFolder(folder) {
  if (!folder) return;
  try {
    const c = await api('/api/check', { folder });
    if (!c.ok) return toast(c.error || 'Folder not found.');
    pick.folder = folder; pick.output = c.output;
    $('#wName').textContent = c.name; $('#wPath').textContent = folder; $('#wPath').title = folder;
    $('#outPath').textContent = c.output; $('#outPath').title = c.output;
    $('#copyHint').textContent = c.not_selected_gb ? `· about ${c.not_selected_gb} GB, ${c.not_selected_time}` : '· usually 9 in 10 photos, large and slow';
    checked = c; setupNote();
    $('#chooser').hidden = true; $('#setup').hidden = false;
  } catch (e) { toast(e.message); }
}
$('#chooseBtn').onclick = async () => {
  $('#chooseBtn').disabled = true;
  try { const r = await api('/api/choose', { kind: 'photos' }); if (r.path) await useFolder(r.path); } catch (e) { toast(e.message); }
  $('#chooseBtn').disabled = false;
};
$('#kitBtn').onclick = async () => {                            // the studio's past weddings and branding
  $('#kitBtn').disabled = true;
  try {
    const r = await api('/api/choose', { kind: 'kit' });
    if (r.path) {
      const k = await api('/api/import-kit', { folder: r.path });
      toast(`Brought over ${k.weddings} past wedding${k.weddings === 1 ? '' : 's'}`
        + (k.branding ? ' and the studio\u2019s branding' : '') + '. KeepSake now chooses like your studio.', 6000);
      setTimeout(() => location.reload(), 2200);
    }
  } catch (e) { toast(e.message, 7000); }
  $('#kitBtn').disabled = false;
};
$('#pasteForm').onsubmit = e => { e.preventDefault(); useFolder($('#pathInput').value.trim().replace(/^"|"$/g, '')); };
$('#changeBtn').onclick = () => { $('#setup').hidden = true; $('#chooser').hidden = false; };
$('#outBtn').onclick = async () => {
  const r = await api('/api/choose', { kind: 'output', start: pick.output }).catch(e => toast(e.message));
  if (r?.path) { pick.output = r.path; $('#outPath').textContent = r.path; }
};
const KEEP_NOTE = { '0.07': 'about 7 in 100', studio: 'about 1 in 10, like your studio', wide: '1.5x your studio: catches more of your picks', shortlist: 'Selected: your usual share · Maybe: the rest, together 9 in 10 of your picks' };
function setupNote() {
  const c = checked || {};
  $('#setupNote').textContent = (c.prepared ? `Read once before (${fmt(c.photos)} photos), so this run is quicker. ` : '')
    + (pick.after === 'view' ? 'Nothing is copied anywhere.' : `${c.free_gb} GB free where it is saved.`);
}
// After choosing: copy into folders, or just view (see and move the photos in KeepSake, copy later)
const AFTER_NOTE = { view: 'See the photos and move them right away · nothing is copied', copy: 'Selected and Maybe go into folders on this computer' };
function setAfter(v, remember = true) {
  pick.after = v === 'view' ? 'view' : 'copy';
  if (remember) try { localStorage.setItem('ks-after', pick.after); } catch (e) { /* the choice just is not remembered */ }
  $('#afterSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.after === pick.after));
  $('#afterNote').textContent = AFTER_NOTE[pick.after];
  $('#outRow').hidden = $('#copyNotRow').hidden = pick.after === 'view';
  if (checked) setupNote();
}
$('#afterSeg').onclick = e => { const b = e.target.closest('button'); if (b) setAfter(b.dataset.after); };
try { setAfter(localStorage.getItem('ks-after') || 'copy', false); } catch (e) { setAfter('copy', false); }
$('#keepSeg').onclick = e => {
  const b = e.target.closest('button'); if (!b) return;
  $('#keepSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  pick.keep = b.dataset.keep; $('#keepNote').textContent = KEEP_NOTE[pick.keep];
};
$('#startBtn').onclick = async () => {
  $('#startBtn').disabled = true;
  try { follow(await api('/api/start', { folder: pick.folder, output: pick.output, keep: pick.keep, copy_not_selected: $('#copyNot').checked, copy: pick.after !== 'view' })); }
  catch (e) { toast(e.message); }
  $('#startBtn').disabled = false;
};

// ---------- 2. working ----------
let polling = false, lastThumbs = 0, sorted = false, shownPct = 0;
const SHORT = { read: 'Reading', quality: 'Sharpness', faces: 'Faces', clip: 'Understanding', groups: 'Moments', choose: 'Choosing', copy: 'Copying' };
// ---------- "Do you know?": the studio's lines, one after another while it works ----------
const FACT_MS = 4200;
let factTimer = null, factIdx = 0;
function startFacts() {
  const list = ((home.brand || {}).facts || []).filter(f => typeof f === 'string' && f.trim());
  $('#facts').hidden = !list.length;
  if (!list.length || factTimer) return;
  $('#factDots').innerHTML = list.map(() => '<i></i>').join('');
  factIdx = 0; showFact(list, true);
  factTimer = setInterval(() => { factIdx = (factIdx + 1) % list.length; showFact(list); }, FACT_MS);
}
function showFact(list, first) {
  const el = $('#factText');
  const write = () => {
    el.textContent = '';
    list[factIdx].trim().split(/\s+/).forEach((w, k) => {          // the words come in one after another
      const span = document.createElement('span');
      span.textContent = w; span.style.animationDelay = `${k * 38}ms`;
      el.append(span, ' ');
    });
    el.classList.remove('out');
    $('#factDots').querySelectorAll('i').forEach((d, k) => d.classList.toggle('on', k === factIdx));
  };
  if (first) return write();
  el.classList.add('out');                                           // the last one drifts up and away
  setTimeout(write, 450);
}
function stopFacts() { clearInterval(factTimer); factTimer = null; }

function follow(st) {
  show('working');
  startFacts();
  $('#wkName').textContent = st.name;
  sorted = false; shownPct = 0; $('#steps').innerHTML = '';
  if (!polling) { polling = true; poll(st); }
}
function render(st) {
  const ol = $('#steps');
  if (!ol.children.length) st.stages.forEach(s => ol.insertAdjacentHTML('beforeend', `<li data-key="${s.key}">${SHORT[s.key] || s.label}</li>`));
  st.stages.forEach(s => { const li = ol.querySelector(`[data-key="${s.key}"]`); li.className = s.state; li.hidden = s.key === 'copy' && st.copy === false; });
  const target = Math.floor(st.progress * 100);
  if (target > shownPct) shownPct = Math.min(target, shownPct + Math.max(1, Math.ceil((target - shownPct) / 2)));
  $('#pct').textContent = shownPct;
  $('#ovBar').style.width = `${(st.progress * 100).toFixed(1)}%`;
  const active = st.stages.find(s => s.state === 'active');
  const counts = active && active.total ? ` · ${fmt(active.done)} of ${fmt(active.total)}` : '';
  $('#stageNow').textContent = active ? active.label + counts : '';
  $('#eta').textContent = st.eta ? `about ${st.eta} left` : 'working';
  $('#elapsed').textContent = st.elapsed_text || '0 s';
  scene.setProgress(st.progress);
  const ready = st.status === 'running' && !!(st.result && st.result.ready);
  document.body.classList.toggle('copying', ready);
  $('#readyRow').hidden = !ready;
  if (ready) {
    result = st.result;
    $('#readyText').innerHTML = `<b>${fmt(st.result.selected)}</b> Selected` + (st.result.maybe ? ` + <b>${fmt(st.result.maybe)}</b> Maybe` : '')
      + ' · copying them now';
  }
}
async function poll(st) {
  try {
    st = st || await api('/api/state');
    render(st);
    if (Date.now() - lastThumbs > 5000 && st.stage !== 'choose') {
      lastThumbs = Date.now();
      api('/api/thumbs?set=any&n=48').then(r => r.photos.length && scene.feed(r.photos)).catch(() => {});
    }
    if (!sorted && (st.stage === 'copy' || st.status === 'done')) {
      sorted = true;
      api('/api/thumbs?set=selected&n=60').then(r => scene.sort(r.photos)).catch(() => {});
    }
    if (st.status === 'running') { setTimeout(() => poll(), 600); return; }
    polling = false;
    stopFacts();
    if (st.status === 'done') return finish(st);
    toast(st.error || 'Stopped.', 9000);
    show('welcome');
  } catch (e) { setTimeout(() => poll(), 1500); }
}
$('#stopBtn').onclick = async () => {
  $('#stopBtn').disabled = true; $('#stopBtn').textContent = 'Stopping…';
  await api('/api/stop', {}).catch(() => {});
  setTimeout(() => { $('#stopBtn').disabled = false; $('#stopBtn').textContent = 'Stop'; }, 3000);
};

// ---------- 3. done ----------
function countUp(el, to, ms = 1800) {
  const t0 = performance.now();
  const step = t => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 4); el.textContent = fmt(Math.round(to * e)); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
let result = {};
function finish(st, fresh = true) {
  document.body.classList.remove('copying');
  if (fresh) cue('done');
  const r = result = st.result || {};
  show('done');
  api('/api/thumbs?set=selected&n=60').then(x => scene.sort(x.photos)).catch(() => {});
  $('#doneKicker').textContent = `Finished · ${st.name}`;
  countUp($('#doneSel'), r.selected || 0);
  const total = r.total || (r.selected || 0) + (r.not_selected || 0);
  $('#doneSub').textContent = (r.maybe ? `+ ${fmt(r.maybe)} more in Maybe · ` : '')
    + `From ${fmt(total)} photos · ${r.time || st.elapsed_text} · `
    + (r.copied === false ? 'nothing copied' : `saved in ${r.output || st.output}`);
  renderStats();
}
function renderStats() {
  const r = result;
  const viewOnly = r.copied === false;                       // just viewed: no folders to open
  $('#done').querySelectorAll('[data-open]').forEach(b => { if (viewOnly) b.hidden = true; });
  $('#viewBtn').className = viewOnly ? 'btn gold' : 'btn line';
  $('#doneNote').hidden = !viewOnly;
  $('#openNot').hidden = viewOnly || !(r.copied_not_selected !== false || r.moved);
  const pct = (a, b) => `${Math.round(100 * (a || 0) / Math.max(b || 0, 1))}%`;
  const stats = [['Selected', fmt(r.selected)]];
  if (r.maybe) stats.push(['Maybe', fmt(r.maybe)]);
  stats.push(['Not selected', fmt(r.not_selected)]);
  if (!r.maybe) stats.push(['Moments', r.shots ? fmt(r.shots) : '—']);
  if (r.studio_picks) {
    stats.push(['Studio’s picks in Selected', `${fmt(r.studio_in_selected)} / ${fmt(r.studio_picks)}`, pct(r.studio_in_selected, r.studio_picks)]);
    if (r.maybe || r.studio_near > r.studio_in_selected)
      stats.push([r.maybe ? 'With Maybe' : 'With near frames', `${fmt(r.studio_near)} / ${fmt(r.studio_picks)}`,
                  `${pct(r.studio_near, r.studio_picks)} · counting near-identical frames`]);
  }
  if (r.moved) stats.push(['Moved by you', fmt(r.moved)]);
  $('#stats').innerHTML = stats.map(([k, v, sub]) =>
    `<div><div class="k">${k}</div><div class="v">${v}</div>${sub ? `<div class="s">${sub}</div>` : ''}</div>`).join('');
  $('#gSeg').querySelector('[data-set=selected]').innerHTML = `Selected <i>${fmt(r.selected)}</i>`;
  $('#gSeg').querySelector('[data-set=maybe]').innerHTML = `Maybe <i>${fmt(r.maybe)}</i>`;
  $('#gSeg').querySelector('[data-set=maybe]').hidden = !r.maybe && gSet !== 'maybe';
  $('#gSeg').querySelector('[data-set=not_selected]').innerHTML = `Not selected <i>${fmt(r.not_selected)}</i>`;
  $('#openMaybe').hidden = viewOnly || !r.maybe;
  $('#done').querySelector('[data-open=selected]').hidden = viewOnly;
}
document.querySelectorAll('[data-open]').forEach(b => (b.onclick = () => api('/api/open', { which: b.dataset.open }).catch(e => toast(e.message))));
$('#againBtn').onclick = async () => {
  await api('/api/reset', {}).catch(() => {});
  $('#setup').hidden = true; $('#chooser').hidden = false;
  scene.setPhotos(home.sample || []); show('welcome');
};

// ---------- gallery and the big view (a person can move any photo across) ----------
let gList = [], gIdx = 0, gSet = 'selected';
const TIERS = ['selected', 'maybe', 'not_selected'];           // the three folders
const LABEL = { selected: 'Selected', maybe: 'Maybe', not_selected: 'Not selected' };
const setOf = p => p.tier || gSet;
const ARROW = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
const STACK = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><rect x="7" y="3" width="14" height="14" rx="2"/><path d="M17 21H5a2 2 0 0 1-2-2V7"/></svg>';
const CHECK = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

async function openGallery(set) {
  gSet = set;
  $('#gallery').hidden = false;
  $('#gSeg').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.set === set));
  $('#gTitle').textContent = LABEL[set];
  const grid = $('#gGrid'); grid.innerHTML = ''; $('#gCount').textContent = '';
  const r = await api(`/api/gallery?set=${set}`).catch(() => ({ photos: [] }));
  if (gSet !== set) return;
  gList = r.photos;
  renderGrid();
}
function renderGrid() {
  const grid = $('#gGrid'), frag = document.createDocumentFragment();
  grid.innerHTML = '';
  gList.forEach(p => frag.appendChild(tile(p)));
  grid.appendChild(frag);
  showCount();
}
const showCount = () => ($('#gCount').textContent = `${fmt(gList.filter(p => !p.gone).length)} photos`);
function tile(p) {
  const f = document.createElement('figure'), img = new Image();
  f.dataset.id = p.id; f.title = p.name;
  img.loading = 'lazy'; img.decoding = 'async'; img.onload = () => img.classList.add('in'); img.src = p.u; img.alt = p.name;
  f.appendChild(img);
  if (p.gone) { f.classList.add('gone'); f.insertAdjacentHTML('beforeend', `<span class="t-badge gone">Now in ${LABEL[setOf(p)]}</span>`); }
  else if (p.moved) f.insertAdjacentHTML('beforeend', `<span class="t-badge">${p.mine ? 'Your frame' : 'Moved by you'}</span>`);
  if (p.frames && gSet !== 'not_selected')
    f.insertAdjacentHTML('beforeend', `<span class="t-frames" title="${p.frames} similar frame${p.frames > 1 ? 's' : ''} of this shot, each in its own folder: open the photo to see them and move any one of them">${STACK}+${fmt(p.frames)} similar</span>`);
  const moves = document.createElement('div');
  moves.className = 't-moves';
  TIERS.filter(t => t !== setOf(p)).forEach(t => {
    const b = document.createElement('button');
    b.className = 't-move'; b.innerHTML = `${LABEL[t]} ${ARROW}`; b.title = `Move to ${LABEL[t]}`;
    b.onclick = e => { e.stopPropagation(); movePhoto(p, t); };
    moves.appendChild(b);
  });
  f.appendChild(moves);
  f.onclick = () => showBig(gList.indexOf(p));
  return f;
}
function updateResult(r) {
  Object.assign(result, r || {});
  $('#doneSel').textContent = fmt(result.selected);
  renderStats();
  showCount();
}
// keep the open gallery in step with a photo's fresh entry after a move or a swap. A photo that
// leaves this folder stays where it is, dimmed and marked with its new folder (so nothing else
// seems to move and Undo is easy), until the gallery is opened again; one that comes in appears at
// its place in time.
function place(e) {
  if (!e) return;
  const grid = $('#gGrid'), here = setOf(e) === gSet, i = gList.findIndex(p => p.id === e.id);
  if (i >= 0) {
    const entry = here ? e : { ...e, gone: true };
    gList[i] = entry;
    const fig = grid.querySelector(`figure[data-id="${e.id}"]`);
    if (fig) { const t = tile(entry); if (here) t.classList.add('back'); fig.replaceWith(t); }
    return;
  }
  if (!here) return;
  let k = gList.findIndex(p => p.t > e.t || (p.t === e.t && p.name > e.name));
  if (k < 0) k = gList.length;
  gList.splice(k, 0, e);
  const t = tile(e); t.classList.add('back');
  const next = gList[k + 1] && grid.querySelector(`figure[data-id="${gList[k + 1].id}"]`);
  grid.insertBefore(t, next || null);
}
const bigOpen = () => !$('#lightbox').hidden;
function reopen(anchorId, preview) {            // the big view again, on this photo of the gallery
  if (!bigOpen()) return;
  if (!gList.length) return closeBig();
  const i = gList.findIndex(p => p.id === anchorId);
  showBig(i >= 0 ? i : Math.min(gIdx, gList.length - 1), i >= 0 ? preview : undefined);
}
async function movePhoto(p, to) {
  if (!p || p.busy) return;
  const from = setOf(p), at = gList.indexOf(p);
  if (!to || to === from) return;
  p.busy = true;
  try {
    const r = await api('/api/move', { id: p.id, to });
    fCache.clear();
    place(r.photo || { ...p, tier: to, sel: to !== 'not_selected' });
    updateResult(r.result);
    if (bigOpen()) reopen(p.id, r.photo);
    toast(`${p.name} moved to ${LABEL[to]}.`, 8000, { label: 'Undo', fn: () => undoMove(p, from) });
  } catch (e) { toast(e.message, 8000); }
  p.busy = false;
}
async function undoMove(p, from) {
  try {
    const r = await api('/api/move', { id: p.id, to: from });
    fCache.clear();
    place(r.photo || { ...p, tier: from, sel: from !== 'not_selected' });
    updateResult(r.result);
    if (bigOpen()) reopen(strip.anchor && strip.anchor.id, strip.cur && strip.cur.id === p.id ? r.photo : strip.cur && strip.cur.id);
    toast(`${p.name} is back in ${LABEL[from]}.`, 3000);
  } catch (e) { toast(e.message, 8000); }
}

// ---------- Swap frame: the other frames of the shot, under the big photo ----------
// A shot = near-identical frames. KeepSake keeps its best frame; the editor can see the others
// here (KeepSake's next best is marked 2nd) and put the one the studio likes into Selected instead.
const fCache = new Map();
let strip = { anchor: null, list: [], shotN: 0, cur: null }, swapping = false;
async function loadFrames(id) {
  if (!fCache.has(id)) fCache.set(id, api(`/api/frames?id=${id}`).catch(() => ({ shot: [], moment: [] })));
  return fCache.get(id);
}
function showBig(i, preview) {               // preview: the frame to show big (an entry or an id; default the photo itself)
  if (!gList.length || i < 0) return;
  gIdx = (i + gList.length) % gList.length;
  const p = gList[gIdx], lb = $('#lightbox');
  $('#lbCount').textContent = `${fmt(gIdx + 1)} / ${fmt(gList.length)}`;
  const pid = preview == null ? p.id : (preview.id ?? preview);
  const kept = strip.list.length > 1 && strip.list.some(f => f.id === p.id);    // same shot: keep the strip while it refreshes
  const old = strip.list.find(f => f.id === pid);
  strip = { anchor: p, list: kept ? strip.list : [], shotN: kept ? strip.shotN : 0, cur: p };
  lb.hidden = false;
  if (!kept) {
    const expect = (p.frames || 0) > 0;
    lb.classList.toggle('has-frames', expect); $('#lbFrames').hidden = !expect;
    $('#lfStrip').innerHTML = ''; $('#lfTitle').textContent = '';
  }
  showFrame(typeof preview === 'object' && preview ? preview : pid === p.id ? p : (old || p));
  loadFrames(p.id).then(r => {
    if (strip.anchor !== p) return;
    strip.list = [...r.shot, ...r.moment.map(f => ({ ...f, near: true }))];
    strip.shotN = r.shot.length;
    showFrame(strip.list.find(f => f.id === strip.cur.id) || strip.cur);
    renderStrip();
  });
  [gIdx + 1, gIdx - 1].forEach(k => { const q = gList[(k + gList.length) % gList.length]; if (q) new Image().src = `${q.u}/big`; });
}
function showFrame(f) {                      // the big photo: this frame (not loaded again if it is already there)
  const img = $('#lightbox img'), url = new URL(`${f.u}/big`, location.href).href;
  if (img.src !== url || !img.complete) {
    img.classList.remove('in', 'drag'); img.style.transform = '';
    img.onload = () => img.classList.add('in');
    img.src = url;
  }
  frameLabels(f);
}
function frameLabels(f) {
  strip.cur = f;
  $('#lbName').textContent = f.name;
  $('#lbReason').textContent = f.reason || ''; $('#lbReason').title = f.reason || '';
  $('#lbMoves').querySelectorAll('button').forEach(b => {
    const here = b.dataset.to === setOf(f);
    b.classList.toggle('on', here); b.disabled = here;
  });
  const k = keptFor(f), use = $('#lbUse');
  use.hidden = !k;
  if (k) use.title = `Put ${f.name} into Selected in place of ${k.name} (Enter)`;
  markCur();
}
function keptFor(f) {                        // the frame in Selected that `f` would replace
  if (!f || setOf(f) !== 'not_selected' || !strip.list.length) return null;
  const shot = strip.list.slice(0, strip.shotN), at = strip.list.findIndex(x => x.id === f.id);
  if (at < 0) return null;
  let best = null;
  shot.forEach((x, k) => { if (x.sel && (!best || Math.abs(k - at) < Math.abs(best.k - at))) best = { x, k }; });
  return best && best.x;
}
function renderStrip() {
  const el = $('#lfStrip'), L = strip.list, box = $('#lbFrames'), lb = $('#lightbox');
  const show = L.length > 1;
  lb.classList.toggle('has-frames', show);
  box.hidden = !show;
  el.innerHTML = '';
  if (!show) return;
  const near = L.length - strip.shotN, n = strip.shotN;
  const inShot = L.slice(0, n), count = t => inShot.filter(f => setOf(f) === t).length;
  const where = TIERS.filter(t => count(t)).map(t => `${fmt(count(t))} in ${LABEL[t]}`).join(', ');
  $('#lfTitle').innerHTML = (n > 1 ? `<b>${fmt(n)}</b> frames of this shot: ${where}` : 'Only frame of this shot')
    + (near ? ` &middot; then ${fmt(near)} more from this moment` : '');
  const frag = document.createDocumentFragment();
  L.forEach((f, k) => {
    if (k === n && k > 0) frag.appendChild(Object.assign(document.createElement('i'),
      { className: 'lf-gap', title: 'Other photos of the same moment (different shots)', innerHTML: '<span>moment</span>' }));
    const b = document.createElement('button'), img = new Image();
    const tier = setOf(f);
    b.className = 'lf ' + tier; b.dataset.id = f.id;
    const second = !f.near && tier === 'not_selected' && f.rank === 2;
    b.title = `${f.name} · in ${LABEL[tier]}` + (second ? " · KeepSake's next best frame" : '') + (f.near ? ' · same moment, another shot' : '');
    img.decoding = 'async'; img.alt = ''; img.onload = () => img.classList.add('in'); img.src = f.u;
    b.appendChild(img);
    if (tier !== 'not_selected') b.insertAdjacentHTML('beforeend', `<span class="lf-t ${tier}">${LABEL[tier]}</span>`);
    else if (second) b.insertAdjacentHTML('beforeend', '<span class="lf-2">2nd</span>');
    b.onclick = e => { e.stopPropagation(); previewFrame(f); };
    frag.appendChild(b);
  });
  el.appendChild(frag);
  markCur(true);
}
function markCur(jump) {
  const el = $('#lfStrip'), id = strip.cur && strip.cur.id;
  let cur = null;
  el.querySelectorAll('.lf').forEach(b => { const on = b.dataset.id === String(id); b.classList.toggle('cur', on); if (on) cur = b; });
  if (cur) el.scrollTo({ left: cur.offsetLeft - el.clientWidth / 2 + cur.offsetWidth / 2, behavior: jump ? 'auto' : 'smooth' });
}
function previewFrame(f) {
  if (!f || (strip.cur && strip.cur.id === f.id)) return;
  showFrame(f);
  const k = strip.list.indexOf(f);
  [k + 1, k - 1].forEach(j => { const q = strip.list[j]; if (q) new Image().src = `${q.u}/big`; });
}
function stepFrame(d) {
  if (strip.list.length < 2) return;
  const k = strip.list.findIndex(f => f.id === strip.cur.id);
  previewFrame(strip.list[Math.max(0, Math.min(strip.list.length - 1, k + d))]);
}
async function useFrame() {
  const f = strip.cur, k = keptFor(f);
  if (!f || !k || swapping) return;
  swapping = true;
  try {
    const r = await api('/api/swap', { id: f.id, replace: k.id });
    const anchor = afterSwap(r);
    reopen(anchor, r.selected);
    toast(`${r.selected.name} is in Selected now; ${r.not_selected.name} went to Not selected.`, 12000,
      { label: 'Undo', fn: () => undoSwap(r) });
  } catch (e) { toast(e.message, 8000); }
  swapping = false;
}
function afterSwap(r) {                      // returns the photo the big view stays on
  fCache.clear();
  const a = strip.anchor && strip.anchor.id;
  place(r.selected); place(r.not_selected);
  updateResult(r.result);
  return a;                                   // both stay in the gallery: the one that left, dimmed
}
async function undoSwap(r) {
  if (swapping) return;
  swapping = true;
  try {
    const u = await api('/api/swap', { id: r.not_selected.id, replace: r.selected.id });
    reopen(afterSwap(u));
    toast(`${u.selected.name} is back in Selected.`, 3000);
  } catch (e) { toast(e.message, 8000); }
  swapping = false;
}
async function moveFrame(f, to) {            // Move on a frame from the strip (not the gallery's own photo)
  if (!f || f.busy) return;
  const from = setOf(f), a = strip.anchor && strip.anchor.id;
  if (!to || to === from) return;
  f.busy = true;
  try {
    const r = await api('/api/move', { id: f.id, to });
    fCache.clear();
    place(r.photo);
    updateResult(r.result);
    reopen(a, r.photo || f.id);
    toast(`${f.name} moved to ${LABEL[to]}.`, 8000, { label: 'Undo', fn: () => undoMove(r.photo || f, from) });
  } catch (e) { toast(e.message, 8000); }
  f.busy = false;
}
const moveShown = to => (strip.cur && strip.anchor && strip.cur.id !== strip.anchor.id ? moveFrame(strip.cur, to) : movePhoto(gList[gIdx], to));
const shownTier = () => setOf(strip.cur || gList[gIdx] || {});

const closeBig = () => ($('#lightbox').hidden = true);
$('#lbPrev').onclick = e => { e.stopPropagation(); showBig(gIdx - 1); };
$('#lbNext').onclick = e => { e.stopPropagation(); showBig(gIdx + 1); };
$('#lbClose').onclick = e => { e.stopPropagation(); closeBig(); };
$('#lbMoves').onclick = e => { e.stopPropagation(); const b = e.target.closest('button'); if (b && !b.disabled) moveShown(b.dataset.to); };
$('#lbUse').onclick = e => { e.stopPropagation(); useFrame(); };
$('#lfStrip').addEventListener('wheel', e => {      // a mouse wheel scrolls the strip sideways
  const el = e.currentTarget;
  if (Math.abs(e.deltaY) > Math.abs(e.deltaX) && el.scrollWidth > el.clientWidth) { el.scrollLeft += e.deltaY; e.preventDefault(); }
}, { passive: false });
(() => {                                   // swipe: drag left = next, right = previous; a plain click outside closes
  const lb = $('#lightbox'), img = lb.querySelector('img');
  let x0 = null, y0 = 0;
  lb.addEventListener('pointerdown', e => { if (e.target.closest('.lb-btn, .lb-frames, .lb-moves')) return; x0 = e.clientX; y0 = e.clientY; img.classList.add('drag'); });
  lb.addEventListener('pointermove', e => { if (x0 === null) return; const m = e.clientX - x0; img.style.transform = `translateX(${m}px) rotate(${m / 80}deg)`; });
  const end = e => {
    if (x0 === null) return;
    const dx = e.clientX - x0, dy = e.clientY - y0; x0 = null; img.classList.remove('drag'); img.style.transform = '';
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) showBig(gIdx + (dx < 0 ? 1 : -1));
    else if (Math.abs(dx) < 6 && Math.abs(dy) < 6 && e.target !== img) closeBig();
  };
  lb.addEventListener('pointerup', end); lb.addEventListener('pointercancel', end);
  let lock = 0;
  lb.addEventListener('wheel', e => {
    if (e.target.closest('.lb-frames') || Math.abs(e.deltaX) < 30 || Math.abs(e.deltaX) < Math.abs(e.deltaY) || Date.now() < lock) return;
    lock = Date.now() + 450; showBig(gIdx + (e.deltaX > 0 ? 1 : -1));
  }, { passive: true });
})();
document.querySelectorAll('[data-gallery]').forEach(b => (b.onclick = () => openGallery(b.dataset.gallery)));
$('#gSeg').onclick = e => { const b = e.target.closest('button'); if (b) openGallery(b.dataset.set); };
$('#gClose').onclick = () => ($('#gallery').hidden = true);
addEventListener('keydown', e => {
  const big = bigOpen();
  if (e.key === 'Escape') { if (big) closeBig(); else $('#gallery').hidden = true; }
  if (!big) return;
  if (e.key === 'ArrowRight') showBig(gIdx + 1);
  if (e.key === 'ArrowLeft') showBig(gIdx - 1);
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); stepFrame(e.key === 'ArrowDown' ? 1 : -1); }
  if (e.key === 'Enter' && !e.repeat) { e.preventDefault(); useFrame(); }
  if (['1', '2', '3'].includes(e.key) && !e.repeat) moveShown(TIERS[+e.key - 1]);
  if ((e.key === 'm' || e.key === 'M') && !e.repeat) moveShown(shownTier() === 'not_selected' ? 'selected' : 'not_selected');
});

boot();
