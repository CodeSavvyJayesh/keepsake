import { Scene } from '/ui/scene.js';

const $ = s => document.querySelector(s);
const fmt = n => (n ?? 0).toLocaleString('en-IN');
const wait = ms => new Promise(r => setTimeout(r, ms));
const scene = new Scene($('#scene'));

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
const pick = { folder: '', output: '', keep: 'studio' };
let home = {};

async function boot() {
  home = await api('/api/home').catch(() => ({ brand: {}, recent: [], sample: [] }));
  const b = home.brand || {};
  if (b.logo) { for (const img of [$('#brandLogo'), $('#introLogo')]) { img.src = b.logo; img.hidden = false; } $('#brandRule').hidden = false; }
  if (b.background) { $('#photoBg').style.backgroundImage = `url("${b.background}")`; document.body.classList.add('has-photo-bg'); }
  if (b.name) { $('#kicker').innerHTML = `${b.name}<span class="sep"></span>Wedding photo culling`; document.title = `KeepSake · ${b.name}`; }
  renderRecent(home.recent || []);
  scene.setPhotos(home.sample || []);

  const st = await api('/api/state').catch(() => ({ status: 'idle' }));
  if (st.status === 'running' || st.status === 'done') {          // the window was reopened: no opening scene
    scene.start(); scene.open(); ready();
    if (st.status === 'running') return follow(st);
    finish(st); return;
  }
  // scene time, with a wall-clock safety net so the app always opens
  const either = (p, ms) => Promise.race([p, wait(ms)]);
  await wait(450);
  await either(scene.start(), 9000);                  // the photos are dealt into a tiny tilted ring
  await either(scene.wait(0.7), 2500);                // a beat to see the ring
  scene.open(); document.body.classList.add('drop');  // the camera drops inside
  await either(scene.wait(1.5), 4000);
  ready(); show('welcome');
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
    $('#setupNote').textContent = (c.prepared ? `Read once before (${fmt(c.photos)} photos), so this run is quicker. ` : '') + `${c.free_gb} GB free where it is saved.`;
    $('#chooser').hidden = true; $('#setup').hidden = false;
  } catch (e) { toast(e.message); }
}
$('#chooseBtn').onclick = async () => {
  $('#chooseBtn').disabled = true;
  try { const r = await api('/api/choose', { kind: 'photos' }); if (r.path) await useFolder(r.path); } catch (e) { toast(e.message); }
  $('#chooseBtn').disabled = false;
};
$('#pasteForm').onsubmit = e => { e.preventDefault(); useFolder($('#pathInput').value.trim().replace(/^"|"$/g, '')); };
$('#changeBtn').onclick = () => { $('#setup').hidden = true; $('#chooser').hidden = false; };
$('#outBtn').onclick = async () => {
  const r = await api('/api/choose', { kind: 'output', start: pick.output }).catch(e => toast(e.message));
  if (r?.path) { pick.output = r.path; $('#outPath').textContent = r.path; }
};
const KEEP_NOTE = { '0.07': 'about 7 in 100', studio: 'about 1 in 10, like your studio', '0.15': 'about 15 in 100' };
$('#keepSeg').onclick = e => {
  const b = e.target.closest('button'); if (!b) return;
  $('#keepSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  pick.keep = b.dataset.keep; $('#keepNote').textContent = KEEP_NOTE[pick.keep];
};
$('#startBtn').onclick = async () => {
  $('#startBtn').disabled = true;
  try { follow(await api('/api/start', { folder: pick.folder, output: pick.output, keep: pick.keep, copy_not_selected: $('#copyNot').checked })); }
  catch (e) { toast(e.message); }
  $('#startBtn').disabled = false;
};

// ---------- 2. working ----------
let polling = false, lastThumbs = 0, sorted = false, shownPct = 0;
const SHORT = { read: 'Reading', quality: 'Sharpness', faces: 'Faces', clip: 'Understanding', groups: 'Moments', choose: 'Choosing', copy: 'Copying' };
function follow(st) {
  show('working');
  $('#wkName').textContent = st.name;
  sorted = false; shownPct = 0; $('#steps').innerHTML = '';
  if (!polling) { polling = true; poll(st); }
}
function render(st) {
  const ol = $('#steps');
  if (!ol.children.length) st.stages.forEach(s => ol.insertAdjacentHTML('beforeend', `<li data-key="${s.key}">${SHORT[s.key] || s.label}</li>`));
  st.stages.forEach(s => { ol.querySelector(`[data-key="${s.key}"]`).className = s.state; });
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
function finish(st) {
  const r = result = st.result || {};
  show('done');
  api('/api/thumbs?set=selected&n=60').then(x => scene.sort(x.photos)).catch(() => {});
  $('#doneKicker').textContent = `Finished · ${st.name}`;
  countUp($('#doneSel'), r.selected || 0);
  const total = r.total || (r.selected || 0) + (r.not_selected || 0);
  $('#doneSub').textContent = `From ${fmt(total)} photos · ${r.time || st.elapsed_text} · saved in ${r.output || st.output}`;
  renderStats();
}
function renderStats() {
  const r = result;
  $('#openNot').hidden = !(r.copied_not_selected !== false || r.moved);
  const stats = [['Selected', fmt(r.selected)], ['Not selected', fmt(r.not_selected)], ['Moments', r.shots ? fmt(r.shots) : '—']];
  if (r.studio_picks) stats.push(['Studio’s picks found', `${fmt(r.studio_in_selected)} / ${fmt(r.studio_picks)}`]);
  if (r.moved) stats.push(['Moved by you', fmt(r.moved)]);
  $('#stats').innerHTML = stats.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  $('#gSeg').querySelector('[data-set=selected]').innerHTML = `Selected <i>${fmt(r.selected)}</i>`;
  $('#gSeg').querySelector('[data-set=not_selected]').innerHTML = `Not selected <i>${fmt(r.not_selected)}</i>`;
}
document.querySelectorAll('[data-open]').forEach(b => (b.onclick = () => api('/api/open', { which: b.dataset.open }).catch(e => toast(e.message))));
$('#againBtn').onclick = async () => {
  await api('/api/reset', {}).catch(() => {});
  $('#setup').hidden = true; $('#chooser').hidden = false;
  scene.setPhotos(home.sample || []); show('welcome');
};

// ---------- gallery and the big view (a person can move any photo across) ----------
let gList = [], gIdx = 0, gSet = 'selected';
const other = set => (set === 'selected' ? 'not_selected' : 'selected');
const LABEL = { selected: 'Selected', not_selected: 'Not selected' };
const ARROW = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';

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
  $('#gCount').textContent = `${fmt(gList.length)} photos`;
}
function tile(p) {
  const f = document.createElement('figure'), img = new Image();
  f.dataset.id = p.id; f.title = p.name;
  img.loading = 'lazy'; img.decoding = 'async'; img.onload = () => img.classList.add('in'); img.src = p.u; img.alt = p.name;
  f.appendChild(img);
  if (p.moved) f.insertAdjacentHTML('beforeend', '<span class="t-badge">Moved by you</span>');
  const b = document.createElement('button');
  b.className = 't-move'; b.innerHTML = `Move to ${LABEL[other(gSet)]} ${ARROW}`; b.title = `Move to ${LABEL[other(gSet)]}`;
  b.onclick = e => { e.stopPropagation(); movePhoto(p); };
  f.appendChild(b);
  f.onclick = () => showBig(gList.indexOf(p));
  return f;
}
function updateResult(r) {
  Object.assign(result, r || {});
  $('#doneSel').textContent = fmt(result.selected);
  renderStats();
  $('#gCount').textContent = `${fmt(gList.length)} photos`;
}
async function movePhoto(p) {
  if (!p || p.busy) return;
  const from = gSet, to = other(from), at = gList.indexOf(p);
  p.busy = true;
  try {
    const r = await api('/api/move', { id: p.id, to });
    const i = gList.indexOf(p);
    if (i >= 0) gList.splice(i, 1);
    const f = $(`#gGrid figure[data-id="${p.id}"]`);
    if (f) { f.classList.add('out'); setTimeout(() => f.remove(), 350); }
    updateResult(r.result);
    if (!$('#lightbox').hidden) { if (gList.length) showBig(Math.min(i, gList.length - 1)); else closeBig(); }
    toast(`${p.name} moved to ${LABEL[to]}.`, 8000, { label: 'Undo', fn: () => undoMove(p, from, at) });
  } catch (e) { toast(e.message, 8000); }
  p.busy = false;
}
async function undoMove(p, from, at) {
  try {
    const r = await api('/api/move', { id: p.id, to: from });
    if (gSet === from && !gList.includes(p)) {
      gList.splice(Math.min(at, gList.length), 0, p);
      const next = $('#gGrid').children[Math.min(at, gList.length - 1)];
      const f = tile(p); f.classList.add('back');
      $('#gGrid').insertBefore(f, next && next.dataset.id !== String(p.id) ? next : null);
    }
    updateResult(r.result);
    toast(`${p.name} is back in ${LABEL[from]}.`, 3000);
  } catch (e) { toast(e.message, 8000); }
}
function showBig(i) {
  if (!gList.length || i < 0) return;
  gIdx = (i + gList.length) % gList.length;
  const p = gList[gIdx], lb = $('#lightbox'), img = lb.querySelector('img');
  img.classList.remove('in', 'drag'); img.style.transform = '';
  img.onload = () => img.classList.add('in');
  img.src = `${p.u}/big`;
  $('#lbCount').textContent = `${fmt(gIdx + 1)} / ${fmt(gList.length)}`;
  $('#lbName').textContent = p.name;
  $('#lbReason').textContent = p.reason || ''; $('#lbReason').title = p.reason || '';
  $('#lbMove').innerHTML = `Move to ${LABEL[other(gSet)]} ${ARROW}<kbd>M</kbd>`;
  lb.hidden = false;
  [gIdx + 1, gIdx - 1].forEach(k => { const q = gList[(k + gList.length) % gList.length]; if (q) new Image().src = `${q.u}/big`; });
}
const closeBig = () => ($('#lightbox').hidden = true);
$('#lbPrev').onclick = e => { e.stopPropagation(); showBig(gIdx - 1); };
$('#lbNext').onclick = e => { e.stopPropagation(); showBig(gIdx + 1); };
$('#lbClose').onclick = e => { e.stopPropagation(); closeBig(); };
$('#lbMove').onclick = e => { e.stopPropagation(); movePhoto(gList[gIdx]); };
(() => {                                   // swipe: drag left = next, right = previous; a plain click outside closes
  const lb = $('#lightbox'), img = lb.querySelector('img');
  let x0 = null, y0 = 0;
  lb.addEventListener('pointerdown', e => { if (e.target.closest('.lb-btn')) return; x0 = e.clientX; y0 = e.clientY; img.classList.add('drag'); });
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
    if (Math.abs(e.deltaX) < 30 || Math.abs(e.deltaX) < Math.abs(e.deltaY) || Date.now() < lock) return;
    lock = Date.now() + 450; showBig(gIdx + (e.deltaX > 0 ? 1 : -1));
  }, { passive: true });
})();
document.querySelectorAll('[data-gallery]').forEach(b => (b.onclick = () => openGallery(b.dataset.gallery)));
$('#gSeg').onclick = e => { const b = e.target.closest('button'); if (b) openGallery(b.dataset.set); };
$('#gClose').onclick = () => ($('#gallery').hidden = true);
addEventListener('keydown', e => {
  const big = !$('#lightbox').hidden;
  if (e.key === 'Escape') { if (big) closeBig(); else $('#gallery').hidden = true; }
  if (big && e.key === 'ArrowRight') showBig(gIdx + 1);
  if (big && e.key === 'ArrowLeft') showBig(gIdx - 1);
  if (big && (e.key === 'm' || e.key === 'M') && !e.repeat) movePhoto(gList[gIdx]);
});

boot();
