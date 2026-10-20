// KeepSake 3D scene, written directly in WebGL (no libraries: works offline on any laptop).
//
//   1. deal   - wedding photos are dealt, one by one, into a tiny tilted ring (cards land and flip)
//   2. drop   - the camera drops inside the ring, which opens around the viewer
//   3. carousel - a slow 3D carousel of prints with soft reflections; drag to spin it
//   working: the carousel turns faster, each photo lights up in gold as it passes the front
//   done:    the carousel holds only the selected photos, all edged in gold

const N = 22;                       // prints in the ring
const R = 6.4;                      // ring radius
const W = 1.25, H = 1.62;           // print size (about 4:5)
const FLOOR = -H / 2 - 0.06;        // where the reflections start

// ring placement (in front of the camera): dealt far away, small and tilted; open around the viewer
const DEALT = { p: [0, -0.95, -16], tilt: 0.5, s: 0.4 };
const OPEN = { p: [0, -0.32, -3.6], tilt: 0.06, s: 1.0 };
const DEAL_S = 0.15 + (N - 1) * 0.075 + 0.7;   // seconds until the last print lands
const DROP_S = 2.8;                             // seconds for the camera to drop inside

// ---------- matrices (column-major, like WebGL) ----------
const I4 = () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
const T = (x, y, z) => { const m = I4(); m[12] = x; m[13] = y; m[14] = z; return m; };
const S = (x, y, z) => { const m = I4(); m[0] = x; m[5] = y; m[10] = z; return m; };
const RX = a => { const m = I4(), c = Math.cos(a), s = Math.sin(a); m[5] = c; m[6] = s; m[9] = -s; m[10] = c; return m; };
const RY = a => { const m = I4(), c = Math.cos(a), s = Math.sin(a); m[0] = c; m[2] = -s; m[8] = s; m[10] = c; return m; };
const RZ = a => { const m = I4(), c = Math.cos(a), s = Math.sin(a); m[0] = c; m[1] = s; m[4] = -s; m[5] = c; return m; };
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(e, c) {
  let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2], l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = zz, xy = 0, xz = -zx; l = Math.hypot(xx, xz); xx /= l; xz /= l;           // up = +y
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
    -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, k) => a + (b - a) * k;
const easeInOut = k => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const easeOutBack = k => 1 + 2.2 * Math.pow(k - 1, 3) + 1.2 * Math.pow(k - 1, 2);
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const rand = (seed => () => (seed = (seed * 16807) % 2147483647) / 2147483647)(1020);

// ---------- shaders ----------
const CARD_VS = `
attribute vec2 aCorner; uniform mat4 uProj, uView, uModel; uniform vec2 uSize; uniform float uPad; varying vec2 vP;
void main(){ vec2 p = aCorner * (uSize + vec2(uPad * 2.0)); vP = p; gl_Position = uProj * uView * uModel * vec4(p, 0.0, 1.0); }`;
const CARD_FS = `
precision highp float;
varying vec2 vP;
uniform vec2 uSize, uUvScale; uniform float uOpacity, uGlow, uPass, uBright, uReflect; uniform sampler2D uTex; uniform vec3 uGold;
float box(vec2 p, vec2 b, float r){ vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
void main(){
  float d = box(vP, uSize * 0.5, 0.02);
  if (uPass < 0.5) {                                   // soft gold light behind the print
    float g = exp(-max(d, 0.0) * 7.0) * smoothstep(-0.03, 0.0, d) * uGlow * uOpacity * 0.7;
    gl_FragColor = vec4(uGold * g, g); return;
  }
  float inside = 1.0 - smoothstep(-0.004, 0.004, d);
  vec2 uv = (vP / uSize) * uUvScale + 0.5; uv.y = 1.0 - uv.y;
  vec3 col;
  bool face = gl_FrontFacing; if (uReflect > 0.5) face = !face;          // the mirror flips sides
  if (face) {
    col = texture2D(uTex, uv).rgb;
    col = pow(col, vec3(0.95)) * vec3(1.03, 1.0, 0.95) * uBright;            // a touch of warmth
    float mat = 1.0 - smoothstep(-0.05, -0.035, d);                          // thin ivory mat
    col = mix(vec3(0.93, 0.88, 0.8) * uBright, col, mat);
  } else {                                                                    // the back of a print
    col = mix(vec3(0.13, 0.095, 0.06), vec3(0.22, 0.165, 0.1), vP.y / uSize.y + 0.5) * uBright;
    col += uGold * 0.55 * smoothstep(0.012, 0.0, abs(d + 0.06));
  }
  col += uGold * smoothstep(0.012, 0.0, abs(d + 0.004)) * (0.35 + uGlow);     // gold edge
  float a = inside * uOpacity;
  if (uReflect > 0.5) a *= 0.2 * (1.0 - smoothstep(-0.5, 0.15, vP.y / uSize.y));  // fades away from the floor
  gl_FragColor = vec4(col * a, a);
}`;
const DUST_VS = `
attribute vec4 aSeed; uniform mat4 uProj, uView; uniform float uTime, uPR; varying float vA;
void main(){
  float y = mod(aSeed.y + uTime * 0.05 * (0.5 + aSeed.w), 9.0) - 4.5;
  float a = aSeed.x * 6.283 + uTime * 0.02 * (aSeed.w - 0.5);
  vec3 p = vec3(cos(a) * aSeed.z, y, sin(a) * aSeed.z - 6.0);
  vec4 mv = uView * vec4(p, 1.0); gl_Position = uProj * mv;
  gl_PointSize = clamp((0.6 + aSeed.w * 1.6) * uPR * (160.0 / -mv.z), 0.0, 6.0);
  vA = (0.25 + 0.5 * aSeed.w) * (0.6 + 0.4 * sin(uTime * 1.3 + aSeed.x * 50.0)) * smoothstep(-30.0, -4.0, mv.z) * smoothstep(-0.5, -3.0, mv.z);
}`;
const DUST_FS = `
precision mediump float; varying float vA;
void main(){ float a = smoothstep(0.5, 0.0, length(gl_PointCoord - 0.5)); a *= a * vA; gl_FragColor = vec4(vec3(1.0, 0.78, 0.45) * a, a); }`;

function program(gl, vs, fs) {
  const make = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, make(gl.VERTEX_SHADER, vs)); gl.attachShader(p, make(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}, a = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const n = gl.getActiveUniform(p, i).name; u[n] = gl.getUniformLocation(p, n); }
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES); i++) { const n = gl.getActiveAttrib(p, i).name; a[n] = gl.getAttribLocation(p, n); }
  return { p, u, a };
}

function blankPrint(i) {          // a warm, softly lit blank print until a real photo arrives
  const c = document.createElement('canvas'); c.width = 200; c.height = 260;
  const g = c.getContext('2d'), k = i % 5;
  const grad = g.createLinearGradient(0, 0, 200, 260);
  grad.addColorStop(0, ['#3a2416', '#2c1c12', '#4a2c18', '#24170f', '#3b2619'][k]); grad.addColorStop(1, '#0d0806');
  g.fillStyle = grad; g.fillRect(0, 0, 200, 260);
  for (let j = 0; j < 6; j++) {
    const x = rand() * 200, y = rand() * 260, r = 10 + rand() * 40, rg = g.createRadialGradient(x, y, 0, x, y, r);
    rg.addColorStop(0, 'rgba(255,190,110,0.22)'); rg.addColorStop(1, 'rgba(255,190,110,0)'); g.fillStyle = rg; g.fillRect(0, 0, 200, 260);
  }
  return c;
}

export class Scene {
  constructor(canvas) {
    this.canvas = canvas; this.ok = false;
    const gl = canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: true });
    if (!gl) return;
    this.gl = gl;
    try { this.cp = program(gl, CARD_VS, CARD_FS); this.dp = program(gl, DUST_VS, DUST_FS); }
    catch (e) { console.warn('3D scene off:', e); return; }
    this.ok = true;
    this.t = 0; this.last = performance.now(); this.started = null;
    this.mode = 'intro'; this.progress = 0;
    this.spin = 0; this.spinSpeed = 0; this.spinTarget = 0.05; this.dragV = 0;
    this.mouse = [0, 0]; this.sm = [0, 0];
    this.ring = { p: DEALT.p.slice(), tilt: DEALT.tilt, s: DEALT.s };
    this.dropT = null; this.timers = [];
    this.quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]), gl.STATIC_DRAW);
    const seeds = new Float32Array(420 * 4);
    for (let i = 0; i < 420; i++) seeds.set([rand(), rand() * 9, 2 + rand() * 13, rand()], i * 4);
    this.dust = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.dust); gl.bufferData(gl.ARRAY_BUFFER, seeds, gl.STATIC_DRAW);
    this.blank = Array.from({ length: 5 }, (_, i) => this.texture(blankPrint(i)));
    this.cards = Array.from({ length: N }, (_, i) => ({ i, tex: this.blank[i % 5], url: null, glow: 0, lift: 0, deal: 0, sel: 0 }));
    this.queue = []; this.pool = [];
    addEventListener('resize', () => this.resize()); this.resize();
    addEventListener('pointermove', e => { this.mouse = [e.clientX / innerWidth * 2 - 1, e.clientY / innerHeight * 2 - 1]; });
    let dragX = null;
    canvas.addEventListener('pointerdown', e => { dragX = e.clientX; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => { if (dragX === null) return; const dx = e.clientX - dragX; dragX = e.clientX; this.spin -= dx * 0.004; this.dragV = -dx * 0.25; });
    canvas.addEventListener('pointerup', () => { dragX = null; });
    requestAnimationFrame(t => this.frame(t));
  }

  texture(src) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_2D, k, v);
    return { t, aspect: src.width / src.height };
  }
  load(card, url) {
    if (!url || card.url === url) return;
    card.url = url;
    const img = new Image();
    img.onload = () => { if (card.url === url) card.tex = this.texture(img); };
    img.src = url;
  }

  // ---------- what the app asks for ----------
  setPhotos(urls) { if (!this.ok || !urls.length) return; this.pool = urls.slice(); this.cards.forEach((c, k) => this.load(c, urls[k % urls.length])); }
  feed(urls) { if (!this.ok) return; const seen = new Set(this.cards.map(c => c.url)); urls.filter(u => !seen.has(u)).forEach(u => this.queue.push(u)); }
  sort(selected) { if (!this.ok || !selected.length) return; this.queue = []; this.selected = selected; this.selIndex = 0; this.cards.forEach(c => (c.sel = 0)); }
  // scene-time waits: the opening keeps its rhythm even on a slow laptop
  wait(sec) { return this.ok ? new Promise(r => this.timers.push({ at: this.t + sec, r })) : Promise.resolve(); }
  start() { if (this.ok && this.started === null) this.started = this.t; return this.ok ? this.wait(DEAL_S - (this.t - this.started)) : Promise.resolve(); }   // deal; resolves when dealt
  open() { if (this.ok && this.dropT === null) this.dropT = this.t; }           // the camera drops inside
  setMode(mode) {
    if (!this.ok) return;
    this.mode = mode;
    this.spinTarget = mode === 'working' ? 0.32 : mode === 'done' ? 0.07 : 0.05;
    if (mode !== 'done') { this.selected = null; this.cards.forEach(c => (c.sel = 0)); }
    if (mode !== 'intro') this.open();
  }
  setProgress(p) { if (this.ok) this.progress = p; }
  resize() {
    if (!this.ok) return;
    const dpr = Math.min(devicePixelRatio || 1, 1.75);
    this.canvas.width = Math.floor(innerWidth * dpr); this.canvas.height = Math.floor(innerHeight * dpr);
    this.dpr = dpr; this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  frame(now) {
    const dt = clamp((now - this.last) / 1000, 0, 0.1); this.last = now; this.t += dt;
    if (this.timers.length) this.timers = this.timers.filter(x => (this.t >= x.at ? (x.r(), false) : true));
    const gl = this.gl, t = this.t;
    const km = 1 - Math.exp(-dt * 2.5);
    this.sm = this.sm.map((v, j) => v + (this.mouse[j] - v) * km);

    // the ring: dealt far away, then opens around the viewer
    if (this.dropT !== null) {
      const k = easeInOut(clamp((t - this.dropT) / DROP_S, 0, 1));
      this.ring.p = DEALT.p.map((v, j) => lerp(v, OPEN.p[j], k));
      this.ring.tilt = lerp(DEALT.tilt, OPEN.tilt, k); this.ring.s = lerp(DEALT.s, OPEN.s, k);
    }
    this.spinSpeed += (this.spinTarget - this.spinSpeed) * (1 - Math.exp(-dt * 1.2));
    this.dragV *= Math.exp(-dt * 2.2);
    this.spin += dt * (this.spinSpeed + this.dragV);

    const aspect = this.canvas.width / this.canvas.height;
    const fov = (aspect < 1.45 ? 56 : 50) * Math.PI / 180;
    const proj = perspective(fov, aspect, 0.05, 80);
    const eye = [this.sm[0] * 0.25, this.sm[1] * -0.12, 0];
    const view = lookAt(eye, [this.sm[0] * 0.9, -0.12 - this.sm[1] * 0.35, -10]);
    const ringM = mul(T(...this.ring.p), mul(RX(this.ring.tilt), S(this.ring.s, this.ring.s, this.ring.s)));

    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);

    // gold dust
    const D = this.dp; gl.useProgram(D.p); gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dust); gl.enableVertexAttribArray(D.a.aSeed); gl.vertexAttribPointer(D.a.aSeed, 4, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(D.u.uProj, false, proj); gl.uniformMatrix4fv(D.u.uView, false, view);
    gl.uniform1f(D.u.uTime, t); gl.uniform1f(D.u.uPR, this.dpr);
    gl.drawArrays(gl.POINTS, 0, 420); gl.disableVertexAttribArray(D.a.aSeed);

    // the prints: place each one, then draw far to near
    const since = this.started === null ? -1 : t - this.started;
    const items = [];
    for (const c of this.cards) {
      const a = c.i / N * Math.PI * 2 + this.spin;
      const front = Math.cos(wrap(a));                       // 1 = right in front of the viewer
      // dealing: each print flies in from the near edge of the ring and flips face up as it lands
      const dk = since < 0 ? 0 : clamp((since - 0.15 - c.i * 0.075) / 0.7, 0, 1);
      c.deal = dk;
      if (dk <= 0) continue;
      const e = easeOutBack(dk);
      // a new photo enters while out of sight (behind the viewer)
      if (front < -0.6 && !c.swapped) {
        if (this.selected && this.selected.length) { this.load(c, this.selected[this.selIndex++ % this.selected.length]); c.swapped = true; }
        else if (this.queue.length && this.mode === 'working') { this.load(c, this.queue.shift()); c.swapped = true; }
      }
      if (front > 0.2) c.swapped = false;
      const rx = R * Math.sin(a), rz = -R * Math.cos(a);
      const deckX = 0, deckY = 2.6, deckZ = R * 1.6;            // the deal starts above the near edge
      const px = lerp(deckX, rx, e), py = lerp(deckY, 0, e) + Math.sin(dk * Math.PI) * 0.9, pz = lerp(deckZ, rz, e);
      const flip = (1 - e) * Math.PI;
      // light: the print passing the front glows while working; all glow when done
      const target = this.mode === 'working' ? Math.pow(Math.max(0, front), 18) * 1.2 :
        this.mode === 'done' ? 0.28 + 0.4 * Math.pow(Math.max(0, front), 6) : 0.04 + 0.1 * Math.pow(Math.max(0, front), 8);
      c.glow += (target - c.glow) * (1 - Math.exp(-dt * 6));
      c.lift += ((this.mode === 'working' ? Math.pow(Math.max(0, front), 18) * 0.18 : 0) - c.lift) * (1 - Math.exp(-dt * 6));
      const local = mul(T(px, py + c.lift, pz), mul(RY(-a + flip), RZ((1 - e) * 0.5)));
      const M = mul(ringM, local);
      const z = view[2] * M[12] + view[6] * M[13] + view[10] * M[14] + view[14];
      const reflM = mul(ringM, mul(T(px, 2 * FLOOR - py - c.lift, pz), mul(RY(-a + flip), S(1, -1, 1))));
      items.push({ c, M, reflM, z, dk, front });
    }
    items.sort((p, q) => p.z - q.z);
    const C = this.cp; gl.useProgram(C.p);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad); gl.enableVertexAttribArray(C.a.aCorner); gl.vertexAttribPointer(C.a.aCorner, 2, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(C.u.uProj, false, proj); gl.uniformMatrix4fv(C.u.uView, false, view);
    gl.uniform3f(C.u.uGold, 0.95, 0.74, 0.42); gl.uniform1i(C.u.uTex, 0); gl.activeTexture(gl.TEXTURE0);
    gl.uniform2f(C.u.uSize, W, H);
    const open = this.dropT === null ? 0 : clamp((t - this.dropT) / DROP_S, 0, 1);
    const draw = (it, reflect) => {
      const c = it.c, asp = c.tex.aspect, fa = W / H;                    // cover-crop the photo into the print
      gl.uniform2f(C.u.uUvScale, asp > fa ? fa / asp : 1, asp > fa ? 1 : asp / fa);
      gl.uniformMatrix4fv(C.u.uModel, false, reflect ? it.reflM : it.M);
      const depth = clamp((it.z + 22) / 18, 0.35, 1);
      gl.uniform1f(C.u.uOpacity, Math.min(1, it.dk * 1.6));
      gl.uniform1f(C.u.uBright, (0.55 + 0.45 * depth) * (0.75 + 0.25 * Math.max(0, it.front)));
      gl.uniform1f(C.u.uGlow, reflect ? 0 : c.glow);
      gl.uniform1f(C.u.uReflect, reflect ? 1 : 0);
      gl.bindTexture(gl.TEXTURE_2D, c.tex.t);
      if (!reflect && c.glow > 0.03) {
        gl.blendFunc(gl.ONE, gl.ONE); gl.uniform1f(C.u.uPass, 0); gl.uniform1f(C.u.uPad, 0.55);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.uniform1f(C.u.uPass, 1); gl.uniform1f(C.u.uPad, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };
    if (open > 0.6) for (const it of items) draw(it, true);       // reflections on the floor, once inside
    for (const it of items) draw(it, false);
    gl.disableVertexAttribArray(C.a.aCorner);
    requestAnimationFrame(n => this.frame(n));
  }
}
