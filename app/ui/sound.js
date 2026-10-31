// KeepSake sound: a short wedding piece for the opening, made on the fly with the Web Audio API
// (no sound files, works offline). In Raag Yaman, on D, the way a shaadi begins:
//
//   start   a tanpura drone, and a santoor rising with the prints as they scatter up
//   launch  a soft whoosh as each print leaves the bottom (panned to where it starts)
//   land    a tiny ghungroo jingle as it lands face up in the ring
//   dealt   a temple bell, then the shehnai sings a short phrase
//   drop    the dholak comes in (keherwa) as the camera drops inside the ring
//   end     a last santoor flourish; the music fades away under the welcome screen
//   done    bell and santoor when the photos are chosen
//
// Sound can be turned off with the speaker button (remembered on this laptop).

const SA = 62;                                                   // Sa = D4
const YAMAN = { S: 0, R: 2, G: 4, M: 6, P: 7, D: 9, N: 11 };     // Ma tivra: Yaman
const hz = m => 440 * Math.pow(2, (m - 69) / 12);
const note = (sw, oct = 0) => hz(SA + YAMAN[sw] + 12 * oct);   // 'G', 1 -> Ga in the upper octave

export class Sound {
  constructor() {
    this.on = true;
    try { this.on = localStorage.getItem('ks-sound') !== 'off'; } catch (e) { /* private window: sound on */ }
    this.ctx = null;
  }
  setOn(on) {
    this.on = on;
    try { localStorage.setItem('ks-sound', on ? 'on' : 'off'); } catch (e) { /* not remembered */ }
    if (this.ctx) (on ? this.ctx.resume() : this.ctx.suspend()).catch(() => {});
  }
  audio() {
    if (!this.on) return null;
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      try { this.ctx = new AC(); } catch (e) { return null; }
      this.mix = mixer(this.ctx, this.ctx.destination);
      const wake = () => this.ctx.state === 'suspended' && this.ctx.resume().catch(() => {});
      addEventListener('pointerdown', wake); addEventListener('keydown', wake);   // a browser that waits for a click
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    return this.ctx;
  }
  play(kind, opts = {}) {
    const ctx = this.audio();
    if (ctx && VOICES[kind]) VOICES[kind](ctx, this.mix, ctx.currentTime + 0.02, opts);
  }
}

// The same sounds, rendered to a WAV file (for a demo video): events = [{kind, at (s), ...opts}]
export async function render(events, seconds, rate = 48000) {
  const ctx = new OfflineAudioContext(2, Math.ceil(seconds * rate), rate);
  const mix = mixer(ctx, ctx.destination);
  for (const e of [...events].sort((a, b) => a.at - b.at)) if (VOICES[e.kind]) VOICES[e.kind](ctx, mix, e.at, e);
  return wav(await ctx.startRendering());
}

// ---------- the mix: music in a warm hall, effects drier, a gentle limiter on top ----------
function mixer(ctx, dest) {
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -14; comp.ratio.value = 3.5; comp.attack.value = 0.005; comp.release.value = 0.3;
  const master = ctx.createGain(); master.gain.value = 0.95;
  comp.connect(master); master.connect(dest);
  const hall = ctx.createConvolver(); hall.buffer = impulse(ctx, 2.6, 2.2);
  const room = ctx.createConvolver(); room.buffer = impulse(ctx, 1.2, 3);
  const music = ctx.createGain(), sfx = ctx.createGain();
  music.gain.value = 0.85; sfx.gain.value = 0.55;
  const hallWet = ctx.createGain(), roomWet = ctx.createGain();
  hallWet.gain.value = 0.38; roomWet.gain.value = 0.18;
  music.connect(comp); music.connect(hall); hall.connect(hallWet); hallWet.connect(comp);
  sfx.connect(comp); sfx.connect(room); room.connect(roomWet); roomWet.connect(comp);
  return { music, sfx, piece: null };
}
function impulse(ctx, seconds, power) {
  const n = Math.floor(ctx.sampleRate * seconds), buf = ctx.createBuffer(2, n, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch), r = rng(7 + ch * 13);
    for (let i = 0; i < n; i++) d[i] = (r() * 2 - 1) * Math.pow(1 - i / n, power);
  }
  return buf;
}
function rng(seed) { return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }
const noises = new WeakMap();
function noise(ctx) {
  if (!noises.has(ctx)) {
    const n = ctx.sampleRate * 2, buf = ctx.createBuffer(1, n, ctx.sampleRate), d = buf.getChannelData(0), r = rng(99);
    for (let i = 0; i < n; i++) d[i] = r() * 2 - 1;
    noises.set(ctx, buf);
  }
  const src = ctx.createBufferSource(); src.buffer = noises.get(ctx); src.loop = true;
  return src;
}
function env(ctx, t, peak, attack, release) {            // a gain that rises, then dies away
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + attack + release);
  return g;
}
function pan(ctx, node, p, out) {
  if (ctx.createStereoPanner) { const s = ctx.createStereoPanner(); s.pan.value = p; node.connect(s); s.connect(out); }
  else node.connect(out);
}
function tone(ctx, out, t, freq, peak, attack, release, type = 'sine') {
  const o = ctx.createOscillator(); o.type = type; o.frequency.value = freq;
  const g = env(ctx, t, peak, attack, release);
  o.connect(g); g.connect(out); o.start(t); o.stop(t + attack + release + 0.05);
}

// ---------- instruments ----------
// santoor: a hammered string, by Karplus-Strong (two strings a few cents apart, as on the real one)
const strings = new WeakMap();
function santoorBuffer(ctx, freq) {
  if (!strings.has(ctx)) strings.set(ctx, new Map());
  const cache = strings.get(ctx), key = Math.round(freq * 10);
  if (cache.has(key)) return cache.get(key);
  const sr = ctx.sampleRate, n = Math.floor(sr * 2.2), buf = ctx.createBuffer(1, n, sr), d = buf.getChannelData(0);
  for (const [cents, w] of [[-4, 0.5], [4, 0.5]]) {
    const N = Math.max(2, Math.round(sr / (freq * Math.pow(2, cents / 1200))));
    const line = new Float32Array(N), r = rng(key + cents + 50);
    for (let i = 0; i < N; i++) line[i] = (r() * 2 - 1) * (i < N / 3 ? 1 : 0.4);       // a hard, bright hammer
    const loss = Math.pow(0.001, 1 / (2.0 * freq));                                    // about 2 s to die away
    for (let i = 0, k = 0; i < n; i++, k = (k + 1) % N) {
      const y = line[k];
      line[k] = loss * (0.62 * y + 0.38 * line[(k + 1) % N]);                         // keeps the shimmer
      d[i] += w * y;
    }
  }
  let peak = 0; for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(d[i]));
  for (let i = 0; i < n; i++) d[i] /= peak || 1;
  cache.set(key, buf);
  return buf;
}
function santoor(ctx, out, t, freq, vol = 0.22, p = 0, tremolo = true) {
  const strike = (at, v) => {
    const src = ctx.createBufferSource(); src.buffer = santoorBuffer(ctx, freq);
    const g = ctx.createGain(); g.gain.value = v;
    src.connect(g); pan(ctx, g, p, out); src.start(at);
  };
  strike(t, vol);
  if (tremolo) strike(t + 0.075, vol * 0.55);                                         // the second mallet
}
// shehnai: a reedy double reed with meend (slides between notes) and a widening vibrato
function shehnai(ctx, out, t, phrase, vol = 0.16) {
  const o = ctx.createOscillator(), o2 = ctx.createOscillator();
  o.type = 'sawtooth'; o2.type = 'square';
  const vib = ctx.createOscillator(), depth = ctx.createGain();
  vib.frequency.value = 5.4; depth.gain.setValueAtTime(0, t);
  let at = t, end = t, prev = null;
  for (const [sw, oct, dur, slide] of phrase) {                // [swara, octave, seconds, slide-in seconds]
    const f = note(sw, oct);
    for (const x of [o.frequency, o2.frequency]) {
      if (slide && prev) { x.setValueAtTime(prev, at); x.linearRampToValueAtTime(f, at + slide); }   // meend
      else x.setValueAtTime(f, at);
    }
    prev = f;
    depth.gain.setValueAtTime(f * 0.002, at); depth.gain.linearRampToValueAtTime(f * 0.009, at + dur);
    at += dur; end = at;
  }
  vib.connect(depth); depth.connect(o.frequency); depth.connect(o2.frequency);
  const shape = ctx.createWaveShaper(), curve = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) { const x = i / 511.5 - 1; curve[i] = Math.tanh(2.2 * x); }
  shape.curve = curve;
  const mixIn = ctx.createGain(); mixIn.gain.value = 0.5;
  const g2 = ctx.createGain(); g2.gain.value = 0.35;
  o.connect(mixIn); o2.connect(g2); g2.connect(mixIn); mixIn.connect(shape);
  const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 420;
  const f1 = ctx.createBiquadFilter(); f1.type = 'peaking'; f1.frequency.value = 1150; f1.Q.value = 1.6; f1.gain.value = 9;
  const f2 = ctx.createBiquadFilter(); f2.type = 'peaking'; f2.frequency.value = 2700; f2.Q.value = 2.5; f2.gain.value = 6;
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 5200;
  const amp = ctx.createGain();
  amp.gain.setValueAtTime(0.0001, t); amp.gain.exponentialRampToValueAtTime(vol, t + 0.12);
  let k = t;
  for (const [, , dur] of phrase.slice(0, -1)) { k += dur; amp.gain.setValueAtTime(vol, k - 0.05); amp.gain.linearRampToValueAtTime(vol * 0.62, k); amp.gain.linearRampToValueAtTime(vol, k + 0.07); }
  amp.gain.setValueAtTime(vol, end - 0.5); amp.gain.exponentialRampToValueAtTime(0.0001, end + 0.25);
  shape.connect(hp); hp.connect(f1); f1.connect(f2); f2.connect(lp); lp.connect(amp); pan(ctx, amp, 0.12, out);
  const br = noise(ctx), bf = ctx.createBiquadFilter(), bg = ctx.createGain();           // breath
  bf.type = 'bandpass'; bf.frequency.value = 3200; bf.Q.value = 0.8;
  bg.gain.setValueAtTime(0.0001, t); bg.gain.exponentialRampToValueAtTime(vol * 0.06, t + 0.1);
  bg.gain.setValueAtTime(vol * 0.06, end - 0.3); bg.gain.exponentialRampToValueAtTime(0.0001, end + 0.2);
  br.connect(bf); bf.connect(bg); bg.connect(out);
  for (const x of [o, o2, vib]) { x.start(t); x.stop(end + 0.4); }
  br.start(t, 0.7); br.stop(end + 0.3);
}
// temple bell (ghanta): bell partials, the low hum ringing longest
function bell(ctx, out, t, f0 = note('S', 1), vol = 0.5) {
  for (const [ratio, a, decay] of [[0.5, 0.5, 4.5], [1, 1, 3.6], [1.19, 0.45, 2.6], [1.5, 0.3, 2.2], [2.0, 0.55, 2.0],
    [2.51, 0.22, 1.4], [2.66, 0.18, 1.2], [3.01, 0.12, 1.0], [4.07, 0.07, 0.6]])
    tone(ctx, out, t, f0 * ratio, vol * a * 0.18, 0.003, decay);
  const s = noise(ctx), f = ctx.createBiquadFilter(), g = env(ctx, t, vol * 0.08, 0.001, 0.035);
  f.type = 'bandpass'; f.frequency.value = 3000; f.Q.value = 1;
  s.connect(f); f.connect(g); g.connect(out); s.start(t, 0.2); s.stop(t + 0.08);
}
// dholak: the low head (dha, ge, dhi) and the high head (na, ti), and a muted slap (ka)
function dholak(ctx, out, t, stroke, vol = 0.5) {
  const low = (v) => {
    const o = ctx.createOscillator(); o.frequency.setValueAtTime(118, t); o.frequency.exponentialRampToValueAtTime(68, t + 0.16);
    const g = env(ctx, t, vol * v, 0.003, 0.34); o.connect(g); pan(ctx, g, -0.15, out); o.start(t); o.stop(t + 0.4);
  };
  const high = (v, f = 2300) => {
    const s = noise(ctx), bp = ctx.createBiquadFilter(), g = env(ctx, t, vol * v * 0.5, 0.001, 0.07);
    bp.type = 'bandpass'; bp.frequency.value = f; bp.Q.value = 2.5;
    s.connect(bp); bp.connect(g); pan(ctx, g, 0.2, out); s.start(t, (f % 7) * 0.1); s.stop(t + 0.1);
    tone(ctx, out, t, f / 4.4, vol * v * 0.22, 0.002, 0.11);
  };
  const slap = () => {
    const s = noise(ctx), lp = ctx.createBiquadFilter(), g = env(ctx, t, vol * 0.25, 0.001, 0.035);
    lp.type = 'lowpass'; lp.frequency.value = 1100; s.connect(lp); lp.connect(g); g.connect(out); s.start(t, 0.4); s.stop(t + 0.06);
  };
  ({ dha: () => { low(1); high(0.8); }, ge: () => low(0.75), dhi: () => { low(0.9); high(0.6, 2600); },
     na: () => high(1), ti: () => high(0.6, 2900), ka: slap })[stroke]();
}
const KEHERWA = ['dha', 'ge', 'na', 'ti', 'na', 'ka', 'dhi', 'na'];
// ghungroo: a little cluster of ankle bells
function ghungroo(ctx, out, t, seed, vol = 0.05) {
  const r = rng(seed * 31 + 7);
  for (let k = 0; k < 3; k++) tone(ctx, out, t + r() * 0.025, 4800 + r() * 4200, vol, 0.001, 0.06 + r() * 0.08);
}
// tanpura: Pa, Sa, Sa, low Sa, plucked again and again under everything
function tanpura(ctx, out, t, until) {
  const cycle = [note('P', -1), note('S'), note('S'), note('S', -1)], step = 0.62;
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 1500; lp.Q.value = 2;
  const g = ctx.createGain(); g.gain.value = 0.2; lp.connect(g); g.connect(out);
  for (let at = t, k = 0; at < until; at += step, k++) {
    const f = cycle[k % 4];
    for (const cents of [-3, 3]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = f * Math.pow(2, cents / 1200);
      const e = env(ctx, at, 0.5, 0.03, 2.4); o.connect(e); e.connect(lp); o.start(at); o.stop(at + 2.5);
    }
  }
}

// ---------- the piece ----------
const VOICES = {
  start(ctx, mix, t) {
    const piece = ctx.createGain(); piece.gain.value = 1; piece.connect(mix.music); mix.piece = piece;
    piece.gain.setValueAtTime(1, t + 20); piece.gain.linearRampToValueAtTime(0.0001, t + 24);   // never longer than this
    tanpura(ctx, piece, t, t + 24);
    // the santoor rises with the prints: Ni Re Ga Ma Pa Ma Ga Re Ga Ma Dha Ni Sa' Re' Ga' Re'
    const run = [['N', -1], ['R', 0], ['G', 0], ['M', 0], ['P', 0], ['M', 0], ['G', 0], ['R', 0],
      ['G', 0], ['M', 0], ['D', 0], ['N', 0], ['S', 1], ['R', 1], ['G', 1], ['R', 1]];
    run.forEach(([sw, oct], k) => santoor(ctx, piece, t + 0.35 + k * 0.19, note(sw, oct), 0.3 + k * 0.008, (k % 4 - 1.5) * 0.15));
  },
  launch(ctx, mix, t, { pan: p = 0, seed = 0 }) {
    const src = noise(ctx), bp = ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.Q.value = 1.4;
    const lift = 1 + ((seed * 37) % 10) / 40;
    bp.frequency.setValueAtTime(420 * lift, t); bp.frequency.exponentialRampToValueAtTime(2600 * lift, t + 0.32);
    const g = env(ctx, t, 0.22, 0.07, 0.33);
    src.connect(bp); bp.connect(g); pan(ctx, g, p * 0.7, mix.sfx);
    src.start(t, (seed * 0.137) % 1.5); src.stop(t + 0.5);
  },
  land(ctx, mix, t, { pan: p = 0, seed = 0 }) {
    const g = ctx.createGain(); g.gain.value = 1; pan(ctx, g, p * 0.5, mix.sfx);
    ghungroo(ctx, g, t, seed);
  },
  dealt(ctx, mix, t) {
    const out = mix.piece || mix.music;
    bell(ctx, out, t, note('S', 1), 0.55);
    santoor(ctx, out, t, note('S', 1), 0.28, 0, false);
    // the shehnai: Ni Re' Ga' (sliding) Re' Sa'
    shehnai(ctx, out, t + 0.3, [['N', 0, 0.38], ['R', 1, 0.34, 0.06], ['G', 1, 0.72, 0.16], ['R', 1, 0.3, 0.08], ['S', 1, 1.3, 0.12]]);
  },
  drop(ctx, mix, t) {
    const out = mix.piece || mix.music, matra = 0.165;
    for (let k = 0; k < 32; k++) dholak(ctx, out, t + 0.25 + k * matra, KEHERWA[k % 8], k < 8 ? 0.35 + k * 0.03 : 0.6);
    // a low swell as the ring opens around the viewer
    const src = noise(ctx), lp = ctx.createBiquadFilter(), g = ctx.createGain();
    lp.type = 'lowpass'; lp.frequency.setValueAtTime(1800, t); lp.frequency.exponentialRampToValueAtTime(300, t + 2.4);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.09, t + 0.9); g.gain.exponentialRampToValueAtTime(0.0001, t + 2.6);
    src.connect(lp); lp.connect(g); g.connect(mix.sfx); src.start(t, 0.3); src.stop(t + 2.7);
  },
  end(ctx, mix, t) {
    const out = mix.piece || mix.music;
    [['S', 1], ['N', 0], ['D', 0], ['P', 0], ['M', 0], ['G', 0], ['R', 0], ['S', 0]].forEach(([sw, oct], k) =>
      santoor(ctx, out, t + k * 0.11, note(sw, oct), 0.2, (0.5 - k / 8) * 0.4, k === 7));
    santoor(ctx, out, t + 0.95, note('P', -1), 0.16, 0, false);
    if (mix.piece) {
      const g = mix.piece.gain;
      g.cancelScheduledValues(t); g.setValueAtTime(1, t + 0.9); g.linearRampToValueAtTime(0.0001, t + 4.2);
      mix.piece = null;
    }
  },
  done(ctx, mix, t) {
    bell(ctx, mix.music, t, note('S', 1), 0.45);
    [['S', 0], ['G', 0], ['P', 0], ['N', 0], ['S', 1]].forEach(([sw, oct], k) => santoor(ctx, mix.music, t + 0.15 + k * 0.12, note(sw, oct), 0.2));
  },
};

// ---------- WAV (for render) ----------
function wav(buf) {
  const ch = buf.numberOfChannels, n = buf.length, rate = buf.sampleRate;
  const out = new DataView(new ArrayBuffer(44 + n * ch * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); out.setUint32(4, 36 + n * ch * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, ch, true); out.setUint32(24, rate, true);
  out.setUint32(28, rate * ch * 2, true); out.setUint16(32, ch * 2, true); out.setUint16(34, 16, true);
  str(36, 'data'); out.setUint32(40, n * ch * 2, true);
  const data = Array.from({ length: ch }, (_, c) => buf.getChannelData(c));
  for (let i = 0, o = 44; i < n; i++) for (let c = 0; c < ch; c++, o += 2)
    out.setInt16(o, Math.max(-1, Math.min(1, data[c][i])) * 0x7fff, true);
  return new Uint8Array(out.buffer);
}
