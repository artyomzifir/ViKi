// 3-D episode viewer: a rotatable canvas-2D projection of the wrist trajectory,
// palm frames, camera frusta and (optional) raw per-camera point clouds.
// No WebGL / vendored deps — line art projected by hand.
import { api, log } from './core.js';

let G = null;                 // last /geometry payload
let frame = 0, playing = false, showRaw = false;
let azim = -0.7, elev = 0.5, dist = 2.2;
const center = [0, 0, 0];
let canvas, ctx, slider, playBtn, rawChk, frameLbl;
let drag = null, raf = 0;

const CAM_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0'];

export function initViewer() {
  canvas = document.getElementById('viewer-canvas');
  if (!canvas || canvas.dataset.wired) return;
  canvas.dataset.wired = '1';
  ctx = canvas.getContext('2d');
  slider = document.getElementById('viewer-frame');
  playBtn = document.getElementById('viewer-play');
  rawChk = document.getElementById('viewer-raw');
  frameLbl = document.getElementById('viewer-frame-lbl');

  canvas.addEventListener('mousedown', e => (drag = { x: e.clientX, y: e.clientY }));
  window.addEventListener('mouseup', () => (drag = null));
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    azim += (e.clientX - drag.x) * 0.01;
    elev = Math.max(-1.5, Math.min(1.5, elev + (e.clientY - drag.y) * 0.01));
    drag = { x: e.clientX, y: e.clientY };
    draw();
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    dist = Math.max(0.3, Math.min(12, dist * (1 + Math.sign(e.deltaY) * 0.1)));
    draw();
  }, { passive: false });

  slider.addEventListener('input', () => { frame = +slider.value; updateLbl(); draw(); });
  playBtn.addEventListener('click', togglePlay);
  rawChk.addEventListener('change', () => { showRaw = rawChk.checked; loadRaw(); });
  resize();
  window.addEventListener('resize', () => { resize(); draw(); });
}

function resize() {
  if (!canvas) return;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, r.width);
  canvas.height = Math.max(240, r.height);
}

export async function loadEpisode(id) {
  try {
    G = await api('GET', `/api/pipeline/episode/${id}/geometry?include_raw=${showRaw ? 1 : 0}`);
  } catch (e) { log('viewer: ' + e, 'error'); return; }
  frame = 0;
  const n = G.n_frames || 0;
  slider.max = Math.max(0, n - 1);
  slider.value = 0;
  if (G.wrist_traj && G.wrist_traj.length) {
    const c = G.wrist_traj.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0]);
    center[0] = c[0] / n; center[1] = c[1] / n; center[2] = c[2] / n;
  }
  updateLbl();
  draw();
}

async function loadRaw() {
  if (!G) return;
  if (showRaw && !G.raw_points) await loadEpisode(G.id);
  else draw();
}

function togglePlay() {
  playing = !playing;
  playBtn.textContent = playing ? '❚❚' : '▶';
  if (playing) tick();
}
function tick() {
  if (!playing || !G) return;
  frame = (frame + 1) % Math.max(1, G.n_frames);
  slider.value = frame; updateLbl(); draw();
  raf = requestAnimationFrame(() => setTimeout(tick, 1000 / (G.fps || 15)));
}
function updateLbl() {
  if (frameLbl && G) frameLbl.textContent = `${frame + 1} / ${G.n_frames || 0}`;
}

// --- projection ------------------------------------------------------------
function project(p) {
  const x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
  const ca = Math.cos(azim), sa = Math.sin(azim);
  let X = ca * x + sa * z;
  let Z = -sa * x + ca * z;
  const ce = Math.cos(elev), se = Math.sin(elev);
  let Y = ce * y - se * Z;
  Z = se * y + ce * Z;
  const f = 1.8, d = Z + dist;
  if (d <= 0.01) return null;
  const s = Math.min(canvas.width, canvas.height) * 0.5;
  return [canvas.width / 2 + (f * X / d) * s, canvas.height / 2 - (f * Y / d) * s, d];
}
function line(a, b, col, w = 1) {
  const p = project(a), q = project(b);
  if (!p || !q) return;
  ctx.strokeStyle = col; ctx.lineWidth = w;
  ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
}
function dot(a, col, r = 2) {
  const p = project(a); if (!p) return;
  ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.fill();
}

function draw() {
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0b0d10'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  // world axes
  line([0, 0, 0], [0.2, 0, 0], '#e6194b', 2);
  line([0, 0, 0], [0, 0.2, 0], '#3cb44b', 2);
  line([0, 0, 0], [0, 0, 0.2], '#4363d8', 2);
  if (!G) { hud('load an episode'); return; }

  // camera frusta
  let ci = 0;
  for (const [dev, c] of Object.entries(G.cameras || {})) {
    const col = CAM_COLORS[ci++ % CAM_COLORS.length];
    const o = c.pos, fdir = c.forward;
    const tip = [o[0] + fdir[0] * 0.15, o[1] + fdir[1] * 0.15, o[2] + fdir[2] * 0.15];
    line(o, tip, col, 1.5);
    dot(o, col, 4);
  }

  // wrist trajectory
  const T = G.wrist_traj || [];
  for (let i = 1; i < T.length; i++) {
    line(T[i - 1], T[i], G.valid && !G.valid[i] ? '#555' : '#cfcfcf', 1.6);
  }
  // palm triad at current frame
  if (T[frame] && G.palm_rot && G.palm_rot[frame]) {
    const o = T[frame], R = G.palm_rot[frame], s = 0.05;
    line(o, [o[0] + R[0] * s, o[1] + R[3] * s, o[2] + R[6] * s], '#ff4d4d', 2.5);
    line(o, [o[0] + R[1] * s, o[1] + R[4] * s, o[2] + R[7] * s], '#4dff4d', 2.5);
    line(o, [o[0] + R[2] * s, o[1] + R[5] * s, o[2] + R[8] * s], '#4d7dff', 2.5);
    dot(o, '#ffcc00', 4);
  }

  // raw per-camera clouds
  if (showRaw && G.raw_points) {
    let ri = 0;
    for (const [dev, cloud] of Object.entries(G.raw_points)) {
      const col = CAM_COLORS[ri++ % CAM_COLORS.length] + '66';
      for (const p of cloud) dot(p, col, 1.5);
    }
  }
  hud(`azim ${azim.toFixed(2)}  elev ${elev.toFixed(2)}  drag to orbit · wheel to zoom`);
}
function hud(t) {
  ctx.fillStyle = '#7a8290'; ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(t, 10, canvas.height - 10);
}
