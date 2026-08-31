// Camera discovery, lifecycle, header status pills. Tabs render their own camera
// cards from `state`; this module owns the model and the global status poll.
import { api, log, state, CAMERA_CONFIG } from './core.js';

let recording = false;             // set by the Record tab during a record job
let statusPoll = null;

// ── model + events ─────────────────────────────────────────────────────────

function emit() {
  document.dispatchEvent(new CustomEvent('cameras:changed'));
}

export function deviceList() {
  return Object.entries(state).map(([id, s]) => ({ id, type: s.type }));
}

export async function scanDevices() {
  log('Scanning for devices...');
  try {
    const data = await api('GET', '/api/cameras/devices');
    log(`Found: ${data.realsense.length} RealSense, ${data.kinect.length} Kinect`, 'ok');
    const seen = new Set();
    for (const [type, ids] of [['realsense', data.realsense], ['kinect', data.kinect]]) {
      for (const id of ids) {
        seen.add(id);
        state[id] = { running: (data.active || []).includes(id), type, fresh: false,
                      ...(state[id] || {}) };
        state[id].type = type;
      }
    }
    for (const id of Object.keys(state)) if (!seen.has(id)) delete state[id];
  } catch (e) {
    log('Scan failed: ' + e, 'error');
  }
  renderPills();
  emit();
}

export async function startCamera(id, cfg = {}) {
  const c = {
    color_width: cfg.color_width, color_height: cfg.color_height,
    fps: cfg.fps, depth_mode: cfg.depth_mode,
  };
  log(`Starting ${id}…`);
  try {
    const res = await api('POST', `/api/cameras/${id}/start`, c);
    if (res.detail) { log(`${id} error: ${res.detail}`, 'error'); return; }
    const msg = { restarted: `${id} restarted with new settings`,
                  unchanged: `${id} already running with these settings` }[res.status]
                || `${id} started`;
    log(msg, 'ok');
    if (state[id]) { state[id].running = true; state[id].fresh = false; state[id].cfg = null; }
  } catch (e) {
    log(`Failed to start ${id}: ${e}`, 'error');
  }
  renderPills();
  emit();
}

export async function stopCamera(id) {
  log(`Stopping ${id}...`);
  try { await api('POST', `/api/cameras/${id}/stop`); log(`${id} stopped`); }
  catch (e) { log(`Failed to stop ${id}: ${e}`, 'error'); }
  if (state[id]) { state[id].running = false; state[id].fresh = false; }
  renderPills();
  emit();
}

export async function startAll() {
  for (const id of Object.keys(state)) if (!state[id].running) await startCamera(id);
}

export async function stopAll() {
  for (const id of Object.keys(state)) if (state[id].running) await stopCamera(id);
}

export function setRecording(on) {
  recording = !!on;
  renderPills();
}

// ── per-tab camera card helpers ───────────────────────────────────────────

// Shared card markup used by the Record and Calibration tabs. The tab points
// the COLOR / DEPTH <img>s at whatever stream it wants (via setStream).
// Pass {depth:false} to omit the DEPTH panel (calibration doesn't need it, and
// each held MJPEG stream costs a browser connection slot).
export function cameraCardHTML(id, type, running, opts = {}) {
  const depthPanel = opts.depth === false ? '' : `
      <div class="stream-divider"></div>
      <div class="stream-panel"><img data-role="depth" data-id="${id}"><span class="stream-label">DEPTH</span></div>`;
  return `
  <div class="cam-card${opts.depth === false ? ' no-depth' : ''}" data-id="${id}">
    <div class="card-header">
      <span class="dot ${running ? 'green' : 'grey'}" data-role="dot" data-id="${id}"></span>
      <span class="name">${id}</span>
      <span class="tag ${type}">${type}</span>
      <button data-role="start" data-id="${id}" ${running ? 'disabled' : ''}>▶ Start</button>
      <button data-role="stop" data-id="${id}" class="danger" ${running ? '' : 'disabled'}>■ Stop</button>
      <span class="cam-warn" data-role="warn" data-id="${id}" hidden></span>
    </div>
    <div class="card-controls">${controlsHTML(id, type)}</div>
    <div class="streams">
      <div class="stream-panel"><img data-role="color" data-id="${id}"><span class="stream-label">COLOR</span></div>${depthPanel}
    </div>
  </div>`;
}

export function controlsHTML(id, type) {
  const cfg = CAMERA_CONFIG[type] || CAMERA_CONFIG.realsense;
  const resOpts = (cfg.resolutions || []).map(r =>
    `<option value="${r}" ${r === cfg.defaultRes ? 'selected' : ''}>${r.replace('x', ' × ')}</option>`).join('');
  const fpsOpts = (cfg.fps || []).map(f =>
    `<option value="${f}" ${f === cfg.defaultFps ? 'selected' : ''}>${f} fps</option>`).join('');
  let depthPart = '';
  if (cfg.depthModes) {
    const dOpts = cfg.depthModes.map(m =>
      `<option value="${m}" ${m === cfg.defaultDepth ? 'selected' : ''}>${m}</option>`).join('');
    depthPart = `<div class="sep"></div><label>depth</label><select data-role="depthmode" data-id="${id}">${dOpts}</select>`;
  }
  return `<label>res</label><select data-role="res" data-id="${id}">${resOpts}</select>
    <label>fps</label><select data-role="fps" data-id="${id}">${fpsOpts}</select>${depthPart}`;
}

export function readCardConfig(root, id) {
  const q = role => root.querySelector(`[data-role="${role}"][data-id="${id}"]`);
  const resVal = q('res')?.value || '1280x720';
  const [w, h] = resVal.split('x').map(Number);
  return {
    color_width: w, color_height: h,
    fps: parseInt(q('fps')?.value || '30', 10),
    depth_mode: q('depthmode')?.value || 'NFOV_UNBINNED',
  };
}

export function updateFpsForDepthMode(root, id) {
  const cfg = CAMERA_CONFIG[state[id]?.type] || CAMERA_CONFIG.realsense;
  if (!cfg.depthModeMaxFps) return;
  const depthMode = root.querySelector(`[data-role="depthmode"][data-id="${id}"]`)?.value;
  const maxFps = cfg.depthModeMaxFps[depthMode] || 30;
  const fpsSelect = root.querySelector(`[data-role="fps"][data-id="${id}"]`);
  if (!fpsSelect) return;
  for (const opt of fpsSelect.options) {
    opt.disabled = parseInt(opt.value, 10) > maxFps;
    if (opt.disabled && fpsSelect.value === opt.value) fpsSelect.value = String(maxFps);
  }
}

// Point an <img> at an MJPEG endpoint (cache-busted). Pass null to detach.
export function setStream(imgEl, url) {
  if (!imgEl) return;
  imgEl.src = url ? `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}` : '';
}

export async function fetchInfo(id) {
  try {
    const info = await api('GET', `/api/cameras/${id}/info`);
    if (state[id]) { state[id].fresh = !!info; state[id].cfg = info?.config || null; }
    return info;
  } catch {
    if (state[id]) state[id].fresh = false;
    return null;
  }
}

// True if the card's selected settings differ from what the camera is actually
// running with (i.e. Start would restart it). null when not comparable yet.
export function configMismatch(root, id) {
  const running = state[id]?.cfg;
  if (!state[id]?.running || !running) return null;
  const want = readCardConfig(root, id);
  const keys = state[id].type === 'kinect'
    ? ['color_width', 'color_height', 'fps', 'depth_mode'] : ['color_width', 'color_height', 'fps'];
  const diff = keys.filter(k => String(running[k]) !== String(want[k]));
  return diff.length ? { running, want, diff } : null;
}

// ── header status pills + ViKi dot ────────────────────────────────────────

function pillClass(id) {
  const s = state[id];
  if (!s || !s.running) return 'grey';
  if (recording) return 'red blink';
  return s.fresh ? 'green blink' : 'grey';
}

export function renderPills() {
  const box = document.getElementById('camera-pills');
  if (box) {
    const ids = Object.keys(state);
    box.innerHTML = ids.length
      ? ids.map(id => `<span class="cam-pill"><span class="dot ${pillClass(id)}"></span>${id}</span>`).join('')
      : '<span class="hint">no cameras</span>';
  }
  const dot = document.getElementById('server-dot');
  if (dot) {
    const anyRunning = Object.values(state).some(s => s.running);
    dot.className = 'dot ' + (recording ? 'red blink' : anyRunning ? 'green blink' : 'grey');
  }
}

export function startStatusPoll() {
  if (statusPoll) return;
  const tick = async () => {
    await Promise.all(Object.keys(state)
      .filter(id => state[id].running)
      .map(id => fetchInfo(id)));
    renderPills();
  };
  tick();
  statusPoll = setInterval(tick, 1500);
}

export function stopStatusPoll() {
  if (statusPoll) { clearInterval(statusPoll); statusPoll = null; }
}
