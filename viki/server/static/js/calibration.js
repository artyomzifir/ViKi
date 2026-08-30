// Calibration tab: extrinsics only (camera -> world via a ChArUco anchor).
// Intrinsics are trusted from the camera SDK, so there is no intrinsics UI.
// Left: one card per camera with the board-overlay stream. Right: preset picker,
// board params, session / capture / solve controls.
import { api, log, state, FRONTEND_CONFIG } from './core.js';
import * as cameras from './cameras.js';

const ARUCO_DICTS = [
  'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
  'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
  'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
  'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
  'DICT_ARUCO_ORIGINAL',
];

let view = null;
let countPoll = null;
let onCamerasChanged = null;

// ── template ──────────────────────────────────────────────────────────────

function template() {
  const c = FRONTEND_CONFIG.calibration || { chess: {}, aruco: {} };
  const arucoOpts = ARUCO_DICTS.map(n =>
    `<option ${n === c.aruco.defaultDict ? 'selected' : ''}>${n}</option>`).join('');
  return `
  <div class="calib-tab">
    <div class="calib-cards" id="calib-cards"></div>
    <aside class="calib-side">
      <section class="calib-sec">
        <div class="calib-sec-title">Active preset</div>
        <select id="calib-preset"></select>
        <div class="hint" id="calib-preset-info">—</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Board</div>
        <div class="cfg-row"><label>Type</label>
          <select id="board-type">
            <option value="chess">Chessboard</option>
            <option value="aruco" selected>ChArUco</option>
          </select></div>
        <div class="cfg-row"><label>Cols</label>
          <input type="number" id="board-width" min="1" value="${c.aruco.boardSize?.[0] ?? 8}"></div>
        <div class="cfg-row"><label>Rows</label>
          <input type="number" id="board-height" min="1" value="${c.aruco.boardSize?.[1] ?? 10}"></div>
        <div class="cfg-row"><label>Square (m)</label>
          <input type="number" id="square-size" step="0.001" min="0.001" value="${c.aruco.squareSize ?? 0.05}"></div>
        <div id="aruco-fields">
          <div class="cfg-row"><label>Marker (m)</label>
            <input type="number" id="marker-size" step="0.001" min="0.001" value="${c.aruco.markerSize ?? 0.035}"></div>
          <div class="cfg-row"><label>Dictionary</label>
            <select id="aruco-dict">${arucoOpts}</select></div>
        </div>
      </section>

      <section class="calib-sec">
        <button id="calib-start-session" class="primary">Start session</button>
        <button id="calib-capture-all">Capture all</button>
        <button id="calib-solve" class="primary">Calibrate extrinsics</button>
        <button id="calib-clear" class="danger">Clear samples</button>
        <div class="hint">Extrinsic calibration needs ≥1 sample per camera, with the
          board visible to every camera at once.</div>
      </section>
    </aside>
  </div>`;
}

function cardHTML(id, type, running) {
  return `
  <div class="calib-card" data-id="${id}">
    <div class="card-header">
      <span class="dot ${running ? 'green' : 'grey'}"></span>
      <span class="name">${id}</span>
      <span class="tag ${type}">${type}</span>
      <span class="calib-count" data-id="${id}">no session</span>
      <button data-role="start" data-id="${id}" ${running ? 'disabled' : ''}>▶ Start</button>
      <button data-role="stop" data-id="${id}" class="danger" ${running ? '' : 'disabled'}>■ Stop</button>
      <button data-role="capture" data-id="${id}" class="primary" ${running ? '' : 'disabled'}>Capture</button>
    </div>
    <div class="calib-stream">
      <img data-id="${id}" alt="${id}">
      <span class="stream-label">${id}</span>
    </div>
  </div>`;
}

function renderCards() {
  const box = view?.querySelector('#calib-cards');
  if (!box) return;
  const ids = Object.keys(state);
  if (!ids.length || !ids.some(id => state[id].running)) {
    box.innerHTML = `<div class="empty-state"><h2>No live cameras</h2>
      <p>Start cameras (top bar) to calibrate.</p></div>`;
    // still show stopped cameras so you can start them
  }
  if (ids.length) {
    box.innerHTML = ids.map(id => cardHTML(id, state[id].type, state[id].running)).join('');
    for (const id of ids) {
      const img = box.querySelector(`img[data-id="${id}"]`);
      if (img) img.src = state[id].running ? `/api/calibration/${id}/stream?t=${Date.now()}` : '';
    }
  }
}

// ── board params ──────────────────────────────────────────────────────────

function boardType() { return view.querySelector('#board-type').value; }

function boardParams() {
  const p = {
    board_size: [
      parseInt(view.querySelector('#board-width').value, 10),
      parseInt(view.querySelector('#board-height').value, 10),
    ],
    square_size: parseFloat(view.querySelector('#square-size').value),
  };
  if (boardType() === 'aruco') {
    p.marker_size = parseFloat(view.querySelector('#marker-size').value);
    p.aruco_dict = view.querySelector('#aruco-dict').value;
  }
  return p;
}

function syncBoardFieldVisibility() {
  view.querySelector('#aruco-fields').style.display =
    boardType() === 'aruco' ? '' : 'none';
}

async function syncParams() {
  try {
    await api('POST', `/api/calibration/sync?board_type=${boardType()}`, boardParams());
  } catch (e) {
    log('Board param sync failed: ' + e, 'error');
  }
}

// ── session / capture / solve ─────────────────────────────────────────────

async function startSession() {
  await syncParams();
  const bt = boardType();
  const params = boardParams();
  const running = Object.keys(state).filter(id => state[id].running);
  if (!running.length) { log('Start cameras before the calibration session', 'error'); return; }
  log(`Starting calibration session (${bt})…`);
  for (const id of running) {
    const ep = bt === 'chess'
      ? `/api/calibration/start/${id}?mode=manual`
      : `/api/calibration/start/aruco/${id}?mode=manual`;
    try { await api('POST', ep, params); }
    catch (e) { log(`Session start failed for ${id}: ${e}`, 'error'); }
  }
  updateCounts();
}

async function captureAll() {
  try { await api('POST', '/api/calibration/capture'); log('Captured a sample on all cameras', 'ok'); }
  catch (e) { log('Capture failed: ' + e, 'error'); }
  updateCounts();
}

async function captureOne(id) {
  try { await api('POST', `/api/calibration/capture/${id}`); log(`Captured a sample on ${id}`, 'ok'); }
  catch (e) { log(`Capture failed for ${id}: ${e}`, 'error'); }
  updateCounts();
}

async function updateCounts() {
  await Promise.all(Object.keys(state).map(async id => {
    const el = view?.querySelector(`.calib-count[data-id="${id}"]`);
    if (!el) return;
    try {
      const s = await api('GET', `/api/calibration/status/${id}?t=${Date.now()}`);
      el.textContent = s.started ? `${s.samples_count} samples` : 'no session';
    } catch { el.textContent = 'error'; }
  }));
}

async function solve() {
  log('Calibrating extrinsics…');
  try {
    await api('POST', '/api/calibration/extrinsics');
    // Snapshot the (now board-free) scene as background depth for scene subtraction.
    await Promise.all(Object.keys(state).map(async id => {
      try { await api('POST', `/api/skeleton/capture_base/${id}`); }
      catch (e) { log(`Base depth capture failed for ${id}: ${e}`, 'error'); }
    }));
    log('Extrinsics solved', 'ok');
    const name = prompt('Save this calibration as a preset (name):', '');
    if (name && name.trim()) {
      await api('POST', '/api/calibration/save-as', { name: name.trim() });
      log(`Saved preset "${name.trim()}"`, 'ok');
    }
    refreshPresets();
  } catch (e) {
    log('Extrinsics calibration failed: ' + e, 'error');
  }
}

async function clearSamples() {
  for (const id of Object.keys(state)) {
    try { await api('POST', `/api/calibration/clear/${id}`); } catch { /* ignore */ }
  }
  log('Cleared calibration samples');
  updateCounts();
}

// ── presets ───────────────────────────────────────────────────────────────

async function refreshPresets() {
  const sel = view?.querySelector('#calib-preset');
  const info = view?.querySelector('#calib-preset-info');
  if (!sel) return;
  try {
    const presets = await api('GET', '/api/calibration/presets');
    sel.innerHTML = '<option value="">(current, unsaved)</option>' +
      presets.map(p => `<option value="${p.name}" ${p.active ? 'selected' : ''}>${p.name}</option>`).join('');
    const active = presets.find(p => p.active);
    info.textContent = active
      ? `${active.cameras.length} cameras · solved ${new Date(active.solved_at * 1000).toLocaleString()}`
      : `${presets.length} saved preset(s)`;
  } catch (e) {
    info.textContent = 'presets unavailable';
    log('Failed to load calibration presets: ' + e, 'error');
  }
}

async function activatePreset(name) {
  if (!name) return;
  try {
    await api('POST', '/api/calibration/activate', { name });
    log(`Activated calibration preset "${name}"`, 'ok');
  } catch (e) {
    log(`Failed to activate "${name}": ${e}`, 'error');
  }
  refreshPresets();
}

// ── mount / unmount ───────────────────────────────────────────────────────

export function mount(container) {
  view = container;
  view.innerHTML = template();
  syncBoardFieldVisibility();
  renderCards();
  refreshPresets();

  view.addEventListener('click', onClick);
  view.addEventListener('change', onChange);
  onCamerasChanged = () => renderCards();
  document.addEventListener('cameras:changed', onCamerasChanged);

  updateCounts();
  countPoll = setInterval(updateCounts, 1000);
}

export function unmount() {
  if (countPoll) { clearInterval(countPoll); countPoll = null; }
  if (onCamerasChanged) document.removeEventListener('cameras:changed', onCamerasChanged);
  onCamerasChanged = null;
  view?.querySelectorAll('.calib-stream img').forEach(img => { img.src = ''; });
  api('POST', '/api/calibration/reset').catch(() => { });
  view = null;
}

// ── events ────────────────────────────────────────────────────────────────

function onClick(e) {
  const btn = e.target.closest('button');
  if (!btn) return;
  const id = btn.dataset.id;
  switch (btn.id || btn.dataset.role) {
    case 'start': cameras.startCamera(id); break;
    case 'stop': cameras.stopCamera(id); break;
    case 'capture': captureOne(id); break;
    case 'calib-start-session': startSession(); break;
    case 'calib-capture-all': captureAll(); break;
    case 'calib-solve': solve(); break;
    case 'calib-clear': clearSamples(); break;
  }
}

function onChange(e) {
  const el = e.target;
  if (el.id === 'board-type') { syncBoardFieldVisibility(); syncParams(); }
  else if (['board-width', 'board-height', 'square-size', 'marker-size'].includes(el.id)
    || el.id === 'aruco-dict') { syncParams(); }
  else if (el.id === 'calib-preset') { activatePreset(el.value); }
}
