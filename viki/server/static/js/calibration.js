// Calibration tab: extrinsics only (camera -> world via a ChArUco anchor).
// Three zones: camera cards (same widget as Record) | captured-sets list |
// preset picker + board params. Captures are rig-wide only ("Capture all"), so
// set N is index-aligned across cameras and can be deleted as a unit. A saved
// preset carries its sets, so it can be reopened, pruned, and re-solved.
import { api, log, state, FRONTEND_CONFIG, sessionGet, sessionSet } from './core.js';
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
let builtIds = '';
let openedPreset = null;   // name of the preset whose sets are shown, or null (live)

// ── template ──────────────────────────────────────────────────────────────

function template() {
  const c = FRONTEND_CONFIG.calibration || { chess: {}, aruco: {} };
  // session board params win over the config defaults (survive tab switches)
  const s = sessionGet('calibBoard', null);
  const b = s || {
    type: 'aruco',
    cols: c.aruco.boardSize?.[0] ?? 8, rows: c.aruco.boardSize?.[1] ?? 10,
    square: c.aruco.squareSize ?? 0.05, marker: c.aruco.markerSize ?? 0.035,
    dict: c.aruco.defaultDict,
  };
  const arucoOpts = ARUCO_DICTS.map(n =>
    `<option ${n === b.dict ? 'selected' : ''}>${n}</option>`).join('');
  return `
  <div class="calib-tab">
    <div class="calib-cards" id="calib-cards"></div>

    <div class="calib-sets">
      <div class="calib-sets-head">
        <span id="calib-sets-title">Captured sets</span>
        <button id="calib-sets-live" hidden>← live</button>
      </div>
      <div id="calib-sets-list"></div>
    </div>

    <aside class="calib-side">
      <section class="calib-sec">
        <details ${s ? '' : 'open'}>
          <summary class="calib-sec-title">Board &nbsp;<span class="hint">${b.type === 'chess' ? 'chess' : 'ChArUco'} ${b.cols}×${b.rows} · ${b.square} m</span></summary>
          <div class="hint">Match the printed board. Lay it <b>flat on the table,
            face up</b>, where the work happens — it defines the world origin.</div>
          <div class="cfg-row"><label>Type</label>
            <select id="board-type">
              <option value="chess" ${b.type === 'chess' ? 'selected' : ''}>Chessboard</option>
              <option value="aruco" ${b.type !== 'chess' ? 'selected' : ''}>ChArUco</option>
            </select></div>
          <div class="cfg-row"><label>Cols</label>
            <input type="number" id="board-width" min="1" value="${b.cols}"></div>
          <div class="cfg-row"><label>Rows</label>
            <input type="number" id="board-height" min="1" value="${b.rows}"></div>
          <div class="cfg-row"><label>Square (m)</label>
            <input type="number" id="square-size" step="0.001" min="0.001" value="${b.square}"></div>
          <div id="aruco-fields">
            <div class="cfg-row"><label>Marker (m)</label>
              <input type="number" id="marker-size" step="0.001" min="0.001" value="${b.marker}"></div>
            <div class="cfg-row"><label>Dictionary</label>
              <select id="aruco-dict">${arucoOpts}</select></div>
          </div>
        </details>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Calibrate</div>
        <div class="hint">Start every camera (top bar) so each one sees the board.</div>
        <button id="calib-start-session" class="primary">1 · Start session</button>
        <button id="calib-capture-all" class="primary">2 · Capture all <span class="hint">(3–10×, board static)</span></button>
        <button id="calib-solve" class="primary">3 · Calibrate extrinsics</button>
        <div class="inline-add">
          <input type="text" id="calib-preset-name" placeholder="preset name">
          <button id="calib-preset-save" class="primary">4 · Save</button>
        </div>
        <div class="hint">Save also grabs the Kinects' depth↔colour calibration.
          Delete bad sets on the left before step 3.</div>
        <button id="calib-clear" class="danger">Clear samples</button>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Load a saved preset</div>
        <select id="calib-preset"></select>
        <div class="hint" id="calib-preset-info">—</div>
        <button id="calib-preset-open" hidden>Open sets</button>
        <button id="calib-preset-k4a" hidden title="attach the running Kinects' raw depth↔colour calibration to this preset">Grab k4a calibration for this preset</button>
      </section>
    </aside>
  </div>`;
}

// ── camera cards (shared widget, in-place sync) ──────────────────────────

function renderCards() {
  const box = view?.querySelector('#calib-cards');
  if (!box) return;
  const ids = Object.keys(state);
  const key = ids.join(',');
  if (key !== builtIds) {
    builtIds = key;
    box.innerHTML = ids.length
      ? ids.map(id => cameras.cameraCardHTML(id, state[id].type, state[id].running, { depth: false })).join('')
      : `<div class="empty-state"><h2>No cameras</h2><p>Scan for devices (top bar).</p></div>`;
  }
  for (const id of ids) {
    const on = !!state[id]?.running;
    const card = box.querySelector(`.cam-card[data-id="${id}"]`);
    if (!card) continue;
    card.querySelector('[data-role="dot"]').className = 'dot ' + (on ? 'green' : 'grey');
    card.querySelector('[data-role="start"]').disabled = on;
    card.querySelector('[data-role="stop"]').disabled = !on;
    const col = card.querySelector('img[data-role="color"]');
    const dep = card.querySelector('img[data-role="depth"]');
    const want = on ? '1' : '';
    if (col && col.dataset.on !== want) {
      cameras.setStream(col, on ? `/api/calibration/${id}/stream` : null); col.dataset.on = want;
    }
    if (dep && dep.dataset.on !== want) {
      cameras.setStream(dep, on ? `/api/cameras/${id}/depth` : null); dep.dataset.on = want;
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
  view.querySelector('#aruco-fields').style.display = boardType() === 'aruco' ? '' : 'none';
}

function persistBoard() {
  const p = boardParams();
  sessionSet('calibBoard', {
    type: boardType(),
    cols: p.board_size[0], rows: p.board_size[1], square: p.square_size,
    marker: p.marker_size ?? 0.035, dict: p.aruco_dict ?? view.querySelector('#aruco-dict').value,
  });
}

async function syncParams() {
  persistBoard();
  try { await api('POST', `/api/calibration/sync?board_type=${boardType()}`, boardParams()); }
  catch (e) { log('Board param sync failed: ' + e, 'error'); }
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
  refreshSets();
}

async function captureAll() {
  try {
    const r = await api('POST', '/api/calibration/capture');
    if (r.captured) log(`Captured set #${r.index}`, 'ok');
    else log(`Board not seen by: ${(r.missing || []).join(', ') || 'a camera'} — set discarded`, 'error');
  } catch (e) { log('Capture failed: ' + e, 'error'); }
  refreshSets();
}

async function solve() {
  log('Calibrating extrinsics…');
  try {
    await api('POST', '/api/calibration/extrinsics');
    await Promise.all(Object.keys(state).map(async id => {
      try { await api('POST', `/api/skeleton/capture_base/${id}`); }
      catch (e) { log(`Base depth capture failed for ${id}: ${e}`, 'error'); }
    }));
    log('Extrinsics solved — name it below and Save to keep it as a preset', 'ok');
    view?.querySelector('#calib-preset-name')?.focus();
    refreshPresets();
  } catch (e) {
    log('Extrinsics calibration failed: ' + e, 'error');
  }
}

async function clearSamples() {
  try { await api('POST', '/api/calibration/clear'); log('Cleared calibration samples'); }
  catch (e) { log('Clear failed: ' + e, 'error'); }
  refreshSets();
}

// ── captured-sets list (live or from an opened preset) ───────────────────

function setRow(row, canDelete) {
  const cams = Object.entries(row.cameras).map(([d, c]) => `
    <div class="set-cam-box">
      <div class="set-cam ${c.detected ? 'ok' : ''}">${d} ${c.detected ? c.corners : '✗'}</div>
      ${c.image ? `<img class="set-thumb" data-role="set-thumb" src="${c.image}" alt="${d}" loading="lazy">` : ''}
    </div>`).join('');
  const del = canDelete
    ? `<button data-role="set-del" data-i="${row.index}" class="danger">del</button>` : '';
  return `<div class="set-row"><div class="set-row-head"><span class="set-n">#${row.index}</span>${del}</div>${cams}</div>`;
}

let _setsSig = '';   // only re-render the list (and reload thumbs) when it changes

async function refreshSets() {
  const box = view?.querySelector('#calib-sets-list');
  if (!box) return;
  if (openedPreset) { renderPresetSets(); return; }
  view.querySelector('#calib-sets-title').textContent = 'Captured sets';
  view.querySelector('#calib-sets-live').hidden = true;
  try {
    const rows = await api('GET', '/api/calibration/samples');
    const sig = JSON.stringify(rows.map(r => [r.index, Object.values(r.cameras).map(c => c.corners)]));
    if (sig === _setsSig) return;
    _setsSig = sig;
    box.innerHTML = rows.length ? rows.map(r => setRow(r, true)).join('')
      : '<div class="hint" style="padding:8px">no sets yet — Start session, then Capture all</div>';
  } catch (e) { box.innerHTML = `<div class="hint" style="padding:8px">${e}</div>`; }
}

let presetDetail = null;

function renderPresetSets() {
  const box = view.querySelector('#calib-sets-list');
  view.querySelector('#calib-sets-title').textContent = `Preset "${openedPreset}"`;
  view.querySelector('#calib-sets-live').hidden = false;
  const sets = presetDetail?.sets || {};
  const devs = Object.keys(sets);
  const n = Math.max(0, ...devs.map(d => sets[d].length));
  if (!n) { box.innerHTML = '<div class="hint" style="padding:8px">this preset has no stored sets</div>'; return; }
  const imgs = presetDetail?.set_images || {};
  const rows = [];
  for (let i = 0; i < n; i++) {
    const camObj = {};
    for (const d of devs) {
      const s = sets[d][i];
      camObj[d] = { detected: !!s, corners: s ? s.corners.length : 0, image: (imgs[i] || {})[d] || null };
    }
    rows.push(setRow({ index: i, cameras: camObj }, true));
  }
  box.innerHTML = rows.join('');
}

async function deleteLiveSet(i) {
  try { await api('DELETE', `/api/calibration/samples/${i}`); log(`Deleted set #${i}`); }
  catch (e) { log('Delete failed: ' + e, 'error'); }
  refreshSets();
}

async function deletePresetSet(i) {
  try {
    presetDetail = await api('DELETE', `/api/calibration/presets/${encodeURIComponent(openedPreset)}/sets/${i}`);
    log(`Preset "${openedPreset}": dropped set #${i}, re-solved`, 'ok');
    renderPresetSets();
    refreshPresets();
  } catch (e) { log('Delete + re-solve failed: ' + e, 'error'); }
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
    const sel_p = presets.find(p => p.name === sel.value);
    const k4a = sel_p && sel_p.k4a && sel_p.k4a.length
      ? ` · k4a ✓ (${sel_p.k4a.length})` : ' · k4a ✗';
    info.textContent = (active
      ? `active: ${active.cameras.length} cam · ${active.sets} sets · ${new Date(active.solved_at * 1000).toLocaleString()}`
      : `${presets.length} saved preset(s)`) + (sel_p ? k4a : '');
    view.querySelector('#calib-preset-open').hidden = !(sel_p && sel_p.sets > 0);
    view.querySelector('#calib-preset-k4a').hidden = !sel_p;
  } catch (e) {
    info.textContent = 'presets unavailable';
    log('Failed to load calibration presets: ' + e, 'error');
  }
}

async function activatePreset(name) {
  if (!name) return;
  try { await api('POST', '/api/calibration/activate', { name }); log(`Activated preset "${name}"`, 'ok'); }
  catch (e) { log(`Failed to activate "${name}": ${e}`, 'error'); }
  refreshPresets();
}

async function openPreset() {
  const name = view.querySelector('#calib-preset').value;
  if (!name) return;
  try {
    presetDetail = await api('GET', `/api/calibration/presets/${encodeURIComponent(name)}`);
    openedPreset = name;
    renderPresetSets();
  } catch (e) { log('Open preset failed: ' + e, 'error'); }
}

function backToLive() { openedPreset = null; presetDetail = null; _setsSig = ''; refreshSets(); }

async function savePreset() {
  const input = view.querySelector('#calib-preset-name');
  const name = (input.value || '').trim();
  if (!name) { input.focus(); return; }
  try {
    await api('POST', '/api/calibration/save-as', { name });
    log(`Saved preset "${name}"`, 'ok');
    input.value = '';
    refreshPresets();
  } catch (e) { log('Save preset failed: ' + e, 'error'); }
}

async function grabPresetK4a() {
  const name = view.querySelector('#calib-preset').value;
  if (!name) { log('Pick a preset first', 'error'); return; }
  try {
    const r = await api('POST', `/api/calibration/presets/${encodeURIComponent(name)}/grab-k4a`);
    log(`Preset "${name}": k4a calibration attached for ${r.devices.join(', ')}`, 'ok');
    refreshPresets();
  } catch (e) { log('Grab k4a failed: ' + e, 'error'); }
}

// ── mount / unmount ───────────────────────────────────────────────────────

export function mount(container) {
  view = container;
  builtIds = '';
  _setsSig = '';
  openedPreset = null;
  view.innerHTML = template();
  syncBoardFieldVisibility();
  syncParams();          // push the (session-remembered) board params to the server
  renderCards();
  refreshPresets();
  refreshSets();

  view.addEventListener('click', onClick);
  view.addEventListener('change', onChange);
  view.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.target.id === 'calib-preset-name') savePreset();
  });
  onCamerasChanged = () => renderCards();
  document.addEventListener('cameras:changed', onCamerasChanged);
  countPoll = setInterval(() => { if (!openedPreset) refreshSets(); }, 1500);
}

export function unmount() {
  if (countPoll) { clearInterval(countPoll); countPoll = null; }
  if (onCamerasChanged) document.removeEventListener('cameras:changed', onCamerasChanged);
  onCamerasChanged = null;
  view?.querySelectorAll('.streams img').forEach(img => { img.src = ''; });
  api('POST', '/api/calibration/reset').catch(() => { });
  view = null;
}

// ── events ────────────────────────────────────────────────────────────────

function onClick(e) {
  if (e.target.dataset.role === 'set-thumb') { window.open(e.target.src, '_blank'); return; }
  const btn = e.target.closest('button');
  if (!btn) return;
  const id = btn.dataset.id;
  switch (btn.id || btn.dataset.role) {
    case 'start': cameras.startCamera(id, cameras.readCardConfig(view, id)); break;
    case 'stop': cameras.stopCamera(id); break;
    case 'calib-start-session': startSession(); break;
    case 'calib-capture-all': captureAll(); break;
    case 'calib-solve': solve(); break;
    case 'calib-clear': clearSamples(); break;
    case 'calib-preset-save': savePreset(); break;
    case 'calib-preset-open': openPreset(); break;
    case 'calib-preset-k4a': grabPresetK4a(); break;
    case 'calib-sets-live': backToLive(); break;
    case 'set-del':
      openedPreset ? deletePresetSet(+btn.dataset.i) : deleteLiveSet(+btn.dataset.i);
      break;
  }
}

function onChange(e) {
  const el = e.target;
  if (el.id === 'board-type') { syncBoardFieldVisibility(); syncParams(); }
  else if (['board-width', 'board-height', 'square-size', 'marker-size'].includes(el.id)
    || el.id === 'aruco-dict') { syncParams(); }
  else if (el.id === 'calib-preset') { activatePreset(el.value); refreshPresets(); }
  else if (['res', 'fps', 'depthmode'].includes(el.dataset.role)) {
    cameras.noteCardChange(view, el.dataset.id);   // session-wide camera config
    renderCards();
  }
}
