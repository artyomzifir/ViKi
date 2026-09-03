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
// Picker selection: null = follow the active preset; "" = user parked on
// "current, unsaved" (a fresh solve, will Save as a NEW preset); "<name>" = a
// specific preset the user picked.
let presetChoice = null;

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
    <aside class="calib-leftcol">
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

      <div class="calib-sets">
        <div class="calib-sets-head">
          <span id="calib-sets-title">Captured sets</span>
          <button id="calib-sets-live" hidden>← live</button>
        </div>
        <div id="calib-sets-list"></div>
      </div>
    </aside>

    <div class="calib-cards" id="calib-cards"></div>

    <aside class="calib-side">
      <section class="calib-sec">
        <div class="calib-sec-title">Load a saved preset</div>
        <select id="calib-preset"></select>
        <div class="hint" id="calib-preset-info">—</div>
        <div class="calib-preset-acts">
          <button id="calib-preset-open" hidden>Open sets</button>
          <button id="calib-preset-rename">Rename</button>
          <button id="calib-preset-del" class="danger">Delete</button>
        </div>
        <div class="inline-add" id="calib-preset-rename-row" hidden>
          <input type="text" id="calib-preset-newname" placeholder="new name">
          <button id="calib-preset-rename-go">OK</button>
          <button id="calib-preset-rename-cancel">✕</button>
        </div>
        <div class="inline-add" id="calib-preset-del-row" hidden>
          <span class="hint">delete this preset (sets + k4a + background)?</span>
          <button id="calib-preset-del-yes" class="danger">yes</button>
          <button id="calib-preset-del-no">no</button>
        </div>
      </section>

      <section class="calib-sec calib-wizard">
        <div class="calib-sec-title">Setup steps</div>

        <div class="wiz-step" data-step="calibrate">
          <div class="wiz-head"><span class="wiz-dot"></span><b>1 · Calibrate</b>
            <span class="wiz-state" data-role="st-calibrate">—</span></div>
          <div class="hint">Start every camera (top bar). Then move the board
            through <b>different</b> poses — tilt it, near &amp; far — and Capture
            each. A pose too close to one already taken is refused.</div>
          <button id="calib-start-session" class="primary">Start session</button>
          <button id="calib-capture-all" class="primary">Capture set</button>
          <div id="calib-readiness" class="wiz-crit"></div>
          <button id="calib-solve" class="primary" disabled>Solve (bundle)</button>
        </div>

        <div class="wiz-step" data-step="anchor">
          <div class="wiz-head"><span class="wiz-dot"></span><b>2 · Anchor</b>
            <span class="wiz-state" data-role="st-anchor">—</span></div>
          <div class="hint">Lay the board flat on the marked <b>home</b> spot.
            One frame — it defines the world origin (viz / workspace only).</div>
          <button id="calib-anchor" class="primary" disabled>Capture anchor</button>
        </div>

        <div class="wiz-step" data-step="background">
          <div class="wiz-head"><span class="wiz-dot"></span><b>3 · Background</b>
            <span class="wiz-state" data-role="st-background">—</span></div>
          <div class="hint"><b>Remove the board</b> and step out of frame.</div>
          <button id="calib-background" class="primary" disabled>Grab background</button>
        </div>

        <div class="wiz-step" data-step="validate">
          <div class="wiz-head"><span class="wiz-dot"></span><b>4 · Validate</b>
            <span class="wiz-state" data-role="st-validate">—</span></div>
          <div class="hint">Empty scene. Checks the per-camera clouds actually
            overlap. Recording is blocked on a <b>red</b> verdict.</div>
          <button id="calib-validate" class="primary" disabled>Run validation</button>
          <div id="calib-validation" class="wiz-crit"></div>
        </div>

        <div class="inline-add">
          <input type="text" id="calib-preset-name" placeholder="preset name">
          <button id="calib-preset-save" class="primary">Save preset</button>
        </div>
        <button id="calib-clear" class="danger">Clear samples</button>
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
      ? ids.map(id => cameras.cameraCardHTML(id, state[id].type, state[id].running)).join('')
      : `<div class="empty-state"><h2>No cameras</h2><p>Scan for devices (top bar).</p></div>`;
  }
  for (const id of ids) {
    const on = !!state[id]?.running;
    const card = box.querySelector(`.cam-card[data-id="${id}"]`);
    if (!card) continue;
    card.querySelector('[data-role="dot"]').className = 'dot ' + (on ? 'green' : 'grey');
    card.querySelector('[data-role="start"]').disabled = on;
    card.querySelector('[data-role="stop"]').disabled = !on;
    // one stream per card; RGB shows the board overlay, D the raw depth
    cameras.syncCardStream(card, id, on, `/api/calibration/${id}/stream`);
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
    else if (r.rejected) log(`Set refused — ${r.reason || `too close to set #${r.conflict}`}`, 'error');
    else log(`Board not seen by: ${(r.missing || []).join(', ') || 'a camera'} — set discarded`, 'error');
  } catch (e) { log('Capture failed: ' + e, 'error'); }
  refreshSets();
  refreshSetup();
}

async function solve() {
  log('Bundle solve…');
  try {
    const r = await api('POST', '/api/calibration/solve');
    const rms = Object.entries(r.solve?.rms_reproj_px || {})
      .map(([d, v]) => `${d} ${(+v).toFixed(2)}px`).join(', ');
    log(`Extrinsics solved (ref ${r.reference_device}; ${rms})`
      + (r.solve?.degenerate ? ' — ⚠ DEGENERATE, collect more varied poses' : ''),
      r.solve?.degenerate ? 'error' : 'ok');
    const sc = r.solve?.stereo_check;
    if (sc && sc.ran && !sc.agrees)
      log(`stereoCalibrate disagrees by ${sc.delta_translation_mm}mm / ${sc.delta_rotation_deg}° — check the solve`, 'error');
    await Promise.all(Object.keys(state).map(async id => {
      try { await api('POST', `/api/skeleton/capture_base/${id}`); } catch (_e) { }
    }));
    presetChoice = '';
    refreshPresets();
  } catch (e) {
    log('Bundle solve failed: ' + e, 'error');
  }
  refreshSetup();
}

async function captureAnchor() {
  try {
    const r = await api('POST', '/api/calibration/anchor');
    log(`World anchor captured (seen by ${Object.keys(r.observations || {}).join(', ')})`, 'ok');
  } catch (e) { log('Anchor failed: ' + e, 'error'); }
  refreshSetup();
}

async function grabBackground() {
  log('Grabbing empty-scene background…');
  try {
    const r = await api('POST', '/api/calibration/background');
    const worst = Math.max(0, ...Object.values(r).map(v => v.invalid_frac));
    log(`Background grabbed (worst ${(worst * 100).toFixed(1)}% invalid pixels)`,
      worst > 0.25 ? 'error' : 'ok');
  } catch (e) { log('Background grab failed: ' + e, 'error'); }
  refreshSetup();
}

async function runValidate() {
  log('Validating cloud agreement…');
  try {
    const r = await api('POST', '/api/calibration/validate');
    renderValidation(r);
    log(`Validation: ${r.verdict.toUpperCase()}`,
      r.verdict === 'green' ? 'ok' : r.verdict === 'amber' ? 'warn' : 'error');
  } catch (e) { log('Validation failed: ' + e, 'error'); }
  refreshSetup();
}

// ── setup-step state ────────────────────────────────────────────────────

function setStepState(step, txt, cls) {
  const el = view?.querySelector(`[data-role="st-${step}"]`);
  if (el) { el.textContent = txt; el.className = 'wiz-state ' + (cls || ''); }
}

function renderReadiness(rd) {
  const box = view?.querySelector('#calib-readiness');
  if (!box) return;
  if (!rd || !rd.criteria) { box.innerHTML = ''; return; }
  box.innerHTML = rd.criteria.map(c =>
    `<div class="crit ${c.ok ? 'ok' : 'bad'}">${c.ok ? '✓' : '·'} ${c.name.replace(/_/g, ' ')}
      <span>${c.value} / ${c.need}</span></div>`).join('');
  const solveBtn = view.querySelector('#calib-solve');
  if (solveBtn) solveBtn.disabled = !rd.ready;
}

function renderValidation(v) {
  const box = view?.querySelector('#calib-validation');
  if (!box) return;
  if (!v || !v.pairs) { box.innerHTML = ''; return; }
  box.innerHTML =
    `<div class="crit ${v.verdict}">verdict <b>${v.verdict}</b></div>` +
    v.pairs.map(p => p.skipped
      ? `<div class="crit bad">${p.a}–${p.b}: ${p.reason}</div>`
      : `<div class="crit ${p.verdict}">${p.a}–${p.b}: NN ${p.nn_median_mm}mm ·
         ICP ${p.icp_translation_mm}mm / ${p.icp_rotation_deg}°</div>`).join('');
}

async function refreshSetup() {
  if (!view || openedPreset) return;
  // readiness (needs a live session)
  let rd = null;
  try { rd = await api('GET', '/api/calibration/readiness'); } catch (_e) { }
  renderReadiness(rd);
  setStepState('calibrate',
    rd ? (rd.ready ? 'ready to solve' : `${rd.n_sets} sets`) : 'no session',
    rd?.ready ? 'ok' : '');

  let anchor = null;
  try { anchor = await api('GET', '/api/calibration/anchor'); } catch (_e) { }
  setStepState('anchor', anchor ? 'captured' : 'not captured', anchor ? 'ok' : '');
  view.querySelector('#calib-anchor').disabled = false;

  let val = null;
  try { val = await api('GET', '/api/calibration/validate'); } catch (_e) { }
  renderValidation(val);
  setStepState('validate', val ? val.verdict : 'not run',
    val ? (val.verdict === 'green' ? 'ok' : val.verdict === 'red' ? 'bad' : 'warn') : '');
  view.querySelector('#calib-background').disabled = false;
  view.querySelector('#calib-validate').disabled = false;
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
    // newest set on top so the running count is obvious after each capture
    box.innerHTML = rows.length ? [...rows].reverse().map(r => setRow(r, true)).join('')
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
  box.innerHTML = rows.reverse().join('');   // newest on top
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
    const active = presets.find(p => p.active);
    // Which option to show: the user's explicit pick (if it still exists),
    // else the active preset, else "current, unsaved".
    let choice = presetChoice;
    if (choice === null) choice = active ? active.name : '';
    if (choice && !presets.some(p => p.name === choice)) choice = '';

    sel.innerHTML =
      `<option value=""${choice === '' ? ' selected' : ''}>— current, unsaved (Save → new preset) —</option>` +
      presets.map(p =>
        `<option value="${p.name}"${p.name === choice ? ' selected' : ''}>` +
        `${p.name}${p.active ? ' ✓' : ''}</option>`).join('');

    const sel_p = presets.find(p => p.name === sel.value);
    if (!sel.value) {
      info.textContent = 'unsaved live solve — "4 · Save" keeps it as a NEW preset'
        + (active ? ` · "${active.name}" stays active until you save & switch` : '');
    } else {
      const k4a = sel_p && sel_p.k4a && sel_p.k4a.length ? ` · k4a ✓ (${sel_p.k4a.length})` : ' · k4a ✗';
      const bg = sel_p && sel_p.background && sel_p.background.length ? ` · bg ✓ (${sel_p.background.length})` : ' · bg ✗';
      info.textContent = (sel_p && sel_p.active
        ? `active: ${sel_p.cameras.length} cam · ${sel_p.sets} sets · ${new Date(sel_p.solved_at * 1000).toLocaleString()}`
        : `${sel_p ? sel_p.sets + ' sets — pick to activate' : ''}`) + k4a + bg;
    }
    view.querySelector('#calib-preset-open').hidden = !(sel_p && sel_p.sets > 0);
    const real = !!sel.value;
    view.querySelector('#calib-preset-rename').disabled = !real;
    view.querySelector('#calib-preset-del').disabled = !real;
    showPresetRow(null);
  } catch (e) {
    info.textContent = 'presets unavailable';
    log('Failed to load calibration presets: ' + e, 'error');
  }
}

function currentPresetName() { return view?.querySelector('#calib-preset')?.value || ''; }

function showPresetRow(which) {   // 'rename' | 'del' | null
  const rn = view?.querySelector('#calib-preset-rename-row');
  const dl = view?.querySelector('#calib-preset-del-row');
  if (!rn || !dl) return;
  rn.hidden = which !== 'rename';
  dl.hidden = which !== 'del';
  if (which === 'rename') {
    const inp = view.querySelector('#calib-preset-newname');
    inp.value = currentPresetName(); inp.focus(); inp.select();
  }
}

async function deletePreset() {
  const name = currentPresetName();
  if (!name) return;
  try {
    await api('DELETE', `/api/calibration/presets/${encodeURIComponent(name)}`);
    log(`Deleted preset "${name}"`, 'ok');
  } catch (e) { log(`Delete preset failed: ${e}`, 'error'); }
  if (openedPreset === name) backToLive();
  presetChoice = null;   // fall back to whatever is active
  refreshPresets();
}

async function renamePreset() {
  const name = currentPresetName();
  const nn = (view.querySelector('#calib-preset-newname').value || '').trim();
  if (!name || !nn || nn === name) { showPresetRow(null); return; }
  try {
    const r = await api('PATCH', `/api/calibration/presets/${encodeURIComponent(name)}`, { new: nn });
    log(`Renamed "${name}" → "${r.name}"`, 'ok');
    if (openedPreset === name) openedPreset = r.name;
    if (presetChoice === name) presetChoice = r.name;
  } catch (e) { log(`Rename preset failed: ${e}`, 'error'); }
  refreshPresets();
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
    await api('POST', '/api/calibration/activate', { name });   // saved → make it active + selected
    log(`Saved preset "${name}" and activated it`, 'ok');
    input.value = '';
    presetChoice = name;
    refreshPresets();
  } catch (e) { log('Save preset failed: ' + e, 'error'); }
}

// ── mount / unmount ───────────────────────────────────────────────────────

export function mount(container) {
  view = container;
  builtIds = '';
  _setsSig = '';
  openedPreset = null;
  presetChoice = null;   // start by following the active preset
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
    else if (e.key === 'Enter' && e.target.id === 'calib-preset-newname') renamePreset();
    else if (e.key === 'Escape' && ['calib-preset-newname'].includes(e.target.id)) showPresetRow(null);
  });
  onCamerasChanged = () => renderCards();
  document.addEventListener('cameras:changed', onCamerasChanged);
  refreshSetup();
  countPoll = setInterval(() => {
    if (!openedPreset) { refreshSets(); refreshSetup(); }
  }, 1800);
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
    case 'view': cameras.setCardView(btn, `/api/calibration/${id}/stream`); break;
    case 'calib-start-session': startSession(); break;
    case 'calib-capture-all': captureAll(); break;
    case 'calib-solve': solve(); break;
    case 'calib-anchor': captureAnchor(); break;
    case 'calib-background': grabBackground(); break;
    case 'calib-validate': runValidate(); break;
    case 'calib-clear': clearSamples(); break;
    case 'calib-preset-save': savePreset(); break;
    case 'calib-preset-open': openPreset(); break;
    case 'calib-preset-rename': showPresetRow('rename'); break;
    case 'calib-preset-rename-go': renamePreset(); break;
    case 'calib-preset-del': showPresetRow('del'); break;
    case 'calib-preset-del-yes': deletePreset(); break;
    case 'calib-preset-rename-cancel':
    case 'calib-preset-del-no': showPresetRow(null); break;
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
  else if (el.id === 'calib-preset') {
    presetChoice = el.value;                 // remember what the user parked on
    if (el.value) activatePreset(el.value);  // real preset → activate it
    else { log('On the unsaved solve — "4 · Save" keeps it as a new preset'); refreshPresets(); }
  }
  else if (['res', 'fps', 'depthmode'].includes(el.dataset.role)) {
    cameras.noteCardChange(view, el.dataset.id);   // session-wide camera config
    renderCards();
  }
}
