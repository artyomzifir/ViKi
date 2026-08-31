// Record tab: pick/create a dataset, give the take an initial label, capture a
// synced RGB-D scene into data/datasets/<dataset>/<id>/, and browse + manage the
// episodes already in that dataset. All input is on-page — no browser dialogs.
import { api, log, state, FRONTEND_CONFIG, sessionGet, sessionPatch } from './core.js';
import * as cameras from './cameras.js';

let view = null;
let onCamerasChanged = null;
let recording = false;
let builtIds = '';               // csv of device ids currently rendered as cards
let editingPath = null;          // episode row in inline metadata-edit mode
let confirmDeletePath = null;    // episode row in inline delete-confirm mode

// ── template ──────────────────────────────────────────────────────────────

function template() {
  const cfg = FRONTEND_CONFIG.recording || { duration: 10, fps: 15 };
  const rec = { duration: cfg.duration ?? 10, fps: cfg.fps ?? 15, ...sessionGet('record', {}) };
  return `
  <div class="record-tab">
    <div class="record-main">
      <div class="record-cards" id="record-cards"></div>
      <div class="record-episodes">
        <div class="record-episodes-head">
          <span>Episodes in <b id="rec-ds-label">—</b></span>
          <button id="rec-eps-refresh">⟳</button>
        </div>
        <div id="rec-episode-list" class="episode-list"></div>
      </div>
    </div>

    <aside class="record-side">
      <section class="calib-sec">
        <div class="calib-sec-title">1 · Dataset</div>
        <select id="rec-dataset"></select>
        <div class="inline-add" id="rec-ds-add" hidden>
          <input type="text" id="rec-ds-name" placeholder="new dataset name">
          <button id="rec-ds-create">Create</button>
        </div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">2 · Label <span class="hint">(optional)</span></div>
        <div class="cfg-row"><label>Task</label><input type="text" id="rec-task" placeholder="pick the cube"></div>
        <div class="cfg-row"><label>Demonstrator</label><input type="text" id="rec-demo"></div>
        <div class="hint">hand is chosen per-run in the Extract tab</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">3 · Capture</div>
        <div class="cfg-row"><label>Seconds</label>
          <input type="number" id="rec-seconds" min="1" value="${rec.duration}"></div>
        <div class="cfg-row"><label>FPS</label>
          <input type="number" id="rec-fps" min="1" value="${rec.fps}"></div>
        <button id="rec-go" class="primary">● Record</button>
        <div class="hint" id="rec-hint">Start ≥1 camera, pick a dataset.</div>
      </section>
    </aside>
  </div>`;
}

const cardHTML = cameras.cameraCardHTML;

// Full rebuild only when the set of devices changes; otherwise sync in place so
// the res / fps / depth selects the user set are not blown away.
function renderCards() {
  const box = view?.querySelector('#record-cards');
  if (!box) return;
  const ids = Object.keys(state);
  const key = ids.join(',');
  if (key !== builtIds) {
    builtIds = key;
    box.innerHTML = ids.length
      ? ids.map(id => cardHTML(id, state[id].type, state[id].running)).join('')
      : `<div class="empty-state"><h2>No cameras</h2><p>Scan for devices (top bar).</p></div>`;
  }
  for (const id of ids) syncCard(box, id);
  updateHint();
}

function syncCard(box, id) {
  const on = !!state[id]?.running;
  const dot = box.querySelector(`[data-role="dot"][data-id="${id}"]`);
  if (dot) dot.className = 'dot ' + (on ? 'green' : 'grey');
  const start = box.querySelector(`[data-role="start"][data-id="${id}"]`);
  const stop = box.querySelector(`[data-role="stop"][data-id="${id}"]`);
  if (start) start.disabled = false;   // allow Start to restart with new settings
  if (stop) stop.disabled = !on;
  const warn = box.querySelector(`[data-role="warn"][data-id="${id}"]`);
  if (warn) {
    const m = cameras.configMismatch(box, id);
    warn.hidden = !m;
    if (m) warn.textContent = `⚠ running as ${(state[id].cfg.depth_mode || '')} `
      + `${state[id].cfg.color_width}×${state[id].cfg.color_height}@${state[id].cfg.fps} — Start to apply`;
  }
  const color = box.querySelector(`img[data-role="color"][data-id="${id}"]`);
  const depth = box.querySelector(`img[data-role="depth"][data-id="${id}"]`);
  // (re)point the stream only when its on/off state flips
  const want = on ? '1' : '';
  if (color && color.dataset.on !== want) {
    cameras.setStream(color, on ? `/api/cameras/${id}/stream` : null); color.dataset.on = want;
  }
  if (depth && depth.dataset.on !== want) {
    cameras.setStream(depth, on ? `/api/cameras/${id}/depth` : null); depth.dataset.on = want;
  }
}

function updateHint() {
  const hint = view?.querySelector('#rec-hint');
  if (!hint) return;
  const anyRunning = Object.values(state).some(s => s.running);
  const ds = view.querySelector('#rec-dataset')?.value;
  view.querySelector('#rec-go').disabled = recording || !anyRunning || !ds;
  hint.textContent = recording ? 'Recording…'
    : !anyRunning ? 'Start ≥1 camera first.'
      : !ds ? 'Pick or create a dataset.'
        : 'Ready.';
}

// ── datasets ──────────────────────────────────────────────────────────────

async function loadDatasets(select) {
  const sel = view.querySelector('#rec-dataset');
  let list = [];
  try { ({ datasets: list } = await api('GET', '/api/datasets')); }
  catch (e) { log('Failed to load datasets: ' + e, 'error'); }
  const keep = select || sel.value || sessionGet('record', {}).dataset;
  sel.innerHTML = '<option value="">＋ new dataset…</option>' +
    list.map(d => `<option value="${d.name}">${d.name} (${d.episodes})</option>`).join('');
  if (keep && list.some(d => d.name === keep)) sel.value = keep;
  else if (!keep && list.length) sel.value = list[0].name;   // prefer a real one
  onDatasetChange();
}

async function createDataset() {
  const input = view.querySelector('#rec-ds-name');
  const name = (input.value || '').trim();
  if (!name) { input.focus(); return; }
  try {
    await api('POST', '/api/datasets', { name });
    log(`Created dataset "${name}"`, 'ok');
    input.value = '';
    await loadDatasets(name);
  } catch (e) { log('Create dataset failed: ' + e, 'error'); }
}

function currentDataset() { return view?.querySelector('#rec-dataset')?.value || ''; }

function onDatasetChange() {
  const ds = currentDataset();
  view.querySelector('#rec-ds-label').textContent = ds || '—';
  view.querySelector('#rec-ds-add').hidden = ds !== '';   // show only for "＋ new dataset…"
  if (ds === '') view.querySelector('#rec-ds-name').focus();
  editingPath = confirmDeletePath = null;
  loadEpisodes();
  updateHint();
}

// ── episode file-manager ─────────────────────────────────────────────────

const STAGES = [
  ['raw', 'RAW', 'raw/ — recorded colour + depth frames'],
  ['rec', 'REC', 'rec.npz — extracted 3-D hand landmarks'],
  ['cln', 'CLN', 'cln.npz — fused + smoothed trajectory'],
  ['plan', 'PLN', 'plan.h5 — retargeted robot joint plan'],
  ['replay', 'RPL', 'replay.h5 — physical replay states'],
];

async function loadEpisodes() {
  const box = view?.querySelector('#rec-episode-list');
  const ds = currentDataset();
  if (!box) return;
  if (!ds) { box.innerHTML = '<div class="hint" style="padding:10px">no dataset</div>'; return; }
  let eps = [];
  try { ({ episodes: eps } = await api('GET', `/api/datasets/${encodeURIComponent(ds)}/episodes`)); }
  catch (e) { log('Failed to list episodes: ' + e, 'error'); return; }
  box.innerHTML = eps.length ? eps.map(rowHTML).join('')
    : '<div class="hint" style="padding:10px">no episodes yet</div>';
}

function esc(s) {
  return String(s ?? '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

function rowHTML(ep) {
  const chips = STAGES.map(([key, label, tip]) =>
    `<span class="badge ${ep.has?.[key] ? 'ok' : ''}" title="${tip}${ep.has?.[key] ? '' : ' (not done)'}">${label}</span>`
  ).join('');

  if (editingPath === ep.path) {
    const hand = (ep.hand || 'right').toLowerCase();
    return `<div class="episode-row editing" data-path="${ep.path}">
      <span class="ep-id" title="capture id (not editable)">${ep.id}</span>
      <input class="ep-edit-name" data-role="ep-name" placeholder="name / task"
             value="${esc(ep.task)}">
      <input class="ep-edit-demo" data-role="ep-demo" placeholder="demonstrator"
             value="${esc(ep.demonstrator)}">
      <select class="ep-edit-hand" data-role="ep-hand">
        <option value="right"${hand === 'right' ? ' selected' : ''}>right</option>
        <option value="left"${hand === 'left' ? ' selected' : ''}>left</option>
      </select>
      <button data-role="ep-edit-save">save</button>
      <button data-role="ep-edit-cancel">cancel</button>
    </div>`;
  }

  const actions = confirmDeletePath === ep.path
    ? `<span class="hint">delete?</span>
       <button data-role="ep-delete-yes" class="danger">yes</button>
       <button data-role="ep-delete-no">no</button>`
    : `<button data-role="ep-edit">edit</button>
       <button data-role="ep-delete" class="danger">del</button>`;
  const meta = [
    ep.demonstrator,
    ep.hand,
    ep.duration_s != null ? `${ep.duration_s}s` : '',
    ep.fps ? `${ep.fps} fps` : '',
  ].filter(Boolean).join(' · ');
  return `<div class="episode-row" data-path="${ep.path}">
    <span class="ep-id" title="capture id">${ep.id}</span>
    <span class="ep-task">
      <span class="ep-name">${ep.task ? esc(ep.task) : '<i>unnamed</i>'}</span>
      ${meta ? `<span class="ep-meta">${esc(meta)}</span>` : ''}
    </span>
    <span class="ep-badges">${chips}</span>
    ${actions}
  </div>`;
}

async function doEditMeta(path, fields) {
  try {
    await api('PATCH', '/api/episodes/meta', { path, ...fields });
    log('Episode updated', 'ok');
  } catch (e) { log('Update failed: ' + e, 'error'); }
  editingPath = null;
  loadEpisodes();
}

async function doDelete(path) {
  try {
    await api('DELETE', '/api/episodes', { path });
    log('Episode deleted', 'ok');
  } catch (e) { log('Delete failed: ' + e, 'error'); }
  confirmDeletePath = null;
  loadEpisodes();
  loadDatasets();
}

// ── record ───────────────────────────────────────────────────────────────

async function record() {
  const body = {
    dataset: currentDataset(),
    task: view.querySelector('#rec-task').value,
    demonstrator: view.querySelector('#rec-demo').value,
    seconds: +view.querySelector('#rec-seconds').value || 10,
    fps: +view.querySelector('#rec-fps').value || 15,
  };
  if (!body.dataset) { log('Pick a dataset first', 'error'); return; }
  const box = view.querySelector('#record-cards');
  const stale = Object.keys(state).filter(id => state[id].running && cameras.configMismatch(box, id));
  if (stale.length) {
    log(`${stale.join(', ')}: card settings not applied — click Start on the card first`, 'error');
    return;
  }
  let job_id;
  try { ({ job_id } = await api('POST', '/api/record/start', body)); }
  catch (e) { log('Record failed: ' + e, 'error'); return; }
  log(`Recording ${body.seconds}s into "${body.dataset}" (${job_id})`);
  recording = true;
  cameras.setRecording(true);
  updateHint();
  const poll = setInterval(async () => {
    let j;
    try { j = await api('GET', `/api/record/jobs/${job_id}`); } catch { return; }
    if (j.status === 'running') return;
    clearInterval(poll);
    recording = false;
    cameras.setRecording(false);
    updateHint();
    if (j.status === 'done') { log(`Recorded → ${j.result?.episode}`, 'ok'); loadEpisodes(); loadDatasets(); }
    else log(`Recording failed: ${j.error}`, 'error');
  }, 1000);
}

// ── events ───────────────────────────────────────────────────────────────

function onClick(e) {
  const btn = e.target.closest('button');
  if (!btn) return;
  const row = e.target.closest('.episode-row');
  const id = btn.dataset.id;
  const path = row?.dataset.path;
  switch (btn.id || btn.dataset.role) {
    case 'start': cameras.startCamera(id, cameras.readCardConfig(view, id)); break;
    case 'stop': cameras.stopCamera(id); break;
    case 'rec-ds-create': createDataset(); break;
    case 'rec-eps-refresh': loadEpisodes(); break;
    case 'rec-go': record(); break;
    case 'ep-edit': editingPath = path; confirmDeletePath = null; loadEpisodes(); break;
    case 'ep-edit-cancel': editingPath = null; loadEpisodes(); break;
    case 'ep-edit-save':
      doEditMeta(path, {
        task: row.querySelector('[data-role="ep-name"]').value.trim(),
        demonstrator: row.querySelector('[data-role="ep-demo"]').value.trim(),
        hand: row.querySelector('[data-role="ep-hand"]').value,
      }); break;
    case 'ep-delete': confirmDeletePath = path; editingPath = null; loadEpisodes(); break;
    case 'ep-delete-no': confirmDeletePath = null; loadEpisodes(); break;
    case 'ep-delete-yes': doDelete(path); break;
  }
}

function onKeydown(e) {
  if (e.key === 'Enter' && e.target.id === 'rec-ds-name') createDataset();
  else if (e.key === 'Enter' && ['ep-name', 'ep-demo'].includes(e.target.dataset.role)) {
    const row = e.target.closest('.episode-row');
    doEditMeta(row.dataset.path, {
      task: row.querySelector('[data-role="ep-name"]').value.trim(),
      demonstrator: row.querySelector('[data-role="ep-demo"]').value.trim(),
      hand: row.querySelector('[data-role="ep-hand"]').value,
    });
  }
}

function onChange(e) {
  if (e.target.id === 'rec-dataset') { persistRec(); onDatasetChange(); }
  else if (e.target.id === 'rec-seconds' || e.target.id === 'rec-fps') persistRec();
  else if (['res', 'fps', 'depthmode'].includes(e.target.dataset.role)) {
    cameras.noteCardChange(view, e.target.dataset.id);  // fold into session config
    renderCards();  // refresh the "running as X" mismatch warning
  }
}

function persistRec() {
  sessionPatch('record', {
    duration: +view.querySelector('#rec-seconds').value || 10,
    fps: +view.querySelector('#rec-fps').value || 15,
    dataset: view.querySelector('#rec-dataset').value || '',
  });
}

// ── mount / unmount ──────────────────────────────────────────────────────

export function mount(container) {
  view = container;
  builtIds = '';
  view.innerHTML = template();
  renderCards();
  loadDatasets();
  view.addEventListener('click', onClick);
  view.addEventListener('change', onChange);
  view.addEventListener('keydown', onKeydown);
  onCamerasChanged = () => renderCards();
  document.addEventListener('cameras:changed', onCamerasChanged);
}

export function unmount() {
  if (onCamerasChanged) document.removeEventListener('cameras:changed', onCamerasChanged);
  onCamerasChanged = null;
  view?.querySelectorAll('.streams img').forEach(img => { img.src = ''; });
  recording = false;
  cameras.setRecording(false);
  view = null;
}
