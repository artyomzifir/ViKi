// Record tab: pick/create a dataset, give the take an initial label, capture a
// synced RGB-D scene into data/datasets/<dataset>/<id>/, and browse + manage the
// episodes already in that dataset. All input is on-page — no browser dialogs.
import { api, log, state, FRONTEND_CONFIG } from './core.js';
import * as cameras from './cameras.js';

let view = null;
let onCamerasChanged = null;
let recording = false;
let builtIds = '';               // csv of device ids currently rendered as cards
let renamingPath = null;         // episode row in inline-rename mode
let confirmDeletePath = null;    // episode row in inline delete-confirm mode

// ── template ──────────────────────────────────────────────────────────────

function template() {
  const rec = FRONTEND_CONFIG.recording || { duration: 10, fps: 15 };
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
        <div class="calib-sec-title">Dataset</div>
        <select id="rec-dataset"></select>
        <div class="inline-add" id="rec-ds-add" hidden>
          <input type="text" id="rec-ds-name" placeholder="new dataset name">
          <button id="rec-ds-create">Create</button>
        </div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Label</div>
        <div class="cfg-row"><label>Task</label><input type="text" id="rec-task" placeholder="pick the cube"></div>
        <div class="cfg-row"><label>Hand</label>
          <select id="rec-hand"><option>right</option><option>left</option></select></div>
        <div class="cfg-row"><label>Demonstrator</label><input type="text" id="rec-demo"></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Capture</div>
        <div class="cfg-row"><label>Seconds</label>
          <input type="number" id="rec-seconds" min="1" value="${rec.duration ?? 10}"></div>
        <div class="cfg-row"><label>FPS</label>
          <input type="number" id="rec-fps" min="1" value="${rec.fps ?? 15}"></div>
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
  if (start) start.disabled = on;
  if (stop) stop.disabled = !on;
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
  const keep = select || sel.value;
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
  renamingPath = confirmDeletePath = null;
  loadEpisodes();
  updateHint();
}

// ── episode file-manager ─────────────────────────────────────────────────

const STAGES = ['raw', 'rec', 'cln', 'plan', 'replay'];

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

function rowHTML(ep) {
  const chips = STAGES.map(s =>
    `<span class="badge ${ep.has?.[s] ? 'ok' : ''}">${s[0].toUpperCase()}</span>`).join('');
  const id = renamingPath === ep.path
    ? `<input class="ep-id-edit" data-role="ep-name" value="${ep.id}">
       <button data-role="ep-rename-ok">save</button>
       <button data-role="ep-rename-cancel">cancel</button>`
    : `<span class="ep-id">${ep.id}</span>`;
  const actions = confirmDeletePath === ep.path
    ? `<span class="hint">delete?</span>
       <button data-role="ep-delete-yes" class="danger">yes</button>
       <button data-role="ep-delete-no">no</button>`
    : `<button data-role="ep-rename">rename</button>
       <button data-role="ep-delete" class="danger">del</button>`;
  return `<div class="episode-row" data-path="${ep.path}">
    ${id}
    <span class="ep-task">${ep.task || '<i>unlabelled</i>'}</span>
    <span class="ep-badges">${chips}</span>
    ${actions}
  </div>`;
}

async function doRename(path, newId) {
  if (!newId || newId === path.split('/').pop()) { renamingPath = null; loadEpisodes(); return; }
  try {
    await api('PATCH', '/api/episodes/rename', { path, new_id: newId });
    log('Episode renamed', 'ok');
  } catch (e) { log('Rename failed: ' + e, 'error'); }
  renamingPath = null;
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
    hand: view.querySelector('#rec-hand').value,
    demonstrator: view.querySelector('#rec-demo').value,
    seconds: +view.querySelector('#rec-seconds').value || 10,
    fps: +view.querySelector('#rec-fps').value || 15,
  };
  if (!body.dataset) { log('Pick a dataset first', 'error'); return; }
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
    case 'ep-rename': renamingPath = path; confirmDeletePath = null; loadEpisodes(); break;
    case 'ep-rename-cancel': renamingPath = null; loadEpisodes(); break;
    case 'ep-rename-ok':
      doRename(path, row.querySelector('[data-role="ep-name"]').value.trim()); break;
    case 'ep-delete': confirmDeletePath = path; renamingPath = null; loadEpisodes(); break;
    case 'ep-delete-no': confirmDeletePath = null; loadEpisodes(); break;
    case 'ep-delete-yes': doDelete(path); break;
  }
}

function onKeydown(e) {
  if (e.key === 'Enter' && e.target.id === 'rec-ds-name') createDataset();
  else if (e.key === 'Enter' && e.target.dataset.role === 'ep-name') {
    const row = e.target.closest('.episode-row');
    doRename(row.dataset.path, e.target.value.trim());
  }
}

function onChange(e) {
  if (e.target.id === 'rec-dataset') onDatasetChange();
  else if (e.target.dataset.role === 'depthmode') cameras.updateFpsForDepthMode(view, e.target.dataset.id);
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
