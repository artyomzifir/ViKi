// Record tab: pick/create a dataset, give the take an initial label, capture a
// synced RGB-D scene into data/datasets/<dataset>/<id>/, and browse + manage the
// episodes already in that dataset.
import { api, log, state, FRONTEND_CONFIG } from './core.js';
import * as cameras from './cameras.js';

let view = null;
let onCamerasChanged = null;
let recording = false;

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
        <button id="rec-dataset-new">＋ New dataset…</button>
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

function cardHTML(id, type, running) {
  return `
  <div class="cam-card" data-id="${id}">
    <div class="card-header">
      <span class="dot ${running ? 'green' : 'grey'}"></span>
      <span class="name">${id}</span>
      <span class="tag ${type}">${type}</span>
      <button data-role="start" data-id="${id}" ${running ? 'disabled' : ''}>▶ Start</button>
      <button data-role="stop" data-id="${id}" class="danger" ${running ? '' : 'disabled'}>■ Stop</button>
    </div>
    <div class="card-controls">${cameras.controlsHTML(id, type)}</div>
    <div class="streams">
      <div class="stream-panel"><img data-role="color" data-id="${id}"><span class="stream-label">COLOR</span></div>
      <div class="stream-divider"></div>
      <div class="stream-panel"><img data-role="depth" data-id="${id}"><span class="stream-label">DEPTH</span></div>
    </div>
  </div>`;
}

function renderCards() {
  const box = view?.querySelector('#record-cards');
  if (!box) return;
  const ids = Object.keys(state);
  if (!ids.length) {
    box.innerHTML = `<div class="empty-state"><h2>No cameras</h2><p>Scan for devices (top bar).</p></div>`;
    return;
  }
  box.innerHTML = ids.map(id => cardHTML(id, state[id].type, state[id].running)).join('');
  for (const id of ids) {
    const on = state[id].running;
    cameras.setStream(box.querySelector(`img[data-role="color"][data-id="${id}"]`),
      on ? `/api/cameras/${id}/stream` : null);
    cameras.setStream(box.querySelector(`img[data-role="depth"][data-id="${id}"]`),
      on ? `/api/cameras/${id}/depth` : null);
  }
  updateHint();
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
  sel.innerHTML = list.length
    ? list.map(d => `<option value="${d.name}">${d.name} (${d.episodes})</option>`).join('')
    : '<option value="">— none —</option>';
  if (keep && list.some(d => d.name === keep)) sel.value = keep;
  onDatasetChange();
}

async function newDataset() {
  const name = prompt('New dataset name:', '');
  if (!name || !name.trim()) return;
  try {
    await api('POST', '/api/datasets', { name: name.trim() });
    log(`Created dataset "${name.trim()}"`, 'ok');
    await loadDatasets(name.trim());
  } catch (e) { log('Create dataset failed: ' + e, 'error'); }
}

function currentDataset() { return view?.querySelector('#rec-dataset')?.value || ''; }

function onDatasetChange() {
  const ds = currentDataset();
  view.querySelector('#rec-ds-label').textContent = ds || '—';
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
  return `
  <div class="episode-row" data-path="${ep.path}">
    <span class="ep-id">${ep.id}</span>
    <span class="ep-task">${ep.task || '<i>unlabelled</i>'}</span>
    <span class="ep-badges">${chips}</span>
    <button data-role="ep-rename">rename</button>
    <button data-role="ep-delete" class="danger">del</button>
  </div>`;
}

async function renameEpisode(path) {
  const cur = path.split('/').pop();
  const nid = prompt('Rename episode to:', cur);
  if (!nid || nid === cur) return;
  try {
    await api('PATCH', '/api/episodes/rename', { path, new_id: nid });
    log('Episode renamed', 'ok');
    loadEpisodes();
  } catch (e) { log('Rename failed: ' + e, 'error'); }
}

async function deleteEpisode(path) {
  if (!confirm(`Delete episode ${path.split('/').pop()}? This removes all its files.`)) return;
  try {
    await api('DELETE', '/api/episodes', { path });
    log('Episode deleted', 'ok');
    loadEpisodes();
  } catch (e) { log('Delete failed: ' + e, 'error'); }
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
  switch (btn.id || btn.dataset.role) {
    case 'start': cameras.startCamera(id, cameras.readCardConfig(view, id)); break;
    case 'stop': cameras.stopCamera(id); break;
    case 'rec-dataset-new': newDataset(); break;
    case 'rec-eps-refresh': loadEpisodes(); break;
    case 'rec-go': record(); break;
    case 'ep-rename': if (row) renameEpisode(row.dataset.path); break;
    case 'ep-delete': if (row) deleteEpisode(row.dataset.path); break;
  }
}

function onChange(e) {
  if (e.target.id === 'rec-dataset') onDatasetChange();
  else if (e.target.dataset.role === 'depthmode') cameras.updateFpsForDepthMode(view, e.target.dataset.id);
}

// ── mount / unmount ──────────────────────────────────────────────────────

export function mount(container) {
  view = container;
  view.innerHTML = template();
  renderCards();
  loadDatasets();
  view.addEventListener('click', onClick);
  view.addEventListener('change', onChange);
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
