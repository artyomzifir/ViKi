// Record tab: pick/create a dataset, give the take an initial label, capture a
// synced RGB-D scene into data/datasets/<dataset>/<id>/, and browse + manage the
// episodes already in that dataset. All input is on-page — no browser dialogs.
import { api, log, state, FRONTEND_CONFIG, sessionGet, sessionPatch } from './core.js';
import * as cameras from './cameras.js';
import * as episodes from './episodes.js';

let view = null;
let onCamerasChanged = null;
let cloudPoll = null;            // interval id for the post-capture cloud-build bar
let cloudProg = null;            // { id, status, pct, label } — bar shown on that episode row
let recording = false;
let builtIds = '';               // csv of device ids currently rendered as cards
let editingPath = null;          // episode row in inline metadata-edit mode
let confirmDeletePath = null;    // episode row in inline delete-confirm mode

// ── template ──────────────────────────────────────────────────────────────

function template() {
  const cfg = FRONTEND_CONFIG.recording || { duration: 10, fps: 15 };
  const rec = { duration: cfg.duration ?? 10, fps: cfg.fps ?? 15, ...sessionGet('record', {}) };
  const cl = { ...(FRONTEND_CONFIG.cloud || { stride: 1, voxel: 0.005, maxPoints: 40000, bbox: [-0.6, 0.6, -0.6, 0.6, -0.2, 1.2] }), ...sessionGet('cloud', {}) };
  return `
  <div class="record-tab">
    <aside class="record-leftcol">
      <section class="calib-sec">
        <div class="calib-sec-title">Dataset</div>
        <select id="rec-dataset"></select>
        <div class="inline-add" id="rec-ds-add" hidden>
          <input type="text" id="rec-ds-name" placeholder="new dataset name">
          <button id="rec-ds-create">Create</button>
        </div>
      </section>
      <div class="record-episodes">
        <div class="record-episodes-head">
          <span>Episodes in <b id="rec-ds-label">—</b></span>
          <button id="rec-eps-refresh">⟳</button>
        </div>
        <div id="rec-episode-list" class="episode-list"></div>
      </div>
    </aside>

    <div class="record-cards" id="record-cards"></div>

    <aside class="record-side">
      <section class="calib-sec">
        <div class="calib-sec-title">Label <span class="hint">(optional)</span></div>
        <div class="cfg-row"><label>Task</label><input type="text" id="rec-task" placeholder="pick the cube"></div>
        <div class="cfg-row"><label>Demonstrator</label><input type="text" id="rec-demo"></div>
        <div class="hint">hand is chosen per-run in the Extract tab</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Capture</div>
        <div class="cfg-row"><label>Max seconds</label>
          <input type="number" id="rec-seconds" min="1" value="${rec.duration}"></div>
        <div class="cfg-row"><label>FPS</label>
          <input type="number" id="rec-fps" min="1" value="${rec.fps}"></div>
        <button id="rec-go" class="primary">● Record</button>
        <div class="hint" id="rec-hint">Start ≥1 camera, pick a dataset.</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Point cloud <span class="hint">(built after capture)</span></div>
        <div class="rec-field"><label>Depth pixel stride</label>
          <input type="number" id="rec-cl-stride" min="1" step="1" value="${cl.stride}"></div>
        <div class="rec-field"><label>Voxel leaf (m) <span class="hint">— 0 keeps every point</span></label>
          <input type="number" id="rec-cl-voxel" min="0" step="0.001" value="${cl.voxel}"></div>
        <div class="rec-field"><label>Max points / frame <span class="hint">— 0 = no cap</span></label>
          <input type="number" id="rec-cl-max" min="0" step="1000" value="${cl.maxPoints}"></div>
        <div class="rec-field"><label>Workspace crop AABB <span class="hint">— empty = full scene (1:1)</span></label>
          <input type="text" id="rec-cl-bbox" placeholder="x0,x1,y0,y1,z0,z1 (m)" value="${(cl.bbox || []).join(', ')}"></div>
        <label class="cfg-row"><span>Subtract calibrated background</span>
          <input type="checkbox" id="rec-cl-bg" ${cl.bgSubtract !== false ? 'checked' : ''}></label>
        <div class="hint">drops points matching the empty-scene depth from the preset — lighter cloud, cleaner segmentation. The build progress shows on the episode row once the capture ends.</div>
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
  const go = view.querySelector('#rec-go');
  // while recording the button becomes Stop (always clickable)
  go.disabled = recording ? false : (!anyRunning || !ds);
  go.textContent = recording ? '■ Stop' : '● Record';
  go.classList.toggle('danger', recording);
  hint.textContent = recording ? 'Recording… (stops at max seconds or Stop)'
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
// Row rendering is shared with the Extract tab (episodes.js); Record opts into
// the inline edit / delete controls via `manage`.

async function loadEpisodes() {
  const box = view?.querySelector('#rec-episode-list');
  const ds = currentDataset();
  if (!box) return;
  if (!ds) { box.innerHTML = '<div class="hint" style="padding:10px">no dataset</div>'; return; }
  let eps = [];
  try { ({ episodes: eps } = await api('GET', `/api/datasets/${encodeURIComponent(ds)}/episodes`)); }
  catch (e) { log('Failed to list episodes: ' + e, 'error'); return; }
  episodes.renderList(box, eps, { manage: true, editingPath, confirmDeletePath, cloudProgress: cloudProg });
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

async function stopRecording() {
  try { await api('POST', '/api/record/stop'); log('Stopping recording…'); }
  catch (e) { log('Stop failed: ' + e, 'error'); }
}

async function record() {
  const body = {
    dataset: currentDataset(),
    task: view.querySelector('#rec-task').value,
    demonstrator: view.querySelector('#rec-demo').value,
    seconds: +view.querySelector('#rec-seconds').value || 10,
    fps: +view.querySelector('#rec-fps').value || 15,
    cloud: cloudOpts(),
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
    if (j.status === 'done') {
      log(`Recorded → ${j.result?.episode}`, 'ok');
      loadEpisodes(); loadDatasets();
      const epId = (j.result?.episode || '').split('/').filter(Boolean).pop();
      watchCloud(epId);
    }
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
    case 'rec-go': recording ? stopRecording() : record(); break;
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
  else if (['rec-cl-stride', 'rec-cl-voxel', 'rec-cl-max', 'rec-cl-bbox', 'rec-cl-bg'].includes(e.target.id)) persistCloud();
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

// ── point-cloud params + post-capture build progress ─────────────────────

function cloudOpts() {
  const bbox = (view.querySelector('#rec-cl-bbox').value || '')
    .split(',').map(s => parseFloat(s.trim())).filter(n => !Number.isNaN(n));
  return {
    stride: Math.max(1, +view.querySelector('#rec-cl-stride').value || 1),
    voxel: Math.max(0, +view.querySelector('#rec-cl-voxel').value || 0),
    max_points: Math.max(0, +view.querySelector('#rec-cl-max').value || 0),
    bbox: bbox.length === 6 ? bbox : null,
    bg_subtract: view.querySelector('#rec-cl-bg').checked,
  };
}

function persistCloud() {
  const o = cloudOpts();
  sessionPatch('cloud', {
    stride: o.stride, voxel: o.voxel, maxPoints: o.max_points,
    bbox: o.bbox || [], bgSubtract: o.bg_subtract,
  });
}

// Poll the job queue for this episode's cloud build and draw a progress bar
// inline on that episode's row (under the name, next to edit / del). Stops
// itself on done / error.
function watchCloud(episodeId) {
  if (!episodeId) return;
  if (cloudPoll) clearInterval(cloudPoll);
  cloudProg = { id: episodeId, status: 'queued', pct: 0, label: 'cloud queued…' };
  loadEpisodes();   // re-render the list so the fresh row carries the bar
  cloudPoll = setInterval(async () => {
    let jobs;
    try { ({ jobs } = await api('GET', '/api/pipeline/jobs')); } catch { return; }
    const j = jobs.find(x => x.kind === 'cloud' && x.episode === episodeId);
    if (!j) return;
    const p = j.progress || {};
    const pct = p.total ? Math.round(100 * (p.frame || 0) / p.total) : 0;
    cloudProg = {
      id: episodeId,
      status: j.status,
      pct,
      label: j.status === 'queued' ? `cloud queued #${j.queue_pos ?? ''}`
        : j.status === 'running' ? `cloud ${p.frame || 0}/${p.total || '?'}`
          : `cloud ${j.status}`,
    };
    paintCloudBar();
    if (j.status === 'done' || j.status === 'error') {
      clearInterval(cloudPoll); cloudPoll = null;
      const ok = j.status === 'done';
      cloudProg = null;
      loadEpisodes();   // drop the bar, refresh stage badges
      ok ? log(`Cloud built for ${episodeId}`, 'ok')
        : log(`Cloud build failed for ${episodeId}: ${j.error}`, 'error');
    }
  }, 1000);
}

// Repaint just the one row's bar — no full list re-render, so hover / an
// in-progress edit elsewhere in the list survive the 1 s tick.
function paintCloudBar() {
  if (!cloudProg || !view) return;
  // Not on screen (other dataset) or that row is mid-edit (no bar slot) — skip;
  // the next full loadEpisodes() re-renders the bar from cloudProg.
  const bar = view.querySelector(`.episode-row[data-id="${cloudProg.id}"] .ep-cloud`);
  if (!bar) return;
  const pct = cloudProg.status === 'done' ? 100 : (cloudProg.status === 'running' ? cloudProg.pct : 0);
  bar.className = `ep-cloud ${cloudProg.status}`;
  bar.querySelector('i').style.width = pct + '%';
  bar.querySelector('.ep-cloud-lbl').textContent = cloudProg.label;
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
  if (cloudPoll) { clearInterval(cloudPoll); cloudPoll = null; }
  cloudProg = null;
  view?.querySelectorAll('.streams img').forEach(img => { img.src = ''; });
  recording = false;
  cameras.setRecording(false);
  view = null;
}
