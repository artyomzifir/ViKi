// Extract tab — the whole perception stage: pick a model + options, send one /
// several / a whole dataset of episodes to a background queue, and inspect the
// result (per-camera + fused hand skeletons, cloud, trajectory) in the shared
// scene3d viewer on the left.
import { api, log } from './core.js';
import * as scene3d from './scene3d.js';

const LM_NAMES = [
  'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
];
const REQUIRED_LM = new Set([0, 5, 9, 17, 4, 8]);   // EE-pose + gripper need these
const DEFAULT_LM = [0, 1, 3, 4, 5, 8, 9, 13, 17];
const LAYER_LABELS = {
  cloud: 'cloud', perCamera: 'per-camera', fused: 'fused', trajectory: 'traj',
  palm: 'palm+grip', frusta: 'frusta', board: 'board', bbox: 'bbox',
};

let root = null, ctl = null, models = {}, episodes = [], poll = 0;

export function mount(view) {
  root = document.createElement('div');
  root.className = 'perception-tab';
  root.innerHTML = `
    <div class="perc-viewer">
      <div class="viewer-canvas" data-role="canvas"></div>
      <div class="viewer-timeline">
        <button data-role="stop" title="stop">■</button>
        <button data-role="prev" title="prev frame">◄</button>
        <button data-role="play" title="play / pause">▶</button>
        <button data-role="next" title="next frame">►</button>
        <button data-role="back5" title="-5s">⏮</button>
        <button data-role="fwd5" title="+5s">⏭</button>
        <input type="range" data-role="time" min="0" max="0" value="0" step="1">
        <span data-role="frame-lbl">0 / 0</span>
        <button data-role="prevep" title="previous episode">‹</button>
        <button data-role="nextep" title="next episode">›</button>
      </div>
      <div class="perc-layers" data-role="layers"></div>
    </div>

    <aside class="perc-side">
      <section class="calib-sec">
        <div class="calib-sec-title">Model</div>
        <div class="cfg-row"><label>Backend</label><select data-role="backend"></select></div>
        <div class="cfg-row"><label>Model</label><select data-role="model"></select></div>
        <button data-role="download" hidden>Download model</button>
        <div class="cfg-row"><label>Hand</label>
          <select data-role="hand"><option>right</option><option>left</option></select></div>
        <div class="cfg-row"><label>Flip for handedness</label>
          <input type="checkbox" data-role="flip"></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Track landmarks</div>
        <details><summary data-role="track-sum">9 / 21</summary>
          <div class="perc-track" data-role="track"></div>
        </details>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Parameters</div>
        <div class="cfg-row"><label>Min confidence</label>
          <input type="number" data-role="minconf" min="0" max="1" step="0.05" value="0.5"></div>
        <div class="cfg-row"><label>Interp max gap</label>
          <input type="number" data-role="gap" min="0" value="0"></div>
        <div class="cfg-row"><label>SG window</label>
          <input type="number" data-role="sgwin" min="3" step="2" value="7"></div>
        <div class="cfg-row"><label>SG polyorder</label>
          <input type="number" data-role="sgpoly" min="1" value="2"></div>
        <div class="cfg-row"><label>Build cloud</label>
          <input type="checkbox" data-role="cloud" checked></div>
        <div class="cfg-row"><label>Cloud stride</label>
          <input type="number" data-role="stride" min="1" max="12" value="6"></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Run</div>
        <div class="cfg-row"><label>Dataset</label><select data-role="dataset"></select></div>
        <div class="perc-eps" data-role="eps"></div>
        <label class="perc-all"><input type="checkbox" data-role="all"> select all</label>
        <button class="primary" data-role="process">Process</button>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Queue</div>
        <div class="perc-queue" data-role="queue"></div>
      </section>
    </aside>`;
  view.appendChild(root);

  ctl = scene3d.create(root.querySelector('[data-role="canvas"]'), { api, log });
  ctl.onFrame((f, n) => {
    root.querySelector('[data-role="time"]').max = Math.max(0, n - 1);
    root.querySelector('[data-role="time"]').value = f;
    root.querySelector('[data-role="frame-lbl"]').textContent = `${n ? f + 1 : 0} / ${n}`;
  });

  renderTrack(DEFAULT_LM);
  renderLayers();
  root.addEventListener('click', onClick);
  root.addEventListener('change', onChange);
  root.addEventListener('input', onInput);

  loadModels();
  loadDatasets();
  refreshQueue();
  poll = setInterval(refreshQueue, 1500);
}

export function unmount() {
  clearInterval(poll); poll = 0;
  ctl?.dispose(); ctl = null;
  root?.remove(); root = null;
}

// ── model + track UI ──────────────────────────────────────────────────

async function loadModels() {
  try { models = await api('GET', '/api/pipeline/models'); }
  catch (e) { log('models: ' + e, 'error'); return; }
  const bsel = root.querySelector('[data-role="backend"]');
  bsel.innerHTML = Object.keys(models).map(b => `<option>${b}</option>`).join('');
  syncModelSelect();
}

function syncModelSelect() {
  const b = root.querySelector('[data-role="backend"]').value;
  const msel = root.querySelector('[data-role="model"]');
  const list = models[b] || [];
  msel.innerHTML = list.map(m =>
    `<option value="${m.id}">${m.tier} — ${m.id}${m.present ? '' : ' (not downloaded)'}</option>`).join('');
  syncDownloadBtn();
}

function syncDownloadBtn() {
  const b = root.querySelector('[data-role="backend"]').value;
  const id = root.querySelector('[data-role="model"]').value;
  const m = (models[b] || []).find(x => x.id === id);
  root.querySelector('[data-role="download"]').hidden = !!(m && m.present);
}

function renderTrack(sel) {
  const set = new Set(sel);
  root.querySelector('[data-role="track"]').innerHTML = LM_NAMES.map((nm, i) =>
    `<label><input type="checkbox" data-lm="${i}" ${set.has(i) || REQUIRED_LM.has(i) ? 'checked' : ''}
      ${REQUIRED_LM.has(i) ? 'disabled' : ''}> ${i} ${nm}</label>`).join('');
  updateTrackSummary();
}

function trackSelection() {
  return [...root.querySelectorAll('[data-lm]')]
    .filter(c => c.checked || REQUIRED_LM.has(+c.dataset.lm))
    .map(c => +c.dataset.lm).sort((a, b) => a - b);
}

function updateTrackSummary() {
  root.querySelector('[data-role="track-sum"]').textContent = `${trackSelection().length} / 21`;
}

function renderLayers() {
  const st = ctl.layerState;
  root.querySelector('[data-role="layers"]').innerHTML = Object.entries(LAYER_LABELS).map(([k, l]) =>
    `<label><input type="checkbox" data-layer="${k}" ${st[k] ? 'checked' : ''}> ${l}</label>`).join('');
}

// ── datasets + episodes ───────────────────────────────────────────────

async function loadDatasets() {
  try {
    const { datasets } = await api('GET', '/api/datasets');
    const sel = root.querySelector('[data-role="dataset"]');
    sel.innerHTML = datasets.map(d => `<option value="${d.name}">${d.name} (${d.episodes})</option>`).join('');
    loadEpisodes();
  } catch (e) { log('datasets: ' + e, 'error'); }
}

async function loadEpisodes() {
  const ds = root.querySelector('[data-role="dataset"]').value;
  if (!ds) return;
  try {
    ({ episodes } = await api('GET', `/api/datasets/${encodeURIComponent(ds)}/episodes`));
  } catch (e) { log('episodes: ' + e, 'error'); return; }
  root.querySelector('[data-role="eps"]').innerHTML = episodes.map(e =>
    `<label class="perc-ep"><input type="checkbox" data-ep="${e.id}">
      ${e.id}${e.task ? ' · ' + e.task : ''}
      <span class="badge ${e.has?.cln ? 'ok' : ''}">CLN</span></label>`).join('')
    || '<div class="hint">no episodes</div>';
}

function selectedEpisodes() {
  return [...root.querySelectorAll('[data-ep]:checked')].map(c => c.dataset.ep);
}

// ── run + queue ───────────────────────────────────────────────────────

function opts() {
  return {
    backend: root.querySelector('[data-role="backend"]').value,
    model: root.querySelector('[data-role="model"]').value || null,
    hand: root.querySelector('[data-role="hand"]').value,
    flip: root.querySelector('[data-role="flip"]').checked,
    track_lm: trackSelection(),
    min_confidence: +root.querySelector('[data-role="minconf"]').value,
    interp_max_gap: +root.querySelector('[data-role="gap"]').value,
    sg_window: +root.querySelector('[data-role="sgwin"]').value,
    sg_polyorder: +root.querySelector('[data-role="sgpoly"]').value,
    build_cloud: root.querySelector('[data-role="cloud"]').checked,
    cloud_stride: +root.querySelector('[data-role="stride"]').value,
  };
}

async function process() {
  const eps = selectedEpisodes();
  if (!eps.length) { log('Pick at least one episode', 'error'); return; }
  try {
    const { job_ids } = await api('POST', '/api/pipeline/perceive', { episodes: eps, opts: opts() });
    log(`Queued perception for ${eps.length} episode(s) (${job_ids.length} jobs)`, 'ok');
    refreshQueue();
  } catch (e) { log('perceive: ' + e, 'error'); }
}

async function downloadModel() {
  const backend = root.querySelector('[data-role="backend"]').value;
  const model = root.querySelector('[data-role="model"]').value;
  try {
    await api('POST', '/api/pipeline/models/download', { backend, model });
    log(`Downloading ${backend}/${model}…`, 'ok');
    refreshQueue();
  } catch (e) { log('download: ' + e, 'error'); }
}

async function refreshQueue() {
  if (!root) return;
  let jobs = [];
  try { ({ jobs } = await api('GET', '/api/pipeline/jobs')); } catch { return; }
  const box = root.querySelector('[data-role="queue"]');
  const rel = jobs.filter(j => ['perceive', 'download', 'cloud', 'extract', 'prepare'].includes(j.kind));
  box.innerHTML = rel.slice(0, 12).map(j => {
    const p = j.progress || {};
    const pct = p.total ? Math.round(100 * (p.frame || 0) / p.total) : 0;
    const label = j.status === 'queued' ? `queued #${j.queue_pos}`
      : j.status === 'running' ? `${p.stage || 'run'} ${p.frame || 0}/${p.total || '?'}`
        : j.status;
    return `<div class="perc-job ${j.status}">
      <span class="perc-job-ep">${j.episode || j.kind}</span>
      <span class="perc-job-st">${label}</span>
      <span class="perc-bar"><i style="width:${j.status === 'running' ? pct : (j.status === 'done' ? 100 : 0)}%"></i></span>
      ${j.status === 'queued' ? `<button data-cancel="${j.id}">✕</button>` : ''}
    </div>`;
  }).join('') || '<div class="hint">no jobs</div>';
}

// ── events ────────────────────────────────────────────────────────────

function onClick(e) {
  if (!ctl) return;
  const epLab = e.target.closest('.perc-ep');
  if (epLab && e.target.tagName !== 'INPUT') {
    ctl.loadEpisode(epLab.querySelector('[data-ep]').dataset.ep, episodes);
    return;
  }
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.cancel) {
    api('DELETE', `/api/pipeline/jobs/${b.dataset.cancel}`).then(refreshQueue)
      .catch(err => log('cancel: ' + err, 'error'));
    return;
  }
  switch (b.dataset.role) {
    case 'process': process(); break;
    case 'download': downloadModel(); break;
    case 'stop': ctl.stop(); setPlayIcon(); break;
    case 'play': setPlayIcon(ctl.togglePlay()); break;
    case 'prev': ctl.step(-1); break;
    case 'next': ctl.step(1); break;
    case 'back5': ctl.skipSeconds(-1); break;
    case 'fwd5': ctl.skipSeconds(1); break;
    case 'prevep': ctl.nextEpisode(-1); break;
    case 'nextep': ctl.nextEpisode(1); break;
  }
}

function setPlayIcon(playing) {
  const btn = root.querySelector('[data-role="play"]');
  if (btn) btn.textContent = (playing ?? ctl.playing) ? '❚❚' : '▶';
}

function onChange(e) {
  const el = e.target;
  if (el.dataset.role === 'backend') syncModelSelect();
  else if (el.dataset.role === 'model') syncDownloadBtn();
  else if (el.dataset.role === 'dataset') loadEpisodes();
  else if (el.dataset.lm !== undefined) updateTrackSummary();
  else if (el.dataset.layer) ctl.setLayer(el.dataset.layer, el.checked);
  else if (el.dataset.role === 'all') {
    root.querySelectorAll('[data-ep]').forEach(c => { c.checked = el.checked; });
  }
}

function onInput(e) {
  const el = e.target;
  if (el.dataset.role === 'time') ctl.setFrame(+el.value);
}
