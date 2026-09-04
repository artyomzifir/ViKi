// Extract tab — the whole perception stage: pick a model + options, send one /
// several / a whole dataset of episodes to a background queue, and inspect the
// result (per-camera + fused hand skeletons, cloud, trajectory) in the shared
// scene3d viewer on the left.
import { api, log, sessionGet, sessionSet, sessionPatch } from './core.js';
import * as scene3d from './scene3d.js';
import * as episodes from './episodes.js';

const LM_NAMES = [
  'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
];
const REQUIRED_LM = new Set([0, 5, 9, 17, 4, 8]);   // EE-pose + gripper need these
const DEFAULT_LM = [...Array(21).keys()];           // track every landmark by default
const CLEAN_BASELINE = 'clean-triangulated-landmarks-v1';
const LAYER_LABELS = {
  cloud: 'cloud', perCamera: 'per-camera', fused: 'fused', trajectory: 'traj',
  palm: 'palm+grip', frusta: 'frusta', board: 'board', bbox: 'bbox', handFit: 'hand fit',
};

// 21-point hand diagram, palm toward you, fingers up. [x, y] in a 0..100 box.
const HAND_XY = [
  [50, 94],                                  // 0 wrist
  [33, 79], [23, 66], [16, 55], [10, 45],    // 1-4 thumb
  [42, 56], [40, 41], [39, 30], [38, 19],    // 5-8 index
  [51, 53], [51, 37], [51, 24], [51, 12],    // 9-12 middle
  [61, 55], [63, 40], [64, 29], [65, 19],    // 13-16 ring
  [70, 61], [74, 49], [77, 40], [79, 32],    // 17-20 pinky
];
const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const DEFAULT_OPTS = {
  profile: CLEAN_BASELINE, model: 'mediapipe', hand: 'right', flip: false,
  track_lm: DEFAULT_LM, min_confidence: 0.5, interp_max_gap: 0,
  sg_window: 7, sg_polyorder: 2,
  regen_cloud: false, cloud_stride: 1, cloud_bbox: '', dataset: '',
};

let root = null, ctl = null, models = {}, epList = [], poll = 0, viewedEp = null;

export function mount(view) {
  const S = { ...DEFAULT_OPTS, ...sessionGet('perceive', {}) };
  root = document.createElement('div');
  root.className = 'perception-tab';
  root.innerHTML = `
    <aside class="perc-runcol">
      <div class="calib-sec-title">4 · Run</div>
      <div class="cfg-row"><label>Dataset</label><select data-role="dataset"></select></div>
      <div class="perc-eps" data-role="eps"></div>
      <label class="perc-all"><input type="checkbox" data-role="all"> select all</label>
      <label class="perc-all" title="rebuild the point cloud too (e.g. to widen the workspace AABB below); off by default — the cloud is already built right after recording">
        <input type="checkbox" data-role="regen-cloud" ${S.regen_cloud ? 'checked' : ''}> regenerate cloud</label>
      <button class="primary" data-role="process">Process</button>
    </aside>

    <div class="perc-viewer">
      <div class="viewer-canvas" data-role="canvas"></div>
      <div class="perc-overlay perc-overlay-layers" data-role="layers"></div>
      <div class="perc-overlay perc-overlay-transport">
        <button data-role="stop" title="stop">■</button>
        <button data-role="prev" title="prev frame">◄</button>
        <button data-role="play" title="play / pause">▶</button>
        <button data-role="next" title="next frame">►</button>
        <button data-role="back5" title="back 5 seconds">«5s</button>
        <button data-role="fwd5" title="forward 5 seconds">5s»</button>
        <input type="range" data-role="time" min="0" max="0" value="0" step="1">
        <span data-role="frame-lbl">0 / 0</span>
        <button data-role="prevep" title="previous episode">‹</button>
        <button data-role="nextep" title="next episode">›</button>
      </div>
    </div>

    <aside class="perc-side">
      <section class="calib-sec">
        <div class="calib-sec-title">1 · Model</div>
        <div class="cfg-row"><label>Pipeline</label>
          <select data-role="profile">
            <option value="${CLEAN_BASELINE}" ${S.profile === CLEAN_BASELINE ? 'selected' : ''}>clean baseline v1</option>
            <option value="" ${!S.profile ? 'selected' : ''}>custom / config</option>
          </select></div>
        <div class="hint" data-role="profile-meta"></div>
        <select data-role="model"></select>
        <div class="hint" data-role="model-meta">—</div>
        <button data-role="download" hidden>Download weights</button>
        <div class="cfg-row"><label>Hand</label>
          <select data-role="hand"><option>right</option><option>left</option></select></div>
        <div class="cfg-row"><label>Flip for handedness</label>
          <input type="checkbox" data-role="flip" ${S.flip ? 'checked' : ''}></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">2 · Track landmarks</div>
        <div class="hint">all 21 tracked by default — click a joint to drop it (locked = needed for EE pose + gripper)</div>
        <svg class="perc-hand" data-role="handmap" viewBox="0 0 100 105"></svg>
        <div class="hint" data-role="track-sum">21 / 21</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">3 · Parameters</div>
        <div class="cfg-row"><label>Min confidence</label>
          <input type="number" data-role="minconf" min="0" max="1" step="0.05" value="${S.min_confidence}"></div>
        <div class="cfg-row"><label>Interp max gap</label>
          <input type="number" data-role="gap" min="0" value="${S.interp_max_gap}"></div>
        <div class="cfg-row"><label>SG window</label>
          <input type="number" data-role="sgwin" min="3" step="2" value="${S.sg_window}"></div>
        <div class="cfg-row"><label>SG polyorder</label>
          <input type="number" data-role="sgpoly" min="1" value="${S.sg_polyorder}"></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Cloud (only if "regenerate cloud")</div>
        <div class="rec-field"><label>Workspace AABB <span class="hint">— x0,x1,y0,y1,z0,z1 (m); empty = full scene</span></label>
          <input type="text" data-role="cloud-bbox" placeholder="-0.8,0.8,-0.8,0.8,-0.8,1.2" value="${S.cloud_bbox || ''}"></div>
        <div class="cfg-row"><label>Depth stride</label>
          <input type="number" data-role="stride" min="1" max="12" value="${S.cloud_stride}"></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Queue</div>
        <div class="perc-queue" data-role="queue"></div>
      </section>
    </aside>`;
  view.appendChild(root);

  ctl = scene3d.create(root.querySelector('[data-role="canvas"]'), {
    api, log, layers: sessionGet('viewerLayers', null),
  });
  ctl.onFrame((f, n) => {
    root.querySelector('[data-role="time"]').max = Math.max(0, n - 1);
    root.querySelector('[data-role="time"]').value = f;
    root.querySelector('[data-role="frame-lbl"]').textContent = `${n ? f + 1 : 0} / ${n}`;
  });

  renderHand(S.track_lm);
  renderLayers();
  root.addEventListener('click', onClick);
  root.addEventListener('change', onChange);
  root.addEventListener('input', onInput);

  loadModels(S);
  loadDatasets(S.dataset);
  refreshQueue();
  poll = setInterval(refreshQueue, 1500);
}

export function unmount() {
  clearInterval(poll); poll = 0;
  ctl?.dispose(); ctl = null;
  root?.remove(); root = null;
}

// ── model + track UI ──────────────────────────────────────────────────

async function loadModels(S) {
  try { models = await api('GET', '/api/pipeline/models'); }   // flat list
  catch (e) { log('models: ' + e, 'error'); return; }
  const sel = root.querySelector('[data-role="model"]');
  sel.innerHTML = models.map(m =>
    `<option value="${m.id}">${m.label}${m.present ? '' : ' — not downloaded'}</option>`).join('');
  const want = (S && S.model) || 'mediapipe';
  if (models.some(m => m.id === want)) sel.value = want;
  root.querySelector('[data-role="hand"]').value = S?.hand || 'right';
  syncProfile();
}

function syncModel() {
  const id = root.querySelector('[data-role="model"]').value;
  const m = models.find(x => x.id === id) || {};
  const bits = [
    m.pck != null ? `PCK@0.2 ${m.pck}` : null,
    m.auc != null ? `AUC ${m.auc}` : null,
    m.epe != null ? `EPE ${m.epe}px` : null,
    m.gflops != null ? `${m.gflops} GFLOPs` : null,
    m.license,
  ].filter(Boolean).join(' · ');
  root.querySelector('[data-role="model-meta"]').textContent =
    (m.note ? m.note + ' — ' : '') + bits;
  const dl = root.querySelector('[data-role="download"]');
  if (m.present) { dl.hidden = true; }
  else {
    dl.hidden = false;
    dl.disabled = !m.downloadable;
    dl.textContent = m.downloadable ? 'Download weights'
      : 'ONNX not published — convert with mmdeploy → models/';
  }
  persist();
}

function syncProfile() {
  const profile = root.querySelector('[data-role="profile"]').value;
  const locked = profile === CLEAN_BASELINE;
  const fixed = {
    model: 'mediapipe', flip: false, minconf: 0.5, gap: 0, sgwin: 7, sgpoly: 2,
  };
  if (locked) {
    Object.entries(fixed).forEach(([role, value]) => {
      const el = root.querySelector(`[data-role="${role}"]`);
      if (!el) return;
      if (el.type === 'checkbox') el.checked = !!value;
      else el.value = String(value);
    });
    renderHand(DEFAULT_LM);
  }
  ['model', 'flip', 'minconf', 'gap', 'sgwin', 'sgpoly'].forEach(role => {
    const el = root.querySelector(`[data-role="${role}"]`);
    if (el) el.disabled = locked;
  });
  root.querySelector('[data-role="profile-meta"]').textContent = locked
    ? 'locked: MediaPipe · all 21 · triangulate · fill all · SG 7/2 · landmarks · no hand fit'
    : 'experimental settings below are used directly';
  syncModel();
}

function renderHand(sel) {
  const set = new Set([...(sel || []), ...REQUIRED_LM]);
  const line = ([a, b]) =>
    `<line x1="${HAND_XY[a][0]}" y1="${HAND_XY[a][1]}" x2="${HAND_XY[b][0]}" y2="${HAND_XY[b][1]}"/>`;
  const dot = (i) => {
    const req = REQUIRED_LM.has(i);
    const cls = `${set.has(i) ? 'on' : ''} ${req ? 'req' : ''}`.trim();
    return `<circle data-lm="${i}" class="${cls}" cx="${HAND_XY[i][0]}" cy="${HAND_XY[i][1]}" r="3.2">
      <title>${i} ${LM_NAMES[i]}${req ? ' (locked)' : ''}</title></circle>`;
  };
  root.querySelector('[data-role="handmap"]').innerHTML =
    `<g class="hand-edges">${HAND_EDGES.map(line).join('')}</g>` +
    `<g class="hand-dots">${HAND_XY.map((_, i) => dot(i)).join('')}</g>`;
  _trackSel = [...set].sort((a, b) => a - b);
  updateTrackSummary();
}

let _trackSel = DEFAULT_LM.slice();

function toggleLm(i) {
  if (root.querySelector('[data-role="profile"]').value === CLEAN_BASELINE) return;
  if (REQUIRED_LM.has(i)) return;
  const s = new Set(_trackSel);
  s.has(i) ? s.delete(i) : s.add(i);
  renderHand([...s]);
  persist();
}

function trackSelection() { return _trackSel.slice(); }

function updateTrackSummary() {
  root.querySelector('[data-role="track-sum"]').textContent =
    `${_trackSel.length} / 21 tracked`;
}

function renderLayers() {
  const st = ctl.layerState;
  root.querySelector('[data-role="layers"]').innerHTML = Object.entries(LAYER_LABELS).map(([k, l]) =>
    `<label><input type="checkbox" data-layer="${k}" ${st[k] ? 'checked' : ''}> ${l}</label>`).join('');
}

// mirror the current form into the session store so leaving the tab and coming
// back keeps every setting.
function persist() {
  if (!root) return;
  const o = opts();
  sessionSet('perceive', {
    ...o,
    regen_cloud: o.build_cloud,
    cloud_bbox: root.querySelector('[data-role="cloud-bbox"]').value || '',
    model: root.querySelector('[data-role="model"]').value || '',
    dataset: root.querySelector('[data-role="dataset"]').value || '',
  });
}

// ── datasets + episodes ───────────────────────────────────────────────

async function loadDatasets(wantDs) {
  try {
    const { datasets } = await api('GET', '/api/datasets');
    const sel = root.querySelector('[data-role="dataset"]');
    sel.innerHTML = datasets.map(d => `<option value="${d.name}">${d.name} (${d.episodes})</option>`).join('');
    if (wantDs && datasets.some(d => d.name === wantDs)) sel.value = wantDs;
    persist();
    loadEpisodes();
  } catch (e) { log('datasets: ' + e, 'error'); }
}

async function loadEpisodes() {
  const ds = root.querySelector('[data-role="dataset"]').value;
  if (!ds) return;
  try {
    ({ episodes: epList } = await api('GET', `/api/datasets/${encodeURIComponent(ds)}/episodes`));
  } catch (e) { log('episodes: ' + e, 'error'); return; }
  // Same row widget as the Record tab, plus a select checkbox (queue) and a
  // view button (scene3d). Keep whatever was already ticked across a reload.
  const keep = new Set(selectedEpisodes());
  episodes.renderList(root.querySelector('[data-role="eps"]'), epList, {
    select: true, view: true, selected: keep, activeId: viewedEp,
    emptyText: 'no episodes',
  });
  syncAllCheckbox();
}

function markViewed(id) {
  viewedEp = id;
  root?.querySelectorAll('.episode-row').forEach(r =>
    r.classList.toggle('active', r.dataset.id === id));
}

function selectedEpisodes() {
  return [...root.querySelectorAll('[data-ep]:checked')].map(c => c.dataset.ep);
}

function syncAllCheckbox() {
  const all = root?.querySelector('[data-role="all"]');
  if (!all) return;
  const boxes = [...root.querySelectorAll('[data-ep]')];
  all.checked = boxes.length > 0 && boxes.every(c => c.checked);
}

// ── run + queue ───────────────────────────────────────────────────────

function opts() {
  const bbox = (root.querySelector('[data-role="cloud-bbox"]').value || '')
    .split(',').map(s => parseFloat(s.trim())).filter(n => !Number.isNaN(n));
  return {
    profile: root.querySelector('[data-role="profile"]').value || null,
    model: root.querySelector('[data-role="model"]').value || 'rtmpose-m-hand5',
    hand: root.querySelector('[data-role="hand"]').value,
    flip: root.querySelector('[data-role="flip"]').checked,
    track_lm: trackSelection(),
    min_confidence: +root.querySelector('[data-role="minconf"]').value,
    interp_max_gap: +root.querySelector('[data-role="gap"]').value,
    sg_window: +root.querySelector('[data-role="sgwin"]').value,
    sg_polyorder: +root.querySelector('[data-role="sgpoly"]').value,
    build_cloud: root.querySelector('[data-role="regen-cloud"]').checked,
    cloud_stride: +root.querySelector('[data-role="stride"]').value,
    cloud_bbox: bbox.length === 6 ? bbox : null,
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
  const model = root.querySelector('[data-role="model"]').value;
  try {
    await api('POST', '/api/pipeline/models/download', { model });
    log(`Downloading ${model}…`, 'ok');
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
  const dot = e.target.closest('[data-lm]');
  if (dot) { toggleLm(+dot.dataset.lm); return; }
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.view) {
    ctl.loadEpisode(b.dataset.view, epList);
    markViewed(b.dataset.view);
    return;
  }
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
  if (el.dataset.role === 'profile') syncProfile();
  else if (el.dataset.role === 'model') syncModel();
  else if (el.dataset.role === 'dataset') { persist(); loadEpisodes(); }
  else if (el.dataset.layer) {
    ctl.setLayer(el.dataset.layer, el.checked);
    sessionPatch('viewerLayers', { [el.dataset.layer]: el.checked });
  }
  else if (el.dataset.role === 'all') {
    root.querySelectorAll('[data-ep]').forEach(c => { c.checked = el.checked; });
  }
  else if (el.dataset.ep) syncAllCheckbox();   // a row checkbox toggled
  else if (el.dataset.role) persist();   // any other param widget
}

function onInput(e) {
  const el = e.target;
  if (el.dataset.role === 'time') ctl.setFrame(+el.value);
  else if (el.dataset.role) persist();   // number fields fire input while typing
}
