// Viewer tab — a thin shell around the shared scene3d controller: episode
// picker, layer toggles, cloud colour + stride, a timeline and transport.
// The Extract tab reuses the same scene3d module.
import { api, log, sessionGet, sessionSet, sessionPatch } from './core.js';
import * as scene3d from './scene3d.js';

let root = null, ctl = null, episodes = [];

const LAYER_LABELS = {
  cloud: 'point cloud', perCamera: 'per-camera skeletons', fused: 'fused skeleton',
  trajectory: 'wrist trajectory', palm: 'palm + gripper', frusta: 'camera frusta',
  board: 'ChArUco board', bbox: 'workspace box',
};

export function mount(view) {
  const vs = sessionGet('viewer', { color: 'rgb', stride: 1 });
  root = document.createElement('div');
  root.className = 'viewer-tab';
  root.innerHTML = `
    <div class="viewer-canvas" data-role="canvas"></div>
    <aside class="viewer-side">
      <label class="viewer-field">Episode
        <select data-role="picker"></select>
      </label>
      <div class="viewer-status" data-role="status">pick an episode</div>
      <button class="viewer-build" data-role="build" hidden>Run perception</button>
      <div class="viewer-field">Cloud colour
        <select data-role="color">
          <option value="rgb" ${vs.color === 'rgb' ? 'selected' : ''}>real colour</option>
          <option value="height" ${vs.color === 'height' ? 'selected' : ''}>by height</option>
        </select>
      </div>
      <label class="viewer-field">Cloud stride
        <input type="number" data-role="stride" min="1" max="12" value="${vs.stride || 1}">
      </label>
      <div class="viewer-layers" data-role="layers"></div>
      <p class="viewer-help">drag orbit · wheel zoom · right-drag pan</p>
    </aside>
    <div class="viewer-timeline">
      <button data-role="stop" title="stop">■</button>
      <button data-role="prev" title="prev frame">◄</button>
      <button data-role="play" title="play / pause">▶</button>
      <button data-role="next" title="next frame">►</button>
      <button data-role="back5" title="back 5 seconds">«5s</button>
      <button data-role="fwd5" title="forward 5 seconds">5s»</button>
      <input type="range" data-role="time" min="0" max="0" value="0" step="1">
      <span data-role="frame-lbl">0 / 0</span>
      <button data-role="prevep" title="previous episode">‹ ep</button>
      <button data-role="nextep" title="next episode">ep ›</button>
    </div>`;
  view.appendChild(root);

  const $ = s => root.querySelector(s);
  ctl = scene3d.create($('[data-role="canvas"]'), {
    api, log, layers: sessionGet('viewerLayers', null),
    colorMode: vs.color, stride: vs.stride,
  });

  ctl.onFrame((f, n) => {
    $('[data-role="time"]').max = Math.max(0, n - 1);
    $('[data-role="time"]').value = f;
    $('[data-role="frame-lbl"]').textContent = `${n ? f + 1 : 0} / ${n}`;
  });

  renderLayers();
  root.addEventListener('click', onClick);
  root.addEventListener('change', onChange);
  root.addEventListener('input', onInput);
  loadEpisodes();
}

export function unmount() {
  ctl?.dispose();
  ctl = null;
  root?.remove();
  root = null;
}

function renderLayers() {
  const box = root.querySelector('[data-role="layers"]');
  const st = ctl.layerState;
  box.innerHTML = Object.entries(LAYER_LABELS).map(([k, label]) =>
    `<label class="viewer-layer"><input type="checkbox" data-layer="${k}" ${st[k] ? 'checked' : ''}> ${label}</label>`
  ).join('');
}

async function loadEpisodes() {
  try {
    const { episodes: eps } = await api('GET', '/api/pipeline/episodes');
    episodes = eps || [];
    root.querySelector('[data-role="picker"]').innerHTML =
      '<option value="">—</option>' +
      episodes.map(e => `<option value="${e.id}">${e.id}${e.task ? ' · ' + e.task : ''}</option>`).join('');
  } catch (e) { log('viewer: ' + e, 'error'); }
}

async function openEpisode(id) {
  const status = root.querySelector('[data-role="status"]');
  const build = root.querySelector('[data-role="build"]');
  if (!id) { status.textContent = 'pick an episode'; build.hidden = true; return; }
  status.textContent = 'loading…';
  const r = await ctl.loadEpisode(id, episodes);
  build.hidden = r.hasCloud;
  const g = r.geo || {};
  status.textContent = r.hasCloud
    ? `${r.cmeta.n_frames} cloud frames · ${g.n_frames || 0} traj frames · fps ${(ctl.fps).toFixed(1)}`
    : (g.n_frames ? `${g.n_frames} traj frames · no point cloud` : 'not processed yet — run perception');
}

async function runPerception(id) {
  const status = root.querySelector('[data-role="status"]');
  status.textContent = 'queued perception…';
  try {
    await api('POST', '/api/pipeline/perceive', { episodes: [id], opts: { build_cloud: true } });
    log(`Perception queued for ${id} — watch the Extract tab`, 'ok');
  } catch (e) { log('perceive: ' + e, 'error'); }
}

function onClick(e) {
  const b = e.target.closest('button');
  if (!b || !ctl) return;
  const id = root.querySelector('[data-role="picker"]').value;
  switch (b.dataset.role) {
    case 'build': runPerception(id); break;
    case 'stop': ctl.stop(); setPlayIcon(); break;
    case 'play': setPlayIcon(ctl.togglePlay()); break;
    case 'prev': ctl.step(-1); break;
    case 'next': ctl.step(1); break;
    case 'back5': ctl.skipSeconds(-1); break;
    case 'fwd5': ctl.skipSeconds(1); break;
    case 'prevep': { const nid = ctl.nextEpisode(-1); if (nid) syncPicker(nid); break; }
    case 'nextep': { const nid = ctl.nextEpisode(1); if (nid) syncPicker(nid); break; }
  }
}

function setPlayIcon(playing) {
  const btn = root.querySelector('[data-role="play"]');
  if (btn) btn.textContent = (playing ?? ctl.playing) ? '❚❚' : '▶';
}

function syncPicker(id) {
  const sel = root.querySelector('[data-role="picker"]');
  sel.value = id;
  openEpisode(id);
}

function onChange(e) {
  const el = e.target;
  if (el.dataset.role === 'picker') openEpisode(el.value);
  else if (el.dataset.role === 'color') {
    ctl.setColorMode(el.value); sessionPatch('viewer', { color: el.value });
  } else if (el.dataset.layer) {
    ctl.setLayer(el.dataset.layer, el.checked);
    sessionPatch('viewerLayers', { [el.dataset.layer]: el.checked });
  }
}

function onInput(e) {
  const el = e.target;
  if (el.dataset.role === 'time') ctl.setFrame(+el.value);
  else if (el.dataset.role === 'stride') {
    ctl.setStride(+el.value); sessionPatch('viewer', { stride: +el.value });
  }
}
