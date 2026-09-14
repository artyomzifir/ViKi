// Viewer tab — a thin shell around the shared scene3d controller: episode
// picker, cloud colour + stride, a timeline and transport. Layer toggles live
// in the scene's own legend overlay (scene3d.js). The Extract tab reuses it.
import { api, log, sessionGet, sessionSet, sessionPatch } from './core.js';
import * as scene3d from './scene3d.js';

let root = null, ctl = null, episodes = [], variants = [];

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
      <label class="viewer-field">Stage variant
        <select data-role="variant"><option value="active">active</option></select>
      </label>
      <div class="viewer-status" data-role="status">pick an episode</div>
      <div class="viewer-object-status" data-role="object-status" hidden></div>
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
      <p class="viewer-help">layers: click a row in the scene legend<br>drag orbit · wheel zoom · right-drag pan</p>
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
    objectModels: true,
  });

  ctl.onFrame((f, n, objectSummary) => {
    $('[data-role="time"]').max = Math.max(0, n - 1);
    $('[data-role="time"]').value = f;
    $('[data-role="frame-lbl"]').textContent = `${n ? f + 1 : 0} / ${n}`;
    renderObjectStatus(objectSummary);
  });
  ctl.onLayerChange(l => sessionSet('viewerLayers', l));

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
  if (!id) {
    status.textContent = 'pick an episode'; build.hidden = true;
    renderObjectStatus(null); return;
  }
  status.textContent = 'loading…';
  try {
    const response = await api('GET', `/api/pipeline/episode/${id}/geometry/variants`);
    variants = response.variants || [];
  } catch { variants = [{ id: 'active', label: 'active' }]; }
  const variantPicker = root.querySelector('[data-role="variant"]');
  variantPicker.innerHTML = variants.map(v =>
    `<option value="${v.id}">${v.label}</option>`).join('');
  variantPicker.value = variants.some(v => v.id === 'active') ? 'active' : (variants[0]?.id || 'active');
  await openVariant(id, variantPicker.value);
}

async function openVariant(id, variant) {
  const status = root.querySelector('[data-role="status"]');
  const build = root.querySelector('[data-role="build"]');
  status.textContent = 'loading variant…';
  const r = await ctl.loadEpisode(id, episodes, variant);
  setPlayIcon(false);
  build.hidden = r.hasCloud;
  const g = r.geo || {};
  const source = [g.fusion_mode, g.checkpoint_stage, g.pose_source].filter(Boolean).join(' · ');
  const sourceLabel = source ? ` · ${source}` : '';
  const objectLabel = r.hasObjectModel
    ? ` · object model ${r.ometa.objects.length}` : '';
  status.textContent = r.hasCloud
    ? `${r.cmeta.n_frames} cloud frames · ${g.n_frames || 0} traj frames · fps ${(ctl.fps).toFixed(1)}${sourceLabel}${objectLabel}`
    : (g.n_frames
      ? `${g.n_frames} traj frames · no point cloud${sourceLabel}${objectLabel}`
      : (r.hasObjectModel ? `object model ${r.ometa.objects.length} · no point cloud` : 'not processed yet — run perception'));
}

function renderObjectStatus(summary) {
  const box = root?.querySelector('[data-role="object-status"]');
  if (!box) return;
  box.replaceChildren();
  box.hidden = !summary?.objects?.length;
  if (box.hidden) return;
  const head = document.createElement('div');
  head.className = 'viewer-object-title';
  head.textContent = `Object model · ${summary.profile}`;
  box.appendChild(head);
  const number = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  for (const object of summary.objects) {
    const row = document.createElement('div');
    row.className = 'viewer-object-row';
    const title = document.createElement('strong');
    title.textContent = `#${object.id} ${object.label}${object.contact ? ' · contact' : ''}`;
    const metrics = document.createElement('span');
    const rotMin = Array.isArray(object.rotation_information)
      ? Math.min(...object.rotation_information.filter(Number.isFinite)) : NaN;
    metrics.textContent = [
      `conf ${number(object.confidence)}`,
      `res ${number(object.residual_median_m * 1000, 1)}/${number(object.residual_p95_m * 1000, 1)} mm`,
      `keep ${number(object.retained_fraction * 100, 0)}%`,
      `rot-info ${number(rotMin)}`,
    ].join(' · ');
    row.append(title, metrics);
    box.appendChild(row);
  }
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
  else if (el.dataset.role === 'variant') {
    const id = root.querySelector('[data-role="picker"]').value;
    if (id) openVariant(id, el.value);
  }
  else if (el.dataset.role === 'color') {
    ctl.setColorMode(el.value); sessionPatch('viewer', { color: el.value });
  }
}

function onInput(e) {
  const el = e.target;
  if (el.dataset.role === 'time') ctl.setFrame(+el.value);
  else if (el.dataset.role === 'stride') {
    ctl.setStride(+el.value); sessionPatch('viewer', { stride: +el.value });
  }
}
