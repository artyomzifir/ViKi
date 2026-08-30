// Viewer tab — a WebGL 3-D scene (three.js) of one episode: the per-frame
// coloured point cloud from the `cloud` stage, plus the fused wrist trajectory,
// the palm triad, and camera frusta, all in the shared ChArUco world frame.
// Orbit / wheel-zoom / right-drag-pan; a timeline scrubs + plays the frames.
//
// three.js is the one vendored dep (static/js/vendor/, resolved via the
// importmap in index.html) — a dense cloud (~10^5 points/frame) is not viable
// on a hand-rolled 2-D canvas.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { api, log } from './core.js';

let root, canvasWrap, picker, timeline, playBtn, frameLbl, statusEl, buildBtn;
let renderer, scene, camera, controls, ro, raf;
let points, trajLine, palmTriad, camGroup;
let geo = null, cmeta = null, epId = null;
let frame = 0, playing = false, lastStep = 0;
const cache = new Map();           // frame -> Promise<{xyz:Float32Array, rgb:Float32Array}>
const CACHE_CAP = 60;

// ── module contract ─────────────────────────────────────────────────────

export function mount(view) {
  root = document.createElement('div');
  root.className = 'viewer-tab';
  root.innerHTML = `
    <div class="viewer-canvas" data-role="canvas"></div>
    <aside class="viewer-side">
      <label class="viewer-field">Episode
        <select data-role="picker"></select>
      </label>
      <div class="viewer-status" data-role="status">pick an episode</div>
      <button class="viewer-build" data-role="build" hidden>Build cloud</button>
      <dl class="viewer-hud" data-role="hud"></dl>
      <p class="viewer-help">drag orbit · wheel zoom · right-drag pan</p>
    </aside>
    <div class="viewer-timeline">
      <button data-role="play">▶</button>
      <input type="range" data-role="time" min="0" max="0" value="0" step="1">
      <span data-role="frame-lbl">0 / 0</span>
    </div>`;
  view.appendChild(root);

  canvasWrap = root.querySelector('[data-role="canvas"]');
  picker = root.querySelector('[data-role="picker"]');
  statusEl = root.querySelector('[data-role="status"]');
  buildBtn = root.querySelector('[data-role="build"]');
  timeline = root.querySelector('[data-role="time"]');
  playBtn = root.querySelector('[data-role="play"]');
  frameLbl = root.querySelector('[data-role="frame-lbl"]');

  picker.addEventListener('change', () => loadEpisode(picker.value));
  buildBtn.addEventListener('click', buildCloud);
  timeline.addEventListener('input', () => setFrame(+timeline.value));
  playBtn.addEventListener('click', togglePlay);

  initThree();
  ro = new ResizeObserver(onResize);
  ro.observe(canvasWrap);
  loop();
  loadEpisodes();
}

export function unmount() {
  playing = false;
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
  ro?.disconnect();
  cache.clear();
  controls?.dispose();
  disposeScene();
  renderer?.dispose();
  renderer?.forceContextLoss?.();
  renderer = scene = camera = controls = null;
  root?.remove();
  root = null;
}

// ── three setup ─────────────────────────────────────────────────────────

function initThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d10);

  camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  camera.position.set(0.6, -0.6, 0.6);
  camera.up.set(0, 0, 1);                       // ChArUco Z is "up-ish"

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  canvasWrap.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AxesHelper(0.2));
  const grid = new THREE.GridHelper(2, 20, 0x333333, 0x1c1c22);
  grid.rotation.x = Math.PI / 2;                // grid in the world XY plane
  scene.add(grid);

  points = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.006, vertexColors: true, sizeAttenuation: true })
  );
  points.frustumCulled = false;
  scene.add(points);

  trajLine = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xcfcfcf })
  );
  scene.add(trajLine);

  palmTriad = new THREE.AxesHelper(0.06);
  palmTriad.visible = false;
  scene.add(palmTriad);

  camGroup = new THREE.Group();
  scene.add(camGroup);

  onResize();
}

function disposeScene() {
  scene?.traverse(o => {
    o.geometry?.dispose?.();
    if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
    else o.material?.dispose?.();
  });
}

function onResize() {
  if (!renderer || !canvasWrap) return;
  const w = canvasWrap.clientWidth || 1, h = canvasWrap.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function loop() {
  raf = requestAnimationFrame(loop);
  if (playing && cmeta && cmeta.n_frames > 1) {
    const dt = performance.now() - lastStep;
    if (dt >= 1000 / (cmeta.fps || geo?.fps || 15)) {
      lastStep = performance.now();
      setFrame((frame + 1) % cmeta.n_frames);
    }
  }
  controls?.update();
  renderer?.render(scene, camera);
}

// ── data ────────────────────────────────────────────────────────────────

async function loadEpisodes() {
  try {
    const { episodes } = await api('GET', '/api/pipeline/episodes');
    picker.innerHTML = '<option value="">—</option>' + (episodes || [])
      .map(e => `<option value="${e.id}">${e.id}${e.task ? ' · ' + e.task : ''}</option>`)
      .join('');
  } catch (e) {
    log('viewer: ' + e, 'error');
  }
}

async function loadEpisode(id) {
  epId = id;
  cache.clear();
  cmeta = null;
  buildBtn.hidden = true;
  if (!id) { statusEl.textContent = 'pick an episode'; return; }
  statusEl.textContent = 'loading…';
  try {
    geo = await api('GET', `/api/pipeline/episode/${id}/geometry?include_raw=0`);
  } catch (e) { geo = null; log('viewer geometry: ' + e, 'error'); }

  drawFrusta();
  drawTrajectory();

  try {
    cmeta = await api('GET', `/api/pipeline/episode/${id}/cloud`);
  } catch {
    cmeta = null;
  }

  if (cmeta) {
    timeline.max = Math.max(0, cmeta.n_frames - 1);
    frameCamera(cmeta.bounds);
    statusEl.textContent = `${cmeta.n_frames} frames · ${(cmeta.per_frame_points?.[0] || 0)} pts · voxel ${cmeta.voxel} m`;
    setFrame(0);
  } else {
    timeline.max = geo?.n_frames ? geo.n_frames - 1 : 0;
    statusEl.textContent = 'no point cloud for this episode yet';
    buildBtn.hidden = false;
    clearPoints();
    setFrame(0);
  }
  renderHud();
}

async function buildCloud() {
  if (!epId) return;
  buildBtn.disabled = true;
  statusEl.textContent = 'building cloud… (this can take a minute)';
  try {
    await api('POST', '/api/pipeline/cloud', { episode: epId });
    const t0 = Date.now();
    while (Date.now() - t0 < 10 * 60 * 1000) {
      await new Promise(r => setTimeout(r, 2000));
      try { cmeta = await api('GET', `/api/pipeline/episode/${epId}/cloud`); break; }
      catch { /* not ready */ }
    }
  } catch (e) {
    log('viewer build: ' + e, 'error');
  }
  buildBtn.disabled = false;
  if (cmeta) { buildBtn.hidden = true; loadEpisode(epId); }
  else statusEl.textContent = 'build failed — check the log';
}

function fetchFrame(i) {
  if (cache.has(i)) return cache.get(i);
  const p = fetch(`/api/pipeline/episode/${epId}/cloud/${i}`)
    .then(r => { if (!r.ok) throw new Error('cloud ' + i + ': ' + r.status); return r.arrayBuffer(); })
    .then(buf => {
      const n = new DataView(buf).getInt32(0, true);
      const xyz = new Float32Array(buf.slice(4, 4 + n * 12));
      const rgbU8 = new Uint8Array(buf, 4 + n * 12, n * 3);
      const rgb = new Float32Array(n * 3);
      for (let k = 0; k < rgb.length; k++) rgb[k] = rgbU8[k] / 255;
      return { xyz, rgb };
    });
  cache.set(i, p);
  if (cache.size > CACHE_CAP) cache.delete(cache.keys().next().value);
  return p;
}

// ── per-frame render ────────────────────────────────────────────────────

function setFrame(i) {
  frame = i;
  timeline.value = i;
  frameLbl.textContent = `${i + 1} / ${(cmeta?.n_frames || geo?.n_frames || 0)}`;
  updatePalm();
  if (!cmeta || !epId) return;
  for (let d = 0; d <= 3; d++) {                // warm a small prefetch window
    if (i + d < cmeta.n_frames) fetchFrame(i + d);
  }
  const want = i;
  fetchFrame(i).then(({ xyz, rgb }) => {
    if (want !== frame || !points) return;
    const g = points.geometry;
    g.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
    g.setAttribute('color', new THREE.BufferAttribute(rgb, 3));
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    g.setDrawRange(0, xyz.length / 3);
    g.computeBoundingSphere();
  }).catch(e => log('' + e, 'error'));
}

function clearPoints() {
  points.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
  points.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
}

function updatePalm() {
  const T = geo?.wrist_traj, R = geo?.palm_rot;
  if (!T || !R || !T[frame] || !R[frame]) { palmTriad.visible = false; return; }
  const o = T[frame], m = R[frame];             // m is row-major 3x3
  palmTriad.visible = true;
  palmTriad.position.set(o[0], o[1], o[2]);
  const basis = new THREE.Matrix4().set(
    m[0], m[1], m[2], 0,
    m[3], m[4], m[5], 0,
    m[6], m[7], m[8], 0,
    0, 0, 0, 1
  );
  palmTriad.quaternion.setFromRotationMatrix(basis);
}

function drawTrajectory() {
  const T = geo?.wrist_traj || [];
  const arr = new Float32Array(T.length * 3);
  T.forEach((p, i) => { arr[i * 3] = p[0]; arr[i * 3 + 1] = p[1]; arr[i * 3 + 2] = p[2]; });
  trajLine.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  trajLine.geometry.setDrawRange(0, T.length);
  trajLine.geometry.computeBoundingSphere();
}

function drawFrusta() {
  while (camGroup.children.length) {
    const c = camGroup.children.pop();
    c.geometry?.dispose?.(); c.material?.dispose?.();
  }
  const cams = geo?.cameras || {};
  const palette = [0xe6194b, 0x3cb44b, 0x4363d8, 0xf58231, 0x911eb4, 0x46f0f0];
  let idx = 0;
  for (const dev of Object.keys(cams)) {
    const col = palette[idx++ % palette.length];
    const o = cams[dev].pos, f = cams[dev].forward;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(o[0], o[1], o[2]),
      new THREE.Vector3(o[0] + f[0] * 0.15, o[1] + f[1] * 0.15, o[2] + f[2] * 0.15),
    ]);
    camGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col })));
    const dotG = new THREE.SphereGeometry(0.012, 8, 8);
    const dot = new THREE.Mesh(dotG, new THREE.MeshBasicMaterial({ color: col }));
    dot.position.set(o[0], o[1], o[2]);
    camGroup.add(dot);
  }
}

function frameCamera(bounds) {
  if (!bounds || bounds.length !== 6) return;
  const c = new THREE.Vector3(
    (bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2, (bounds[2] + bounds[5]) / 2);
  const span = Math.max(
    bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2], 0.3);
  controls.target.copy(c);
  camera.position.set(c.x + span, c.y - span, c.z + span * 0.8);
  camera.near = span / 100;
  camera.far = span * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

function togglePlay() {
  playing = !playing;
  playBtn.textContent = playing ? '❚❚' : '▶';
  lastStep = performance.now();
}

function renderHud() {
  const hud = root.querySelector('[data-role="hud"]');
  const rows = [];
  if (geo) {
    rows.push(['cameras', Object.keys(geo.cameras || {}).join(', ') || '—']);
    rows.push(['traj frames', geo.n_frames || 0]);
  }
  if (cmeta) {
    rows.push(['cloud fps', (cmeta.fps || 0).toFixed(1)]);
    rows.push(['stride', cmeta.stride]);
  }
  hud.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}
