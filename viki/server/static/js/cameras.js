// Camera discovery, cards, live streams, start/stop, RGB-D recording.
import { api, log, state, CAMERA_CONFIG, FRONTEND_CONFIG } from './core.js';

export async function scanDevices() {
  log('Scanning for devices...');
  try {
    const data = await api('GET', '/api/cameras/devices');
    log(`Found: ${data.realsense.length} RealSense, ${data.kinect.length} Kinect`, 'ok');
    const all = [
      ...data.realsense.map(id => ({ id, type: 'realsense' })),
      ...data.kinect.map(id => ({ id, type: 'kinect' })),
    ];
    renderCards(all, data.active || []);
  } catch (e) {
    log('Scan failed: ' + e, 'error');
  }
}

function renderCards(devices, active) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  if (!devices.length) {
    grid.innerHTML = `<div class="empty-state"><h2>No cameras detected</h2><p>Make sure cameras are connected and container has USB access.</p></div>`;
    return;
  }
  devices.forEach(d => {
    const isActive = active.includes(d.id);
    state[d.id] = { running: isActive, type: d.type, ...(state[d.id] || {}) };
    grid.appendChild(buildCard(d.id, d.type, isActive));
    if (isActive) attachStreams(d.id);
  });
}

function buildControls(id, type) {
  const cfg = CAMERA_CONFIG[type] || CAMERA_CONFIG.realsense;

  const resOpts = cfg.resolutions.map(r =>
    `<option value="${r}" ${r === cfg.defaultRes ? 'selected' : ''}>${r.replace('x', ' × ')}</option>`
  ).join('');

  const fpsOpts = cfg.fps.map(f =>
    `<option value="${f}" ${f === cfg.defaultFps ? 'selected' : ''}>${f} fps</option>`
  ).join('');

  let depthPart = '';
  if (cfg.depthModes) {
    const dOpts = cfg.depthModes.map(m =>
      `<option value="${m}" ${m === cfg.defaultDepth ? 'selected' : ''}>${m}</option>`
    ).join('');
    depthPart = `<div class="sep"></div><label>depth</label><select id="depthmode-${id}" data-change="updateFpsForDepthMode" data-id="${id}">${dOpts}</select>`;
  }

  return `
    <label>res</label><select id="res-${id}">${resOpts}</select>
    <label>fps</label><select id="fps-${id}">${fpsOpts}</select>
    ${depthPart}
  `;
}

function buildCard(id, type, running) {
  const card = document.createElement('div');
  card.className = 'camera-card' + (running ? ' running' : '');
  card.id = 'card-' + id;
  card.innerHTML = `
    <div class="card-header">
      <div class="dot ${running ? 'alive' : ''}" id="dot-${id}"></div>
      <span class="name">${id}</span>
      <span class="tag ${type}">${type}</span>
      ${running ? `<span class="tag running" id="live-tag-${id}">LIVE</span>` : `<span class="tag running" id="live-tag-${id}" style="display:none">LIVE</span>`}
    </div>
    <div class="card-controls">
      <button id="btn-start-${id}" ${running ? 'disabled' : ''} data-action="startCamera" data-id="${id}">▶ Start</button>
      <button id="btn-stop-${id}"  ${running ? '' : 'disabled'} class="danger" data-action="stopCamera" data-id="${id}">■ Stop</button>
      <div class="sep"></div>
      ${buildControls(id, type)}
    </div>
    <div class="streams">
      <div class="stream-panel">
        <img id="color-${id}" src="" alt="color">
        <canvas id="skel-canvas-${id}" class="skel-overlay"></canvas>
        <span class="stream-label">COLOR</span>
      </div>
      <div class="stream-divider"></div>
      <div class="stream-panel">
        <img id="depth-${id}" src="" alt="depth">
        <span class="stream-label">DEPTH</span>
      </div>
    </div>
    <div class="card-footer" id="footer-${id}"><span>—</span></div>
  `;
  return card;
}

export function updateFpsForDepthMode(id) {
  const cfg = CAMERA_CONFIG[state[id]?.type] || CAMERA_CONFIG.realsense;
  if (!cfg.depthModeMaxFps) return;
  const depthMode = document.getElementById(`depthmode-${id}`)?.value;
  const maxFps = cfg.depthModeMaxFps[depthMode] || 30;
  const fpsSelect = document.getElementById(`fps-${id}`);
  if (!fpsSelect) return;
  for (const opt of fpsSelect.options) {
    opt.disabled = parseInt(opt.value) > maxFps;
    if (parseInt(opt.value) > maxFps && fpsSelect.value === opt.value) {
      fpsSelect.value = maxFps;
    }
  }
}

function getCardConfig(id) {
  const resVal = document.getElementById(`res-${id}`)?.value || '640x480';
  const [w, h] = resVal.split('x').map(Number);
  const fps = parseInt(document.getElementById(`fps-${id}`)?.value || '30');
  const depth = document.getElementById(`depthmode-${id}`)?.value || 'NFOV_UNBINNED';
  return { color_width: w, color_height: h, fps, depth_mode: depth };
}

function attachStreams(id) {
  const colorImg = document.getElementById('color-' + id);
  const depthImg = document.getElementById('depth-' + id);
  // Small delay to let the camera worker produce first frame before streaming
  setTimeout(() => {
    if (colorImg) colorImg.src = `/api/cameras/${id}/stream?t=${Date.now()}`;
    if (depthImg) depthImg.src = `/api/cameras/${id}/depth?t=${Date.now()}`;
  }, 1500);
  if (state[id]?.infoInterval) clearInterval(state[id].infoInterval);
  state[id].infoInterval = setInterval(() => updateInfo(id), 2000);
}

function detachStreams(id) {
  const colorImg = document.getElementById('color-' + id);
  const depthImg = document.getElementById('depth-' + id);
  if (colorImg) colorImg.src = '';
  if (depthImg) depthImg.src = '';
  if (state[id]?.infoInterval) clearInterval(state[id].infoInterval);
}

async function updateInfo(id) {
  try {
    const info = await api('GET', `/api/cameras/${id}/info`);
    const footer = document.getElementById('footer-' + id);
    if (!footer) return;

    if (!info) return;

    // Update color panel aspect ratio
    if (info.color_shape) {
      const [ch, cw] = info.color_shape;
      const colorPanel = document.querySelector(`#card-${id} .stream-panel:first-child`);
      if (colorPanel) colorPanel.style.aspectRatio = `${cw}/${ch}`;
    }

    // Update depth panel aspect ratio
    if (info.depth_shape) {
      const [dh, dw] = info.depth_shape;
      const depthPanel = document.querySelector(`#card-${id} .stream-panel:last-child`);
      if (depthPanel) depthPanel.style.aspectRatio = `${dw}/${dh}`;
    }

    const shape = info.color_shape ? `${info.color_shape[1]}×${info.color_shape[0]}` : '—';
    const dshape = info.depth_shape ? `depth: ${info.depth_shape[1]}×${info.depth_shape[0]}` : '';
    const ci = info.color_intrinsics;
    const intr = ci ? `fx=${ci.fx.toFixed(1)} fy=${ci.fy.toFixed(1)}` : '';
    footer.innerHTML = `<span>color: ${shape}</span><span>${dshape}</span><span>${intr}</span>`;
  } catch { /* camera stopped */ }
}

export async function startCamera(id) {
  const cfg = getCardConfig(id);
  log(`Starting ${id} @ ${cfg.color_width}×${cfg.color_height} ${cfg.fps}fps...`);
  try {
    const res = await api('POST', `/api/cameras/${id}/start`, cfg);
    if (res.detail) { log(`${id} error: ${res.detail}`, 'error'); return; }
    log(`${id} started`, 'ok');
    setRunning(id, true);
  } catch (e) {
    log(`Failed to start ${id}: ${e}`, 'error');
  }
}

export async function startRGBDRecording() {
  const btn = document.getElementById('btn-record-rgbd');
  const duration = FRONTEND_CONFIG.recording.duration;
  const fps = FRONTEND_CONFIG.recording.fps;

  log('Starting RGB-D recording...');
  try {
    const res = await api('POST', '/api/record/start', {
      duration: duration,
      fps: fps
    });
    log(`Recording started: ${res.status}`, 'ok');

    // Visual feedback
    btn.disabled = true;
    btn.textContent = 'Recording...';
    btn.classList.remove('primary');
    btn.classList.add('danger');

    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = 'Record';
      btn.classList.remove('danger');
      btn.classList.add('primary');
      log('Recording duration elapsed', 'ok');
    }, duration * 1000);

  } catch (e) {
    log('Recording failed: ' + e, 'error');
  }
}

export async function stopCamera(id) {
  log(`Stopping ${id}...`);
  await api('POST', `/api/cameras/${id}/stop`);
  log(`${id} stopped`);
  setRunning(id, false);
  // Clear the overlay canvas for the stopped camera
  const canvas = document.getElementById(`skel-canvas-${id}`);
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function setRunning(id, running) {
  if (!state[id]) state[id] = {};
  state[id].running = running;
  const card = document.getElementById('card-' + id);
  const dot = document.getElementById('dot-' + id);
  const liveTag = document.getElementById('live-tag-' + id);
  const btnStart = document.getElementById('btn-start-' + id);
  const btnStop = document.getElementById('btn-stop-' + id);
  if (!card) return;
  card.className = 'camera-card' + (running ? ' running' : '');
  if (dot) dot.className = 'dot' + (running ? ' alive' : '');
  if (liveTag) liveTag.style.display = running ? '' : 'none';
  if (btnStart) btnStart.disabled = running;
  if (btnStop) btnStop.disabled = !running;
  if (running) attachStreams(id);
  else detachStreams(id);
}

export async function startAll() {
  for (const id of Object.keys(state))
    if (!state[id].running) await startCamera(id);
}

export async function stopAll() {
  for (const id of Object.keys(state))
    if (state[id].running) await stopCamera(id);
}
