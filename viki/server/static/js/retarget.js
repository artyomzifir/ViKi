// Retarget tab — whole-trajectory IK plus a shared scene3d comparison view.
import { api, log, FRONTEND_CONFIG, sessionGet, sessionSet, sessionPatch } from './core.js';
import * as scene3d from './scene3d.js';
import * as episodes from './episodes.js';

const LAYERS = {
  cloud: 'cloud', trajectory: 'input wrist', targetTrajectory: 'EE target pose',
  achievedTrajectory: 'robot EE pose', robot: 'robot', fused: 'hand', handFit: 'hand fit',
  board: 'board', frusta: 'cameras', bbox: 'bbox', palm: 'palm',
};

let root = null, ctl = null, epList = [], robots = [], viewedEp = null, viewedPlan = null;
let poll = 0, lastDoneSignature = '', showingPreview = false;

function field(role, label, value, step = 'any', min = '') {
  return `<div class="cfg-row"><label>${label}</label><input type="number" data-role="${role}"
    value="${value}" step="${step}" ${min !== '' ? `min="${min}"` : ''}></div>`;
}

function vectorFields(prefix, values, unit = 'm') {
  return `<div class="ret-vector"><span>${prefix.replaceAll('-', ' ')} <i>${unit}</i></span>
    ${['x', 'y', 'z'].map((axis, i) => `<label>${axis}<input type="number" data-role="${prefix}-${axis}"
      value="${values[i]}" step="0.001"></label>`).join('')}</div>`;
}

export function mount(view) {
  const defaults = FRONTEND_CONFIG.retarget || {};
  // v3 resets position-only sessions now that the calibrated orientation term
  // is part of the default objective.
  const S = { ...defaults, ...sessionGet('retarget-v3', {}) };
  root = document.createElement('div');
  root.className = 'retarget-tab';
  root.innerHTML = `
    <aside class="ret-runcol">
      <div class="calib-sec-title">4 · Run</div>
      <div class="cfg-row"><label>Dataset</label><select data-role="dataset"></select></div>
      <div class="ret-eps" data-role="eps"></div>
      <label class="perc-all"><input type="checkbox" data-role="all"> select all</label>
      <button class="primary" data-role="process">Retarget</button>
      <div class="hint">cln.npz → one globally optimised plan.h5 per episode</div>
    </aside>

    <div class="ret-viewer">
      <div class="viewer-canvas" data-role="canvas"></div>
      <div class="perc-overlay perc-overlay-layers" data-role="layers"></div>
      <div class="perc-overlay ret-metrics" data-role="metrics">Select an episode to compare.</div>
      <div class="perc-overlay perc-overlay-transport">
        <button data-role="stop" title="stop">■</button>
        <button data-role="prev" title="previous frame">◄</button>
        <button data-role="play" title="play / pause">▶</button>
        <button data-role="next" title="next frame">►</button>
        <button data-role="back5">«5s</button><button data-role="fwd5">5s»</button>
        <input type="range" data-role="time" min="0" max="0" value="0" step="1">
        <span data-role="frame-lbl">0 / 0</span>
        <button data-role="prevep">‹</button><button data-role="nextep">›</button>
      </div>
    </div>

    <aside class="ret-side">
      <section class="calib-sec">
        <div class="calib-sec-title">1 · Robot & command</div>
        <div class="cfg-row"><label>Robot</label><select data-role="robot"></select></div>
        <div class="hint" data-role="robot-meta">URDF model</div>
        <div class="cfg-row"><label>Pose source</label><select data-role="pose-source">
          <option value="landmarks">landmarks</option><option value="hand_fit">hand fit</option>
        </select></div>
        <div class="cfg-row"><label>Gripper</label><select data-role="gripper" disabled>
          <option value="binary">binary · open/closed</option></select></div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">2 · Calibration-frame placement</div>
        <div class="hint">Robot axes equal calibration axes. Registration is translation only.</div>
        ${vectorFields('base', S.basePosition || [0, 0, 0])}
        ${vectorFields('hand-to-ee', S.handToEeTranslation || [0, 0, 0])}
        ${vectorFields('hand-to-ee-rpy', S.handToEeRpyDeg || [0, 0, 0], 'deg')}
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">3 · Trajectory objective</div>
        ${field('w-position', 'position', S.wPosition ?? 1, 0.01, 0)}
        ${field('w-orientation', 'orientation', S.wOrientation ?? 0.01, 0.01, 0)}
        <div class="hint">SO(3) tracking through the fixed hand→EE rotation above.</div>
        ${field('w-velocity', 'velocity λᵥ', S.wVelocity ?? 0.002, 0.0001, 0)}
        ${field('w-acceleration', 'acceleration λₐ', S.wAcceleration ?? 0.00002, 0.00001, 0)}
        ${field('w-posture', 'posture λᵣ', S.wPosture ?? 0.0001, 0.0001, 0)}
        ${field('huber-delta', 'Huber δ', S.huberDelta ?? 0.05, 0.005, 0.0001)}
        ${field('confidence-floor', 'confidence floor', S.confidenceFloor ?? 0.05, 0.01, 0)}
        ${field('lm-damping', 'LM damping', S.lmDamping ?? 0.00001, 0.00001, 0.0000001)}
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Constraints & solve</div>
        <div class="cfg-row"><label>Self collision</label><input type="checkbox" data-role="collision"
          ${S.collisionEnabled !== false ? 'checked' : ''}></div>
        ${field('collision-pairs', 'closest pairs', S.collisionPairs ?? 8, 1, 0)}
        ${field('collision-distance', 'min distance, m', S.collisionMinDistanceM ?? 0.02, 0.005, 0)}
        ${field('max-iterations', 'GN iterations', S.maxIterations ?? 40, 1, 1)}
        ${field('max-step', 'trust step, rad', S.maxStepRad ?? 0.2, 0.05, 0.01)}
        ${field('approach-sec', 'approach, s', S.approachSec ?? 2, 0.5, 0)}
        <div class="hint">joint limits · velocity limits · OSQP · sparse whole-trajectory QP</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">Queue</div><div class="perc-queue" data-role="queue"></div>
      </section>
    </aside>`;
  view.appendChild(root);

  ctl = scene3d.create(root.querySelector('[data-role="canvas"]'), {
    api, log,
    layers: { cloud: true, trajectory: true, targetTrajectory: true,
      achievedTrajectory: true, robot: true, fused: false, palm: false,
      ...sessionGet('retargetLayers', {}) },
  });
  ctl.onFrame((frameNo, count) => {
    root.querySelector('[data-role="time"]').max = Math.max(0, count - 1);
    root.querySelector('[data-role="time"]').value = frameNo;
    root.querySelector('[data-role="frame-lbl"]').textContent = `${count ? frameNo + 1 : 0} / ${count}`;
    const frameError = root.querySelector('[data-role="frame-error"]');
    if (frameError) frameError.textContent = frameErrorText(viewedPlan, frameNo);
  });
  root.querySelector('[data-role="pose-source"]').value = S.poseSource || 'landmarks';
  renderLayers();
  root.addEventListener('click', onClick);
  root.addEventListener('change', onChange);
  root.addEventListener('input', onInput);
  Promise.all([loadRobots(S.robot), loadDatasets(S.dataset)]).catch(e => log('retarget: ' + e, 'error'));
  refreshQueue();
  poll = setInterval(refreshQueue, 1500);
}

export function unmount() {
  clearInterval(poll); poll = 0;
  ctl?.dispose(); ctl = null;
  viewedPlan = null;
  root?.remove(); root = null;
}

function renderLayers() {
  const state = ctl.layerState;
  root.querySelector('[data-role="layers"]').innerHTML = Object.entries(LAYERS).map(([key, label]) =>
    `<label><input type="checkbox" data-layer="${key}" ${state[key] ? 'checked' : ''}> ${label}</label>`
  ).join('');
}

async function loadRobots(want) {
  ({ robots } = await api('GET', '/api/pipeline/retarget/robots'));
  const select = root.querySelector('[data-role="robot"]');
  select.innerHTML = robots.map(robot => `<option value="${robot.key}">${robot.key}</option>`).join('');
  if (robots.some(robot => robot.key === want)) select.value = want;
  syncRobotMeta();
}

async function loadDatasets(want) {
  const result = await api('GET', '/api/datasets');
  const select = root.querySelector('[data-role="dataset"]');
  select.innerHTML = result.datasets.map(ds =>
    `<option value="${ds.name}">${ds.name} (${ds.episodes})</option>`).join('');
  if (want && result.datasets.some(ds => ds.name === want)) select.value = want;
  await loadEpisodes();
}

async function loadEpisodes() {
  const dataset = root.querySelector('[data-role="dataset"]').value;
  if (!dataset) return;
  ({ episodes: epList } = await api('GET', `/api/datasets/${encodeURIComponent(dataset)}/episodes`));
  const selected = new Set(selectedEpisodes());
  episodes.renderList(root.querySelector('[data-role="eps"]'), epList, {
    select: true, view: true, selected, activeId: viewedEp,
    emptyText: 'no episodes',
  });
  syncAll();
  persist();
}

function selectedEpisodes() {
  return [...(root?.querySelectorAll('[data-ep]:checked') || [])].map(el => el.dataset.ep);
}

function syncAll() {
  const boxes = [...root.querySelectorAll('[data-ep]')];
  const all = root.querySelector('[data-role="all"]');
  all.checked = boxes.length > 0 && boxes.every(box => box.checked);
}

function number(role) { return +root.querySelector(`[data-role="${role}"]`).value; }
function vector(prefix) { return ['x', 'y', 'z'].map(axis => number(`${prefix}-${axis}`)); }

function options() {
  return {
    robot: root.querySelector('[data-role="robot"]').value,
    pose_source: root.querySelector('[data-role="pose-source"]').value,
    gripper: 'binary',
    base_position: vector('base'),
    hand_to_ee_translation: vector('hand-to-ee'),
    hand_to_ee_rpy_deg: vector('hand-to-ee-rpy'),
    w_position: number('w-position'), w_orientation: number('w-orientation'),
    w_velocity: number('w-velocity'), w_acceleration: number('w-acceleration'),
    w_posture: number('w-posture'), huber_delta: number('huber-delta'),
    confidence_floor: number('confidence-floor'), lm_damping: number('lm-damping'),
    max_iterations: number('max-iterations'), max_step_rad: number('max-step'),
    convergence_rad: FRONTEND_CONFIG.retarget?.convergenceRad ?? 0.001,
    qp_solver: FRONTEND_CONFIG.retarget?.qpSolver ?? 'osqp',
    collision_enabled: root.querySelector('[data-role="collision"]').checked,
    collision_pairs: number('collision-pairs'),
    collision_min_distance_m: number('collision-distance'),
    approach_sec: number('approach-sec'),
  };
}

function persist() {
  if (!root) return;
  const o = options();
  sessionSet('retarget-v3', {
    robot: o.robot, poseSource: o.pose_source, basePosition: o.base_position,
    handToEeTranslation: o.hand_to_ee_translation, handToEeRpyDeg: o.hand_to_ee_rpy_deg,
    wPosition: o.w_position, wOrientation: o.w_orientation, wVelocity: o.w_velocity,
    wAcceleration: o.w_acceleration, wPosture: o.w_posture, huberDelta: o.huber_delta,
    confidenceFloor: o.confidence_floor, lmDamping: o.lm_damping,
    maxIterations: o.max_iterations, maxStepRad: o.max_step_rad,
    collisionEnabled: o.collision_enabled, collisionPairs: o.collision_pairs,
    collisionMinDistanceM: o.collision_min_distance_m, approachSec: o.approach_sec,
    dataset: root.querySelector('[data-role="dataset"]').value,
  });
}

function syncRobotMeta() {
  const item = robots.find(r => r.key === root.querySelector('[data-role="robot"]').value);
  root.querySelector('[data-role="robot-meta"]').textContent = item
    ? `${item.description} · EE ${item.ee_frame} · ${item.joints.length} DoF` : 'URDF model';
}

async function previewRobot() {
  if (!showingPreview || !ctl) return;
  const [x, y, z] = vector('base');
  const robot = encodeURIComponent(root.querySelector('[data-role="robot"]').value);
  try {
    const data = await api('GET', `/api/pipeline/retarget/preview?robot=${robot}&x=${x}&y=${y}&z=${z}`);
    ctl.setRetargetData(data);
  } catch (e) { log('robot preview: ' + e, 'error'); }
}

function renderMetrics(data) {
  const box = root.querySelector('[data-role="metrics"]');
  if (!data?.ready || data.preview) {
    box.innerHTML = `<b>${data?.robot_key || 'robot'} · neutral URDF preview</b><span>Run Retarget to compare trajectories.</span>`;
    return;
  }
  const m = data.metrics || {};
  const orientation = Number(m.orientation_weight) > 0
    ? `orientation RMSE ${Number(m.orientation_rmse_deg).toFixed(1)}°`
    : 'orientation not constrained';
  box.innerHTML = `<b>${data.robot_key} · ${data.solver_status}</b>
    <span>position RMSE ${Number(m.position_rmse_mm).toFixed(1)} mm</span>
    <span>${orientation}</span>
    <span data-role="frame-error">${frameErrorText(data, ctl?.frame || 0)}</span>
    <span>velocity max ${Number(m.max_joint_velocity_rad_s).toFixed(2)} rad/s</span>`;
}

function frameErrorText(data, frameNo) {
  const position = Number(data?.position_error_m?.[frameNo]);
  const orientation = Number(data?.orientation_error_rad?.[frameNo]);
  if (!Number.isFinite(position) || !Number.isFinite(orientation)) return 'frame error unavailable';
  return `frame error ${(position * 1000).toFixed(1)} mm · ${(orientation * 180 / Math.PI).toFixed(1)}°`;
}

async function viewEpisode(id) {
  viewedEp = id;
  viewedPlan = null;
  root.querySelectorAll('.episode-row').forEach(row => row.classList.toggle('active', row.dataset.id === id));
  await ctl.loadEpisode(id, epList);
  try {
    const data = await api('GET', `/api/pipeline/episode/${encodeURIComponent(id)}/retarget`);
    if (data.ready) {
      showingPreview = false; viewedPlan = data; ctl.setRetargetData(data); renderMetrics(data);
    } else {
      showingPreview = true; await previewRobot();
      renderMetrics({ ready: true, preview: true, robot_key: options().robot });
    }
  } catch (e) { showingPreview = true; await previewRobot(); log('plan view: ' + e, 'error'); }
}

async function processSelected() {
  const selected = selectedEpisodes();
  if (!selected.length) { log('Pick at least one episode', 'error'); return; }
  try {
    const result = await api('POST', '/api/pipeline/retarget', { episodes: selected, opts: options() });
    log(`Queued retarget for ${selected.length} episode(s) (${result.job_ids.length} jobs)`, 'ok');
    refreshQueue();
  } catch (e) { log('retarget: ' + e, 'error'); }
}

async function refreshQueue() {
  if (!root) return;
  let jobs = [];
  try { ({ jobs } = await api('GET', '/api/pipeline/jobs')); } catch { return; }
  const relevant = jobs.filter(job => job.kind === 'retarget');
  root.querySelector('[data-role="queue"]').innerHTML = relevant.slice(0, 12).map(job => {
    const p = job.progress || {};
    const pct = p.total ? Math.round(100 * (p.frame || 0) / p.total) : 0;
    const label = job.status === 'queued' ? `queued #${job.queue_pos}`
      : job.status === 'running' ? `${p.stage || 'solve'} ${p.frame || 0}/${p.total || '?'}` : job.status;
    return `<div class="perc-job ${job.status}"><span class="perc-job-ep">${job.episode}</span>
      <span class="perc-job-st">${label}</span><span class="perc-bar"><i style="width:${job.status === 'done' ? 100 : pct}%"></i></span>
      ${job.status === 'queued' ? `<button data-cancel="${job.id}">✕</button>` : ''}</div>`;
  }).join('') || '<div class="hint">no retarget jobs</div>';
  const signature = relevant.filter(job => job.status === 'done').map(job => `${job.id}:${job.finished}`).join('|');
  if (viewedEp && signature && signature !== lastDoneSignature
      && relevant.some(job => job.episode === viewedEp && job.status === 'done')) viewEpisode(viewedEp);
  lastDoneSignature = signature;
}

function setPlayIcon(value) {
  const button = root.querySelector('[data-role="play"]');
  button.textContent = (value ?? ctl.playing) ? '❚❚' : '▶';
}

function adjacentEpisode(delta) {
  if (!epList.length) return;
  let index = epList.findIndex(ep => ep.id === viewedEp);
  index = (Math.max(index, 0) + delta + epList.length) % epList.length;
  viewEpisode(epList[index].id);
}

function onClick(event) {
  const button = event.target.closest('button');
  if (!button || !ctl) return;
  if (button.dataset.view) { viewEpisode(button.dataset.view); return; }
  if (button.dataset.cancel) {
    api('DELETE', `/api/pipeline/jobs/${button.dataset.cancel}`).then(refreshQueue)
      .catch(e => log('cancel: ' + e, 'error')); return;
  }
  ({
    process: processSelected,
    stop: () => { ctl.stop(); setPlayIcon(false); },
    play: () => setPlayIcon(ctl.togglePlay()), prev: () => ctl.step(-1), next: () => ctl.step(1),
    back5: () => ctl.skipSeconds(-1), fwd5: () => ctl.skipSeconds(1),
    prevep: () => adjacentEpisode(-1), nextep: () => adjacentEpisode(1),
  })[button.dataset.role]?.();
}

function onChange(event) {
  const el = event.target;
  if (el.dataset.layer) {
    ctl.setLayer(el.dataset.layer, el.checked);
    sessionPatch('retargetLayers', { [el.dataset.layer]: el.checked });
  } else if (el.dataset.role === 'dataset') loadEpisodes().catch(e => log('episodes: ' + e, 'error'));
  else if (el.dataset.role === 'all') root.querySelectorAll('[data-ep]').forEach(box => { box.checked = el.checked; });
  else if (el.dataset.ep) syncAll();
  else if (el.dataset.role === 'robot') { syncRobotMeta(); persist(); previewRobot(); }
  else if (el.dataset.role) {
    persist();
    if (el.dataset.role.startsWith('base-')) previewRobot();
  }
}

function onInput(event) {
  const el = event.target;
  if (el.dataset.role === 'time') ctl.setFrame(+el.value);
  else if (el.dataset.role) persist();
}
