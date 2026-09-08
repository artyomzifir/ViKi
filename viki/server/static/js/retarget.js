// Retarget tab — whole-trajectory IK plus a shared scene3d comparison view.
import { api, log, FRONTEND_CONFIG, sessionGet, sessionSet, sessionPatch } from './core.js';
import { refreshJobs } from './jobs.js';
import * as scene3d from './scene3d.js';
import * as episodes from './episodes.js';

let root = null, ctl = null, epList = [], robots = [], grippers = [], viewedEp = null, viewedPlan = null;
let lastDoneSignature = '', showingPreview = false;

function field(role, label, value, step = 'any', min = '') {
  return `<div class="cfg-row"><label>${label}</label><input type="number" data-role="${role}"
    value="${value}" step="${step}" ${min !== '' ? `min="${min}"` : ''}></div>`;
}

function vectorFields(prefix, values, unit = 'm', label = null, step = 0.001) {
  return `<div class="ret-vector"><span>${label || prefix.replaceAll('-', ' ')} <i>${unit}</i></span>
    ${['x', 'y', 'z'].map((axis, i) => `<label>${axis}<input type="number" data-role="${prefix}-${axis}"
      value="${values[i]}" step="${step}"></label>`).join('')}</div>`;
}

function rpyFields(prefix, values, label = 'RPY', step = 1) {
  const axes = ['x', 'y', 'z'];
  const names = ['roll', 'pitch', 'yaw'];
  return `<div class="ret-vector"><span>${label} <i>deg</i></span>
    ${axes.map((axis, i) => `<label>${names[i]}<input type="number" data-role="${prefix}-${axis}"
      value="${values[i]}" step="${step}"></label>`).join('')}</div>`;
}

export function mount(view) {
  const defaults = FRONTEND_CONFIG.retarget || {};
  // v3 resets position-only sessions now that the calibrated orientation term
  // is part of the default objective.
  const S = { ...defaults, ...sessionGet('retarget-v5', {}) };
  // A fresh tab starts with the neutral assembly instead of an empty scene.
  // Selecting an episode below replaces it with that episode's plan.
  showingPreview = true;
  viewedEp = null;
  viewedPlan = null;
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
        <div class="cfg-row"><label>Position anchor</label><select data-role="target-anchor">
          <option value="pinch_center">thumb–index midpoint</option>
          <option value="wrist">wrist (legacy)</option>
        </select></div>
        <div class="cfg-row"><label>Gripper</label><select data-role="gripper"></select></div>
        <div class="hint" data-role="gripper-meta">physical URDF · continuous opening</div>
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">2 · Calibration-frame placement</div>
        <div class="hint">Robot-base pose in the calibrated scene. RPY is extrinsic XYZ in degrees.</div>
        ${vectorFields('base', S.basePosition || [0, 0, 0], 'm', 'position')}
        ${rpyFields('base-rpy', S.baseRpyDeg || [0, 0, 0], 'base RPY')}
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">3 · User adapter (PoC cylinder)</div>
        <div class="hint">Flange → gripper-base offset. The cylinder spans this vector and is used for visualisation and collision.</div>
        ${vectorFields('adapter', S.adapterTranslationMm || [0, 0, 11], 'mm', 'offset', 0.1)}
        ${vectorFields('adapter-rpy', S.adapterRpyDeg || [0, 0, 0], 'deg', 'gripper RPY', 0.1)}
        ${field('adapter-radius', 'cylinder radius, mm', S.adapterRadiusMm ?? 35, 0.5, 0.1)}
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">4 · Human target → TCP</div>
        <div class="hint">Position is tracked at the midpoint of both jaw panels; orientation comes from the stable palm frame.</div>
        ${vectorFields('hand-to-ee', S.handToEeTranslation || [0, 0, 0], 'm', 'target to TCP')}
        ${vectorFields('hand-to-ee-rpy', S.handToEeRpyDeg || [0, 0, 0], 'deg', 'palm to TCP RPY')}
      </section>

      <section class="calib-sec">
        <div class="calib-sec-title">5 · Trajectory objective</div>
        ${field('w-position', 'position', S.wPosition ?? 1, 0.01, 0)}
        ${field('w-orientation', 'orientation', S.wOrientation ?? 0.01, 0.01, 0)}
        <div class="hint">SO(3) tracking through the fixed hand→gripper TCP rotation above.</div>
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
        <div class="hint">collision-geometry floor z ≥ 0 · joint limits · velocity limits · OSQP · sparse whole-trajectory QP</div>
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
  root.querySelector('[data-role="target-anchor"]').value = S.targetPositionAnchor || 'pinch_center';
  ctl.onLayerChange(l => sessionSet('retargetLayers', l));
  root.addEventListener('click', onClick);
  root.addEventListener('change', onChange);
  root.addEventListener('input', onInput);
  Promise.all([
    loadRobots(S.robot), loadGrippers(S.gripper), loadDatasets(S.dataset),
  ]).then(() => previewRobot()).catch(e => log('retarget: ' + e, 'error'));
  document.addEventListener('jobs:updated', onJobsUpdated);
  refreshJobs();
}

export function unmount() {
  document.removeEventListener('jobs:updated', onJobsUpdated);
  ctl?.dispose(); ctl = null;
  viewedPlan = null;
  root?.remove(); root = null;
}

// The global job widget owns the queue UI now; the Retarget tab only still
// cares that when a retarget job for the episode on screen finishes, the
// comparison view refreshes to the freshly solved plan.
function onJobsUpdated(e) {
  if (!root) return;
  const done = (e.detail || [])
    .filter(job => job.kind === 'retarget' && job.status === 'done');
  const signature = done.map(job => `${job.id}:${job.finished}`).join('|');
  if (viewedEp && signature && signature !== lastDoneSignature
      && done.some(job => job.episode === viewedEp)) {
    viewEpisode(viewedEp);
  }
  lastDoneSignature = signature;
}


async function loadRobots(want) {
  ({ robots } = await api('GET', '/api/pipeline/retarget/robots'));
  const select = root.querySelector('[data-role="robot"]');
  select.innerHTML = robots.map(robot => `<option value="${robot.key}">${robot.key}</option>`).join('');
  if (robots.some(robot => robot.key === want)) select.value = want;
  syncRobotMeta();
}

async function loadGrippers(want) {
  ({ grippers } = await api('GET', '/api/pipeline/retarget/grippers'));
  const select = root.querySelector('[data-role="gripper"]');
  select.innerHTML = grippers.map(item =>
    `<option value="${item.key}" ${item.available ? '' : 'disabled'}>${item.label}${item.available ? '' : ' — pending URDF'}</option>`
  ).join('');
  const selected = grippers.find(item => item.key === want && item.available)
    || grippers.find(item => item.available);
  if (selected) select.value = selected.key;
  syncGripperMeta();
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
    target_position_anchor: root.querySelector('[data-role="target-anchor"]').value,
    gripper: root.querySelector('[data-role="gripper"]').value,
    base_position: vector('base'),
    base_rpy_deg: vector('base-rpy'),
    adapter_translation_mm: vector('adapter'),
    adapter_rpy_deg: vector('adapter-rpy'),
    adapter_radius_mm: number('adapter-radius'),
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
  sessionSet('retarget-v5', {
    robot: o.robot, gripper: o.gripper, poseSource: o.pose_source,
    targetPositionAnchor: o.target_position_anchor, basePosition: o.base_position,
    baseRpyDeg: o.base_rpy_deg,
    adapterTranslationMm: o.adapter_translation_mm, adapterRpyDeg: o.adapter_rpy_deg,
    adapterRadiusMm: o.adapter_radius_mm,
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
    ? `${item.description} · mount ${item.mount_frame} · ${item.joints.length} DoF` : 'URDF model';
}

function syncGripperMeta() {
  const item = grippers.find(g => g.key === root.querySelector('[data-role="gripper"]').value);
  root.querySelector('[data-role="gripper-meta"]').textContent = item
    ? `${item.description || 'URDF pending'} · point ${item.tracking_point || item.tcp_frame || 'pending'} · ${(1000 * item.max_width_m).toFixed(0)} mm · ${(1000 * item.max_speed_m_s).toFixed(0)} mm/s · ${item.license}`
    : 'physical URDF · continuous opening';
}

async function previewRobot() {
  if (!showingPreview || !ctl) return;
  const [x, y, z] = vector('base');
  const [baseRoll, basePitch, baseYaw] = vector('base-rpy');
  const robot = root.querySelector('[data-role="robot"]').value;
  const gripper = root.querySelector('[data-role="gripper"]').value;
  const [adapterX, adapterY, adapterZ] = vector('adapter');
  const [adapterRoll, adapterPitch, adapterYaw] = vector('adapter-rpy');
  const params = new URLSearchParams({
    robot, gripper, x, y, z,
    base_roll_deg: baseRoll, base_pitch_deg: basePitch, base_yaw_deg: baseYaw,
    adapter_x_mm: adapterX, adapter_y_mm: adapterY, adapter_z_mm: adapterZ,
    adapter_roll_deg: adapterRoll, adapter_pitch_deg: adapterPitch,
    adapter_yaw_deg: adapterYaw, adapter_radius_mm: number('adapter-radius'),
    target_position_anchor: root.querySelector('[data-role="target-anchor"]').value,
  });
  try {
    const data = await api('GET', `/api/pipeline/retarget/preview?${params}`);
    ctl.setRetargetData(data);
    renderMetrics(data);
  } catch (e) { log('robot preview: ' + e, 'error'); }
}

function renderMetrics(data) {
  const box = root.querySelector('[data-role="metrics"]');
  if (!data?.ready || data.preview) {
    const adapter = 1000 * Number(data?.adapter?.length_m || 0);
    const anchor = data?.target_position_anchor === 'pinch_center' ? 'thumb–index midpoint' : 'wrist';
    box.innerHTML = `<b>${data?.robot_key || 'robot'} + ${data?.gripper_label || data?.gripper_model || 'gripper'} · neutral URDF preview</b><span>adapter ${adapter.toFixed(1)} mm · target ${anchor}</span><span>Run Retarget to compare trajectories.</span>`;
    return;
  }
  const m = data.metrics || {};
  const orientation = Number(m.orientation_weight) > 0
    ? `orientation RMSE ${Number(m.orientation_rmse_deg).toFixed(1)}°`
    : 'orientation not constrained';
  const adapter = 1000 * Math.hypot(...(data.adapter?.translation_m || [0, 0, 0]));
  const anchor = data.target_position_anchor === 'pinch_center' ? 'thumb–index midpoint' : 'wrist';
  const floor = Math.max(0, Number(m.min_floor_margin_mm));
  const floorText = Number.isFinite(floor) ? `floor clearance ${floor.toFixed(1)} mm` : 'floor constraint unavailable';
  box.innerHTML = `<b>${data.robot_key} + ${data.gripper_model} · ${data.solver_status}</b>
    <span>target ${anchor} → jaw-panel midpoint · adapter ${adapter.toFixed(1)} mm</span>
    <span>position RMSE ${Number(m.position_rmse_mm).toFixed(1)} mm</span>
    <span>${orientation}</span>
    <span data-role="frame-error">${frameErrorText(data, ctl?.frame || 0)}</span>
    <span>${floorText}</span>
    <span>velocity max ${Number(m.max_joint_velocity_rad_s).toFixed(2)} rad/s</span>`;
}

function frameErrorText(data, frameNo) {
  const position = Number(data?.position_error_m?.[frameNo]);
  const orientation = Number(data?.orientation_error_rad?.[frameNo]);
  if (!Number.isFinite(position) || !Number.isFinite(orientation)) return 'frame error unavailable';
  const opening = Number(data?.gripper_opening_m?.[frameNo]);
  const grip = Number.isFinite(opening) ? ` · gripper ${(opening * 1000).toFixed(1)} mm` : '';
  return `frame error ${(position * 1000).toFixed(1)} mm · ${(orientation * 180 / Math.PI).toFixed(1)}°${grip}`;
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
    }
  } catch (e) { showingPreview = true; await previewRobot(); log('plan view: ' + e, 'error'); }
}

async function processSelected() {
  const selected = selectedEpisodes();
  if (!selected.length) { log('Pick at least one episode', 'error'); return; }
  try {
    const result = await api('POST', '/api/pipeline/retarget', { episodes: selected, opts: options() });
    log(`Queued retarget for ${selected.length} episode(s) (${result.job_ids.length} jobs)`, 'ok');
    refreshJobs();
  } catch (e) { log('retarget: ' + e, 'error'); }
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
  if (el.dataset.role === 'dataset') loadEpisodes().catch(e => log('episodes: ' + e, 'error'));
  else if (el.dataset.role === 'all') root.querySelectorAll('[data-ep]').forEach(box => { box.checked = el.checked; });
  else if (el.dataset.ep) syncAll();
  else if (el.dataset.role === 'robot') { syncRobotMeta(); persist(); previewRobot(); }
  else if (el.dataset.role === 'gripper') { syncGripperMeta(); persist(); previewRobot(); }
  else if (el.dataset.role) {
    persist();
    if (el.dataset.role === 'target-anchor' || el.dataset.role.startsWith('base-')
        || el.dataset.role.startsWith('adapter-')) previewRobot();
  }
}

function onInput(event) {
  const el = event.target;
  if (el.dataset.role === 'time') ctl.setFrame(+el.value);
  else if (el.dataset.role) persist();
}
