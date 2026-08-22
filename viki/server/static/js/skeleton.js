// 3D skeleton estimation: WebSocket stream, 2D overlays, 3D canvas, recording.
import { api, log, state } from './core.js';

let skeletonWs = null;

let skelPoints = []; // Current 3D points for the selected camera
let selectedSkelCam = null;
let smoothedCenter = null;
let skelFollowEE = false; // Whether the 3D view follows the end effector
let skelViewMode = 'projections'; // 'projections', 'isometric', or 'camera'
let skelDepthDebug = false; // Whether to draw depth-projection debug dots
let skelDepthMarks = null; // Last received debug_depth_marks (per device)

// Distinct colour per camera so multiple hands can be told apart in the 3D
// panel. Two cameras are expected, but a few extras are provided for safety.
const SKEL_DEVICE_COLORS = ['#5b7fff', '#ff8844', '#44aaff', '#ff44aa', '#aaff44'];
const _deviceColorIndex = {};
function colorForDevice(id) {
  if (!(id in _deviceColorIndex)) _deviceColorIndex[id] = Object.keys(_deviceColorIndex).length;
  return SKEL_DEVICE_COLORS[_deviceColorIndex[id] % SKEL_DEVICE_COLORS.length];
}
let cameraExtrinsics = {}; // deviceId -> { rvec, tvec }
let calibBoard = null; // { board_size, square_size } or null
let calibCameras = null; // [{ device_id, rvec, tvec, fx, fy, cx, cy, ... }]
let calibOverlayVisible = true;

// Set by the calibration module after extrinsics calibration.
export function setCameraExtrinsics(extrinsics) {
  cameraExtrinsics = extrinsics;
}

export function setCalibBoard(board) {
  calibBoard = board;
}

export function setCalibCameras(cameras) {
  calibCameras = cameras;
}

export function toggleCalibOverlay() {
  calibOverlayVisible = !calibOverlayVisible;
  drawSkeleton3D([], null);
}

export async function toggleDepthDebug() {
  skelDepthDebug = !skelDepthDebug;
  try {
    await api('POST', '/api/skeleton/depth-debug', { enabled: skelDepthDebug });
  } catch (e) {
    log('Depth debug toggle failed: ' + e, 'error');
  }
  drawSkeleton3D([], null);
}

function _calibVisible() {
  return calibOverlayVisible && calibCameras && calibCameras.length > 0;
}

const SKEL_NAMES = ['Wrist', 'Thumb CMC', 'Thumb MCP', 'Thumb IP', 'Thumb Tip', 'Index MCP', 'Index PIP', 'Index DIP', 'Index Tip', 'Middle MCP', 'Middle PIP', 'Middle DIP', 'Middle Tip', 'Ring MCP', 'Ring PIP', 'Ring DIP', 'Ring Tip', 'Pinky MCP', 'Pinky PIP', 'Pinky DIP', 'Pinky Tip', 'Elbow', 'Shoulder'];
const HAND_CONNS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
];

// Capture the current static background depth for a camera. If deviceId is
// omitted, the currently selected visualize camera is used.
export async function captureBaseDepth(deviceId = null) {
  if (!deviceId) deviceId = document.getElementById('skel-viz-cam')?.value;
  if (!deviceId) {
    log('No camera selected for base depth capture', 'error');
    return;
  }
  log(`Capturing base depth for ${deviceId}...`);
  try {
    await api('POST', `/api/skeleton/capture_base/${deviceId}`);
    log(`Base depth captured for ${deviceId}`, 'ok');
  } catch (e) {
    log(`Base depth capture failed for ${deviceId}: ${e}`, 'error');
  }
}

function populateSkelVizCams() {
  const select = document.getElementById('skel-viz-cam');
  if (!select) return;
  select.innerHTML = '';
  Object.keys(state).forEach(id => {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = id;
    select.appendChild(opt);
  });
  updateSkelVizCam();
}

export function toggleFollowEE() {
  const select = document.getElementById('skel-follow-ee');
  skelFollowEE = select?.value !== 'off';
  smoothedCenter = null;
  drawSkeleton3D([], null);
}

export function updateSkelVizCam() {
  selectedSkelCam = document.getElementById('skel-viz-cam')?.value;
  // Clear current viz if changing camera
  skelPoints = [];
  smoothedCenter = null;
  drawSkeleton3D([]);
  document.getElementById('skel-table-body').innerHTML = '';
}

function rodrigues(rvec) {
  const theta = Math.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);
  if (theta < 1e-6) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  const ux = rvec[0] / theta, uy = rvec[1] / theta, uz = rvec[2] / theta;
  const cosT = Math.cos(theta), sinT = Math.sin(theta), oneMinusCosT = 1 - cosT;

  return [
    [cosT + ux * ux * oneMinusCosT, ux * uy * oneMinusCosT - uz * sinT, ux * uz * oneMinusCosT + uy * sinT],
    [uy * ux * oneMinusCosT + uz * sinT, cosT + uy * uy * oneMinusCosT, uy * uz * oneMinusCosT - ux * sinT],
    [uz * ux * oneMinusCosT - uy * sinT, uz * uy * oneMinusCosT + ux * sinT, cosT + uz * uz * oneMinusCosT]
  ];
}

function drawWristAxes(ctx, ee, projFn, cx, cy, scale) {
  if (!ee || !ee.valid) return;
  // Skip axes when rotation is identity (fallback centroid — no meaningful orientation).
  const R = ee.R_world_palm;
  if (!R || (R[0][1] === 0 && R[0][2] === 0 && R[1][0] === 0 && R[1][2] === 0 && R[2][0] === 0 && R[2][1] === 0)) return;
  const pos = ee.position;
  const len = 0.05;
  const colors = ['#ff4444', '#44ff44', '#4488ff'];
  const labels = ['X', 'Y', 'Z'];
  const origin = projFn(pos);
  if (!origin) return;
  // R_world_palm is column-major: each column is a world-space axis.
  for (let i = 0; i < 3; i++) {
    const tip = [
      pos[0] + len * R[0][i],
      pos[1] + len * R[1][i],
      pos[2] + len * R[2][i],
    ];
    const pTip = projFn(tip);
    if (!pTip) continue;
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx + origin.x * scale, cy - origin.y * scale);
    ctx.lineTo(cx + pTip.x * scale, cy - pTip.y * scale);
    ctx.stroke();
    ctx.fillStyle = colors[i];
    ctx.font = '9px SF Mono';
    ctx.fillText(labels[i], cx + pTip.x * scale + 2, cy - pTip.y * scale);
  }
}

function _drawHand(ctx, proj, cx, cy, scale, landmarks, color) {
  if (!landmarks) return;
  // Arm chain (wrist -> elbow -> shoulder) for context.
  ctx.strokeStyle = 'rgba(150,150,170,0.7)';
  ctx.lineWidth = 1;
  [[0, 21], [21, 22]].forEach(([a, b]) => {
    const pa = landmarks[a], pb = landmarks[b];
    if (!pa || !pb || isNaN(pa[0]) || isNaN(pb[0])) return;
    const qa = proj(pa), qb = proj(pb);
    if (!qa || !qb) return;
    ctx.beginPath();
    ctx.moveTo(cx + qa.x * scale, cy - qa.y * scale);
    ctx.lineTo(cx + qb.x * scale, cy - qb.y * scale);
    ctx.stroke();
  });
  // Hand bones (camera colour).
  ctx.strokeStyle = color || 'rgba(91,127,255,0.9)';
  ctx.lineWidth = 1.5;
  HAND_CONNS.forEach(([a, b]) => {
    const pa = landmarks[a], pb = landmarks[b];
    if (!pa || !pb || isNaN(pa[0]) || isNaN(pb[0])) return;
    const qa = proj(pa), qb = proj(pb);
    if (!qa || !qb) return;
    ctx.beginPath();
    ctx.moveTo(cx + qa.x * scale, cy - qa.y * scale);
    ctx.lineTo(cx + qb.x * scale, cy - qb.y * scale);
    ctx.stroke();
  });
  // Joints (wrist highlighted; others in camera colour).
  for (let i = 0; i <= 20; i++) {
    const p = landmarks[i];
    if (!p || isNaN(p[0]) || isNaN(p[1]) || isNaN(p[2])) continue;
    const q = proj(p);
    if (!q) continue;
    ctx.fillStyle = (i === 0) ? '#ffcc00' : (color || '#5b7fff');
    ctx.beginPath();
    ctx.arc(cx + q.x * scale, cy - q.y * scale, i === 0 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Raw depth-projection debug dots (red), one per landmark that the depth
// camera could deproject. Drawn on top of the skeleton for visual comparison.
function _drawDepthMarks(ctx, proj, cx, cy, scale, marks) {
  if (!marks) return;
  ctx.fillStyle = '#ff2d2d';
  for (let i = 0; i < marks.length; i++) {
    const p = marks[i];
    if (!p || isNaN(p[0]) || isNaN(p[1]) || isNaN(p[2])) continue;
    const q = proj(p);
    if (!q) continue;
    ctx.beginPath();
    ctx.arc(cx + q.x * scale, cy - q.y * scale, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function _drawBoard(ctx, proj, cx, cy, scale) {
  if (!calibOverlayVisible || !calibBoard) return;
  const bs = calibBoard.board_size, ss = calibBoard.square_size;
  if (!bs || bs.length < 2 || !ss) return;
  const bw = bs[0] * ss, bh = bs[1] * ss;
  const corners = [
    [-bw / 2, -bh / 2, 0], [bw / 2, -bh / 2, 0],
    [bw / 2, bh / 2, 0], [-bw / 2, bh / 2, 0],
  ];
  const pts = corners.map(p => proj(p));
  if (pts.some(p => !p)) return;
  ctx.strokeStyle = 'rgba(255,255,100,0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx + pts[0].x * scale, cy - pts[0].y * scale);
  for (let i = 1; i < 4; i++)
    ctx.lineTo(cx + pts[i].x * scale, cy - pts[i].y * scale);
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,100,0.08)';
  ctx.fill();
}

function _camWorldPos(cam) {
  const R = rodrigues(cam.rvec), t = cam.tvec;
  // Camera position in world frame is -R^T @ tvec.
  return [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ];
}

function _drawCameras(ctx, proj, cx, cy, scale) {
  if (!_calibVisible()) return;
  const colors = ['#00ff88', '#ff8844', '#44aaff', '#ff44aa', '#aaff44'];
  calibCameras.forEach((cam, i) => {
    const p = proj(_camWorldPos(cam));
    if (!p) return;
    const col = colors[i % colors.length];
    const px = cx + p.x * scale, py = cy - p.y * scale;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '9px SF Mono';
    ctx.fillText(cam.device_id, px + 5, py + 3);
  });
}

function drawSkeleton3D(hands, depthMarks) {
  const canvas = document.getElementById('skel-canvas-3d');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const container = document.getElementById('skeleton-viz');
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scale = canvas.width / 5;
  let projCenter = { x: 0, y: 0, z: 0 };

  // Use the first camera's hand as the reference for centring/following.
  const primary = (hands && hands.length) ? hands[0] : null;

  if (skelFollowEE) {
    if (primary && primary.endEffector && primary.endEffector.valid) {
      projCenter = {
        x: primary.endEffector.position[0],
        y: primary.endEffector.position[1],
        z: primary.endEffector.position[2],
      };
    } else if (primary && primary.landmarks && primary.landmarks.length > 0) {
      const valid = primary.landmarks.map((p, i) => (p && !isNaN(p[0]) && !isNaN(p[1]) && !isNaN(p[2]) && p[2] > 0.1) ? i : -1).filter(i => i >= 0);
      if (valid.length > 0) {
        projCenter = {
          x: valid.reduce((s, i) => s + primary.landmarks[i][0], 0) / valid.length,
          y: valid.reduce((s, i) => s + primary.landmarks[i][1], 0) / valid.length,
          z: valid.reduce((s, i) => s + primary.landmarks[i][2], 0) / valid.length,
        };
      }
    }
  }

  if (!smoothedCenter) smoothedCenter = { ...projCenter };
  else {
    const a = 0.1;
    smoothedCenter.x += a * (projCenter.x - smoothedCenter.x);
    smoothedCenter.y += a * (projCenter.y - smoothedCenter.y);
    smoothedCenter.z += a * (projCenter.z - smoothedCenter.z);
  }

  const pc = smoothedCenter;

  if (skelViewMode === 'projections') {
    const viewW = canvas.width / 3;
    const viewH = canvas.height;
    const views = [
      { label: '(X-Z)', proj: (p) => ({ x: -(p[0] - pc.x), y: p[2] - pc.z }), offset: 0 },
      { label: '(X-Y)', proj: (p) => ({ x: -(p[0] - pc.x), y: p[1] - pc.y }), offset: viewW },
      { label: '(Z-Y)', proj: (p) => ({ x: p[2] - pc.z, y: p[1] - pc.y }), offset: viewW * 2 },
    ];
    views.forEach(view => {
      const cx = view.offset + viewW / 2, cy = viewH / 2;
      ctx.fillStyle = '#6b6b80';
      ctx.font = '10px SF Mono';
      ctx.fillText(view.label, view.offset + 10, 20);
      ctx.fillStyle = '#fff';
      // Wrist dot (primary camera)
      if (primary && primary.endEffector && primary.endEffector.valid) {
        const wp = view.proj(primary.endEffector.position);
        ctx.beginPath();
        ctx.arc(cx + wp.x * scale, cy - wp.y * scale, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      drawWristAxes(ctx, primary ? primary.endEffector : null, view.proj, cx, cy, scale);
      hands.forEach(h => _drawHand(ctx, view.proj, cx, cy, scale, h.landmarks, h.color));
      if (skelDepthDebug) _drawDepthMarks(ctx, view.proj, cx, cy, scale, depthMarks);
      _drawBoard(ctx, view.proj, cx, cy, scale);
      _drawCameras(ctx, view.proj, cx, cy, scale);
    });
  } else if (skelViewMode === 'isometric') {
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const projectIso = (p) => {
      const dx = -(p[0] - pc.x), dy = p[1] - pc.y, dz = p[2] - pc.z;
      const x1 = dx * Math.cos(Math.PI / 4) + dz * Math.sin(Math.PI / 4);
      const z1 = -dx * Math.sin(Math.PI / 4) + dz * Math.cos(Math.PI / 4);
      return { x: x1, y: dy * Math.cos(Math.PI / 6) - z1 * Math.sin(Math.PI / 6) };
    };
    const axisLen = 0.2;
    const po = projectIso([0, 0, 0]);
    [[axisLen, 0, 0], [0, axisLen, 0], [0, 0, axisLen]].forEach((a, i) => {
      const pe = projectIso(a);
      const colors = ['#ff5b5b', '#5bff5b', '#5b7fff'], labels = ['X', 'Y', 'Z'];
      ctx.strokeStyle = colors[i]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx + po.x * scale, cy - po.y * scale); ctx.lineTo(cx + pe.x * scale, cy - pe.y * scale); ctx.stroke();
      ctx.fillStyle = colors[i]; ctx.font = '10px SF Mono'; ctx.fillText(labels[i], cx + pe.x * scale + 2, cy - pe.y * scale);
    });
    ctx.fillStyle = '#fff';
    if (primary && primary.endEffector && primary.endEffector.valid) {
      const wp = projectIso(primary.endEffector.position);
      ctx.beginPath();
      ctx.arc(cx + wp.x * scale, cy - wp.y * scale, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    drawWristAxes(ctx, primary ? primary.endEffector : null, projectIso, cx, cy, scale);
    hands.forEach(h => _drawHand(ctx, projectIso, cx, cy, scale, h.landmarks, h.color));
    if (skelDepthDebug) _drawDepthMarks(ctx, projectIso, cx, cy, scale, depthMarks);
    _drawBoard(ctx, projectIso, cx, cy, scale);
    _drawCameras(ctx, projectIso, cx, cy, scale);
  } else if (skelViewMode === 'camera') {
    const extrins = cameraExtrinsics[selectedSkelCam];
    if (!extrins) {
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.fillText('No extrinsics', canvas.width / 2, canvas.height / 2); ctx.textAlign = 'left'; return;
    }
    const R = rodrigues(extrins.rvec), t = extrins.tvec;
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const projectCam = (p) => {
      const xc = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0];
      const yc = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1];
      const zc = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2];
      if (zc <= 0.1) return null;
      return { x: xc / zc, y: yc / zc };
    };
    ctx.fillStyle = '#fff';
    if (primary && primary.endEffector && primary.endEffector.valid) {
      const wp = projectCam(primary.endEffector.position);
      if (wp) { ctx.beginPath(); ctx.arc(cx + wp.x * scale, cy - wp.y * scale, 4, 0, Math.PI * 2); ctx.fill(); }
    }
    drawWristAxes(ctx, primary ? primary.endEffector : null, projectCam, cx, cy, scale);
    hands.forEach(h => _drawHand(ctx, projectCam, cx, cy, scale, h.landmarks, h.color));
    if (skelDepthDebug) _drawDepthMarks(ctx, projectCam, cx, cy, scale, depthMarks);
    if (calibOverlayVisible) {
      _drawBoard(ctx, projectCam, cx, cy, scale);
    }
    if (calibOverlayVisible && calibCameras && calibCameras.length > 0) {
      const colors = ['#00ff88', '#ff8844', '#44aaff', '#ff44aa', '#aaff44'];
      calibCameras.forEach((cam, i) => {
        const p = projectCam(_camWorldPos(cam));
        if (!p) return;
        const col = colors[i % colors.length];
        const px = cx + p.x * scale, py = cy - p.y * scale;
        ctx.strokeStyle = col;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.font = '9px SF Mono';
        ctx.fillStyle = col;
        ctx.fillText(cam.device_id, px + 6, py + 3);
      });
    }
  }
}

function updateSkeletonUI(hands, depthMarks) {
  // Show the landmark table for the selected camera if present, else the first.
  const sel = (hands && hands.length)
    ? (hands.find(h => h.device_id === selectedSkelCam) || hands[0])
    : null;

  const tableBody = document.getElementById('skel-table-body');
  if (tableBody) {
    tableBody.innerHTML = '';
    if (sel && sel.landmarks) {
      for (let k = 0; k < SKEL_NAMES.length; k++) {
        const p = sel.landmarks[k];
        if (!p || p[0] === null || isNaN(p[0])) continue;
        const name = SKEL_NAMES[k] || k;
        const row = document.createElement('tr');
        row.innerHTML = `<td>${name}</td><td>${Number(p[0]).toFixed(3)}</td><td>${Number(p[1]).toFixed(3)}</td><td>${Number(p[2]).toFixed(3)}</td>`;
        tableBody.appendChild(row);
      }
    }
  }

  // Depth-projection debug marks. Merge every camera that reported marks so
  // the dots show whenever ANY camera detects the hand. Left red, never fused.
  let depthMarksArray = null;
  if (skelDepthMarks) {
    depthMarksArray = [];
    for (let i = 0; i < SKEL_NAMES.length; i++) depthMarksArray[i] = null;
    for (const camId of Object.keys(skelDepthMarks)) {
      const marks = skelDepthMarks[camId];
      if (!marks) continue;
      for (let i = 0; i < SKEL_NAMES.length; i++) {
        const m = marks[i];
        if (m && !isNaN(m[0]) && !isNaN(m[1]) && !isNaN(m[2])) {
          depthMarksArray[i] = m;
        }
      }
    }
  }
  drawSkeleton3D(hands || [], depthMarksArray);
}

export function toggleSkeleton() {
  const panel = document.getElementById('skeleton-panel');
  const isVisible = panel.style.display === 'block';
  panel.style.display = isVisible ? 'none' : 'block';
  if (!isVisible) {
    populateSkelVizCams();
    updateSkelStatus();
    if (!calibCameras) {
      api('GET', '/api/calibration/viz').then(viz => {
        if (viz.board) setCalibBoard(viz.board);
        if (viz.cameras && viz.cameras.length > 0) setCalibCameras(viz.cameras);
      }).catch(() => {});
    }
  }
}

export function toggleSkelView() {
  const modes = ['projections', 'isometric', 'camera'];
  const currIdx = modes.indexOf(skelViewMode);
  skelViewMode = modes[(currIdx + 1) % modes.length];

  const labels = {
    'projections': 'Projections',
    'isometric': 'Isometric',
    'camera': 'Camera'
  };
  document.getElementById('btn-skel-view').textContent = `View: ${labels[skelViewMode]}`;
}

async function updateSkelStatus() {
  try {
    const data = await api('GET', '/api/skeleton/status');
    const statusEl = document.getElementById('skeleton-status');
    statusEl.textContent = `Status: ${data.enabled ? 'Running' : 'Idle'} ${data.recording ? ' (RECORDING)' : ''}`;

    document.getElementById('btn-skel-start').disabled = data.enabled;
    document.getElementById('btn-skel-stop').disabled = !data.enabled;
    document.getElementById('btn-skel-rec-start').disabled = !data.enabled;
    document.getElementById('btn-skel-rec-stop').disabled = !data.recording;
  } catch (e) {
    log('Skel status check failed', 'error');
  }
}

export async function toggleEstimation(enable) {
  log(`Turning skeleton estimation ${enable ? 'ON' : 'OFF'}...`);
  try {
    await api('POST', '/api/skeleton/toggle', { enabled: enable });
    updateSkelStatus();
    if (enable) {
      startSkelStream();
    } else {
      stopSkelStream();
      // Clear all overlay canvases when estimation is turned off
      Object.keys(state).forEach(id => {
        const canvas = document.getElementById(`skel-canvas-${id}`);
        if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      });
      drawSkeleton3D([]);
      document.getElementById('skel-table-body').innerHTML = '';
    }
  } catch (e) {
    log('Estimation toggle failed: ' + e);
  }
}

export async function toggleRecording(enable) {
  log(`Turning skeleton recording ${enable ? 'ON' : 'OFF'}...`);
  try {
    await api('POST', '/api/skeleton/record', { enabled: enable });
    updateSkelStatus();
  } catch (e) {
    log('Recording toggle failed: ' + e, 'error');
  }
}

function startSkelStream() {
  if (skeletonWs) return;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/api/skeleton/stream`;
  skeletonWs = new WebSocket(wsUrl);

  skeletonWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const detections = data.detections || {};
    const frames = data.frames || [];
    skelDepthMarks = data.debug_depth_marks || null;

    // Build one hand object per camera, each with its own colour.
    const hands = frames.map(f => {
      const landmarksArray = [];
      for (let i = 0; i < SKEL_NAMES.length; i++) {
        landmarksArray[i] = (f.landmarks && (i in f.landmarks)) ? f.landmarks[i] : null;
      }
      return {
        device_id: f.device_id,
        landmarks: landmarksArray,
        endEffector: f.end_effector || null,
        color: colorForDevice(f.device_id),
      };
    });

    // Update detection status in panel
    const detEl = document.getElementById('skeleton-detections');
    if (detEl) {
      const statusHtml = Object.keys(state).map(id => {
        const det = detections[id];
        const detected = det && Object.keys(det).length > 0;
        return `<div style="color: ${detected ? 'var(--green)' : 'var(--red)'}">${id}: ${detected ? 'Detected' : 'Not Detected'}</div>`;
      }).join('');
      detEl.innerHTML = statusHtml;
    }

    for (const [deviceId, det] of Object.entries(detections)) {
      const canvas = document.getElementById(`skel-canvas-${deviceId}`);
      if (!canvas) continue;
      const ctx = canvas.getContext('2d');
      const color = colorForDevice(deviceId);

      if (!det) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        continue;
      }

      const pts2d = (typeof det === 'object' && 'px' in det) ? det.px : det;
      if (!pts2d || typeof pts2d !== 'object') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        continue;
      }

      const img = document.getElementById(`color-${deviceId}`);
      if (!img) continue;

      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;

      const scaleX = canvas.width / img.naturalWidth;
      const scaleY = canvas.height / img.naturalHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;

      const coords = {};
      Object.keys(pts2d).forEach(k => {
        const p = pts2d[k];
        if (p && isFinite(p[0]) && isFinite(p[1])) {
          coords[k] = [p[0] * scaleX, p[1] * scaleY];
        }
      });

      HAND_CONNS.forEach(([a, b]) => {
        const ca = coords[a], cb = coords[b];
        if (ca && cb) {
          ctx.beginPath();
          ctx.moveTo(ca[0], ca[1]);
          ctx.lineTo(cb[0], cb[1]);
          ctx.stroke();
        }
      });

      Object.entries(coords).forEach(([k, p]) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], Number(k) === 0 ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    updateSkeletonUI(hands);
  };

  skeletonWs.onclose = () => {
    skeletonWs = null;
    log('Skeleton stream closed');
  };
}

function stopSkelStream() {
  if (skeletonWs) {
    skeletonWs.close();
    skeletonWs = null;
  }
}
