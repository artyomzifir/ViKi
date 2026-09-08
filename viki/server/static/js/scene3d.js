// scene3d.js — the shared three.js scene for Viewer, Extract, and Retarget.
// One episode at a time: the ChArUco world (board on the Z=0 plane at the
// origin), the per-frame coloured point cloud, per-camera lifted hand skeletons,
// the fused+smoothed skeleton that goes to IK, the wrist trajectory, a palm
// triad + gripper marker, and camera frusta. Orbit / wheel-zoom / right-drag pan.
//
// create(canvasEl, {api, log}) -> a controller the tab drives.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';

// URDF visual meshes for the retarget robot overlay. Files come from
// GET /api/pipeline/retarget/mesh/<path> (served out of models/robot_descriptions);
// the browser HTTP-caches them, so a fresh load per rebuild is cheap and keeps
// geometry ownership per-scene (no shared-buffer disposal hazards).
const _stlLoader = new STLLoader();
const _objLoader = new OBJLoader();
const _daeLoader = new ColladaLoader();
function loadRobotMesh(url) {
  const ext = url.split('?')[0].split('.').pop().toLowerCase();
  return new Promise((resolve, reject) => {
    if (ext === 'stl') _stlLoader.load(url, g => resolve(new THREE.Mesh(g)), undefined, reject);
    else if (ext === 'obj') _objLoader.load(url, resolve, undefined, reject);
    else if (ext === 'dae') _daeLoader.load(url, c => resolve(c.scene), undefined, reject);
    else reject(new Error('unsupported robot mesh: ' + url));
  });
}
function rowMajorMatrix4(m) {
  return new THREE.Matrix4().set(
    m[0][0], m[0][1], m[0][2], m[0][3],
    m[1][0], m[1][1], m[1][2], m[1][3],
    m[2][0], m[2][1], m[2][2], m[2][3],
    m[3] ? m[3][0] : 0, m[3] ? m[3][1] : 0, m[3] ? m[3][2] : 0, m[3] ? m[3][3] : 1);
}

// MediaPipe / RTMPose 21-point hand topology.
const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const CAM_PALETTE = [0xe6194b, 0x3cb44b, 0x4363d8, 0xf58231, 0x911eb4, 0x46f0f0];
const CAM_COLORS = CAM_PALETTE.map(h => new THREE.Color(h));
const cssHex = n => '#' + (n >>> 0).toString(16).padStart(6, '0');

// Every hand in the scene — the fused skeleton, the per-camera skeletons and
// the fitted hand — is drawn the same way: capsules for the 21-point topology
// plus a sphere at each joint. WebGL ignores line widths, so a LineSegments
// skeleton is a 1px hairline; capsules read at any zoom and match hand_fit.
const HAND_BONE_R = 0.0038;
const HAND_JOINT_R = 0.0052;
const MAX_SKEL_CAMS = 8;

const _up = new THREE.Vector3(0, 1, 0);
const _pa = new THREE.Vector3(), _pb = new THREE.Vector3(), _pmid = new THREE.Vector3();
const _pdir = new THREE.Vector3(), _pquat = new THREE.Quaternion();
const _pscl = new THREE.Vector3(), _pmat = new THREE.Matrix4();

// Write one 21-point hand into `bones` (unit cylinders) + `joints` (unit
// spheres) starting at the given instance offsets. `pts` is 21×[x,y,z] whose
// entries may be null / hold null. When `color` is given every written instance
// gets it (so several hands can share one mesh, tinted per camera). Returns the
// instance counts written, for a caller packing hands back to back.
function packCapsuleHand(bones, joints, boneOff, jointOff, pts, boneR, jointR, color) {
  let nb = 0;
  for (const [ia, ib] of HAND_EDGES) {
    const a = pts?.[ia], b = pts?.[ib];
    if (!a || !b || a[0] == null || b[0] == null) continue;
    _pa.fromArray(a); _pb.fromArray(b);
    _pdir.subVectors(_pb, _pa);
    const len = _pdir.length();
    if (!Number.isFinite(len) || len < 1e-7) continue;
    _pmid.addVectors(_pa, _pb).multiplyScalar(0.5);
    _pquat.setFromUnitVectors(_up, _pdir.multiplyScalar(1 / len));
    _pscl.set(boneR, len, boneR);
    _pmat.compose(_pmid, _pquat, _pscl);
    bones.setMatrixAt(boneOff + nb, _pmat);
    if (color) bones.setColorAt(boneOff + nb, color);
    nb++;
  }
  let nj = 0;
  _pquat.identity(); _pscl.setScalar(jointR);
  for (const p of (pts || [])) {
    if (!p || p[0] == null) continue;
    _pa.fromArray(p);
    if (!Number.isFinite(_pa.x + _pa.y + _pa.z)) continue;
    _pmat.compose(_pa, _pquat, _pscl);
    joints.setMatrixAt(jointOff + nj, _pmat);
    if (color) joints.setColorAt(jointOff + nj, color);
    nj++;
  }
  return { nb, nj };
}

// One hand filling a dedicated pair of meshes from instance 0.
function drawCapsuleHand(bones, joints, pts, boneR, jointR) {
  const { nb, nj } = packCapsuleHand(bones, joints, 0, 0, pts, boneR, jointR, null);
  bones.count = nb; joints.count = nj;
  bones.instanceMatrix.needsUpdate = true;
  joints.instanceMatrix.needsUpdate = true;
}

const DEFAULT_LAYERS = {
  axes: true, grid: true,
  cloud: true, perCamera: false, fused: true, trajectory: true,
  palm: true, frusta: true, board: true, bbox: false, handFit: false,
  robot: true, robotMesh: true, targetTrajectory: true, achievedTrajectory: true,
};

// The legend is one grouped list: a block per lifecycle/data-source, its rows
// the individual scene elements. Each row's `key` is the DEFAULT_LAYERS flag it
// toggles; `show()` gates the row on whether that thing can be on screen at all
// (data loaded), and a whole block hides when none of its rows can show.
const LEGEND_GROUPS = [
  {
    title: 'World',
    rows: [
      { key: 'axes', label: 'coordinate axes', swatch: 'triad' },
      { key: 'grid', label: 'ground grid', swatch: '#2c2c34' },
    ],
  },
  {
    title: 'Scene',
    rows: [
      { key: 'board', label: 'ChArUco board', swatch: '#5b7fff', src: 'board' },
      { key: 'bbox', label: 'workspace box', swatch: '#5b6370', src: 'workspace_bbox' },
      { key: 'frusta', label: 'camera frusta', swatch: 'camKey', src: 'cameras' },
      { key: 'trajectory', label: 'wrist path', swatch: '#9aa4b2', src: 'geo' },
    ],
  },
  {
    title: 'Hand',
    rows: [
      { key: 'perCamera', label: 'per-camera (raw lift)', swatch: 'camKey', src: 'geo' },
      { key: 'fused', label: 'fused → IK', swatch: '#ffd166', src: 'geo' },
      { key: 'handFit', label: 'articulated fit', swatch: '#7dd3fc', src: 'geo' },
      { key: 'palm', label: 'palm frame', swatch: 'triad', src: 'geo' },
    ],
  },
  {
    title: 'Retarget',
    rows: [
      { key: 'robot', label: 'robot arm / gripper', src: 'plan',
        swatch: 'linear-gradient(90deg,#8aa4ba 0 55%,#f59e0b 55% 100%)' },
      { key: 'robotMesh', label: 'solid URDF mesh', swatch: '#b9c4d0', src: 'mesh' },
      { key: 'targetTrajectory', label: 'target TCP', swatch: '#f472b6', src: 'plan' },
      { key: 'achievedTrajectory', label: 'achieved TCP', swatch: '#22d3ee', src: 'plan' },
    ],
  },
  {
    title: 'Cloud',
    rows: [
      { key: 'cloud', label: 'point cloud', swatch: 'cloud', src: 'cloud' },
    ],
  },
];

export function create(canvasEl, {
  api, log, layers: initLayers, colorMode: initColor, stride: initStride,
}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d10);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.005, 200);
  camera.up.set(0, 0, 1);
  camera.position.set(0.6, -0.7, 0.6);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  canvasEl.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // ── static world ──────────────────────────────────────────────────────
  // Origin triad as thin cylinders (WebGL ignores LineMaterial.linewidth), so
  // the axes actually read as ~2x an AxesHelper hairline.
  function fatAxes(len = 0.15, radius = 0.003, opacity = 1) {
    const g = new THREE.Group();
    const arm = (color, ax) => {
      const geo = new THREE.CylinderGeometry(radius, radius, len, 12);
      geo.translate(0, len / 2, 0);            // base at origin, tip at +len
      if (ax === 'x') geo.rotateZ(-Math.PI / 2);
      if (ax === 'z') geo.rotateX(Math.PI / 2);
      return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1,
      }));
    };
    g.add(arm(0xff0000, 'x'), arm(0x00ff00, 'y'), arm(0x0000ff, 'z'));
    return g;
  }
  const worldAxes = fatAxes(0.15, 0.003);
  scene.add(worldAxes);
  const grid = new THREE.GridHelper(2, 40, 0x2c2c34, 0x18181d);
  grid.rotation.x = Math.PI / 2;                    // grid on world XY
  scene.add(grid);

  const boardGroup = new THREE.Group();
  scene.add(boardGroup);
  const bboxGroup = new THREE.Group();
  scene.add(bboxGroup);

  // Sensor-derived data lives in the RIG (reference-camera) frame; the world
  // anchor maps it into the installed calibration frame for presentation.
  const worldGroup = new THREE.Group();
  worldGroup.matrixAutoUpdate = false;
  scene.add(worldGroup);

  // Retarget output is already written in that installed calibration frame.
  // Keeping it outside worldGroup is essential: applying T_world_display again
  // would rotate and translate the robot a second time.
  const calibrationGroup = new THREE.Group();
  scene.add(calibrationGroup);

  const frustaGroup = new THREE.Group();
  worldGroup.add(frustaGroup);

  // ── dynamic ───────────────────────────────────────────────────────────
  const cloud = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.006, vertexColors: true, sizeAttenuation: true })
  );
  cloud.frustumCulled = false;
  worldGroup.add(cloud);

  const trajLine = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x9aa4b2 })
  );
  worldGroup.add(trajLine);

  // Retarget overlay. Link origins and parent edges are derived from the URDF
  // by Pinocchio on the backend, so this scene remains dependency-free and the
  // exact same kinematics that produced the plan is what gets displayed.
  const robotGroup = new THREE.Group();
  calibrationGroup.add(robotGroup);
  // URDF visual meshes, sibling of the schematic links so one can replace the
  // other. Populated async in rebuildRobot(); placed per frame in updateRobotFrame().
  const robotMeshGroup = new THREE.Group();
  robotGroup.add(robotMeshGroup);
  let robotMeshReady = false, robotMeshToken = 0;
  const targetTrajLine = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xf472b6 })
  );
  const achievedTrajLine = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x22d3ee })
  );
  calibrationGroup.add(targetTrajLine, achievedTrajLine);
  const targetDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.014, 12, 10),
    new THREE.MeshBasicMaterial({ color: 0xf472b6, wireframe: true })
  );
  const achievedDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.010, 12, 10),
    new THREE.MeshBasicMaterial({ color: 0x22d3ee })
  );
  calibrationGroup.add(targetDot, achievedDot);
  targetDot.visible = achievedDot.visible = false;

  // A position line cannot show whether SO(3) is being tracked.  Draw both
  // end-effector frames at the current sample: target is longer/translucent,
  // achieved is shorter/solid.  Their matching RGB axes coincide only when
  // the robot has matched the target orientation.
  const targetPoseFrame = fatAxes(0.10, 0.0022, 0.48);
  const achievedPoseFrame = fatAxes(0.07, 0.0032);
  targetPoseFrame.visible = achievedPoseFrame.visible = false;
  calibrationGroup.add(targetPoseFrame, achievedPoseFrame);

  // Fused skeleton → IK: solid amber capsule hand.
  const fusedBones = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(1, 1, 1, 10),
    new THREE.MeshBasicMaterial({ color: 0xffd166 }),
    HAND_EDGES.length,
  );
  const fusedJoints = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 12, 9),
    new THREE.MeshBasicMaterial({ color: 0xffdf9e }),
    21,
  );
  fusedBones.count = fusedJoints.count = 0;
  fusedBones.frustumCulled = fusedJoints.frustumCulled = false;
  worldGroup.add(fusedBones, fusedJoints);

  // Per-camera lifted skeletons: one capsule hand per camera, tinted from
  // CAM_PALETTE (same key as the frusta). All cameras share one instanced mesh.
  const perCamBones = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(1, 1, 1, 8),
    new THREE.MeshBasicMaterial(),
    MAX_SKEL_CAMS * HAND_EDGES.length,
  );
  const perCamJoints = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 10, 8),
    new THREE.MeshBasicMaterial(),
    MAX_SKEL_CAMS * 21,
  );
  perCamBones.count = perCamJoints.count = 0;
  perCamBones.frustumCulled = perCamJoints.frustumCulled = false;
  worldGroup.add(perCamBones, perCamJoints);

  // Fitted hand: translucent cylinders are used because WebGL ignores line
  // widths. Reconstructing the MediaPipe joints from capsule endpoints lets us
  // draw the complete 21-joint topology, including the palm cross-links.
  const handBones = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(1, 1, 1, 10),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8 }),
    HAND_EDGES.length
  );
  const handJoints = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 12, 9),
    new THREE.MeshBasicMaterial({ color: 0x7dd3fc }),
    21
  );
  handBones.count = handJoints.count = 0;
  handBones.frustumCulled = handJoints.frustumCulled = false;
  worldGroup.add(handBones, handJoints);

  // Same fat-cylinder axes as the achieved-TCP pose frame, so the two read the
  // same in the scene.
  const palmTriad = fatAxes(0.07, 0.0032);
  palmTriad.visible = false;
  worldGroup.add(palmTriad);

  // ── state ─────────────────────────────────────────────────────────────
  let geo = null, cmeta = null, retarget = null, epId = null, variantId = 'active', episodes = [], epIndex = -1;
  let frame = 0, playing = false, playTimer = 0, playSerial = 0, playPending = false;
  let colorMode = initColor || 'rgb', stride = initStride || 1;
  let layers = { ...DEFAULT_LAYERS, ...(initLayers || {}) };
  let frameCb = null, layerCb = null;
  const cloudCache = new Map();     // frame -> Promise<{xyz, rgb}>
  const fgCache = new Map();        // frame -> Promise<geometry?frame= payload>
  const CACHE_CAP = 80;
  let loadSerial = 0, loadingEpisode = false;
  let cloudPos = new Float32Array(0), cloudCol = new Uint8Array(0);
  let raf = 0, disposed = false;

  // ── legend overlay ────────────────────────────────────────────────────
  // One grouped list pinned into the canvas: a block per data-source/lifecycle
  // (see LEGEND_GROUPS), each row a scene element toggle. A row dims when its
  // layer is off and hides when its data isn't loaded; a block hides when none
  // of its rows can show.
  const legendEl = document.createElement('div');
  legendEl.className = 'scene-legend';
  canvasEl.appendChild(legendEl);
  const camKeyGradient = () =>
    `linear-gradient(90deg,${CAM_PALETTE.slice(0, 3).map(cssHex).join(',')})`;
  // Is the data a row draws from currently available?
  function rowPresent(src) {
    switch (src) {
      case undefined: return true;                 // World: axes + grid
      case 'geo': return !!geo;
      case 'board': return !!geo?.board;
      case 'workspace_bbox': return !!geo?.workspace_bbox;
      case 'cameras': return !!geo?.cameras;
      case 'cloud': return !!cmeta;
      case 'plan': return !!retarget?.ready;
      case 'mesh': return !!retarget?.ready && !!retarget?.robot_visuals?.length;
      default: return true;
    }
  }
  function rowSwatch(row) {
    if (row.swatch === 'triad') return null;       // painted by the .triad class
    if (row.swatch === 'camKey') return camKeyGradient();
    if (row.swatch === 'cloud') {
      return colorMode === 'height'
        ? 'linear-gradient(90deg,#2b6cff,#57c06a,#ff5a5a)' : '#cfd6df';
    }
    return row.swatch;
  }
  for (const group of LEGEND_GROUPS) {
    const head = document.createElement('div');
    head.className = 'scene-legend-group';
    head.textContent = group.title;
    legendEl.appendChild(head);
    group._head = head;
    for (const row of group.rows) {
      const el = document.createElement('button');
      el.className = 'scene-legend-row';
      el.type = 'button';
      el.title = 'toggle ' + row.label;
      el.addEventListener('click', () => setLayer(row.key, !layers[row.key]));
      const tick = document.createElement('span');
      tick.className = 'scene-legend-tick';
      const sw = document.createElement('span');
      sw.className = 'scene-legend-sw' + (row.swatch === 'triad' ? ' triad' : '');
      const label = document.createElement('span');
      label.className = 'scene-legend-label';
      label.textContent = row.label;
      el.append(tick, sw, label);
      legendEl.appendChild(el);
      row._el = el; row._sw = sw;
    }
  }
  function updateLegend() {
    let anyVisible = false;
    for (const group of LEGEND_GROUPS) {
      let groupVisible = false;
      for (const row of group.rows) {
        const present = rowPresent(row.src);
        row._el.hidden = !present;
        if (!present) continue;
        groupVisible = true;
        const on = !!layers[row.key];
        row._el.classList.toggle('off', !on);
        row._el.setAttribute('aria-pressed', String(on));
        if (row.swatch !== 'triad') row._sw.style.background = rowSwatch(row);
      }
      group._head.hidden = !groupVisible;
      anyVisible = anyVisible || groupVisible;
    }
    legendEl.hidden = !anyVisible;
  }
  updateLegend();

  // ── render loop ───────────────────────────────────────────────────────
  function resize() {
    const w = canvasEl.clientWidth || 1, h = canvasEl.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvasEl);
  resize();

  function tick() {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  // ── helpers ───────────────────────────────────────────────────────────
  function fps() { return cmeta?.fps || geo?.fps || 15; }
  function nFrames() { return retarget?.n_frames || cmeta?.n_frames || geo?.n_frames || 0; }

  function clearGroup(g) {
    while (g.children.length) {
      const c = g.children.pop();
      c.geometry?.dispose?.();
      if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
      else c.material?.dispose?.();
    }
  }

  function applyLayerVisibility() {
    worldAxes.visible = layers.axes;
    grid.visible = layers.grid;
    cloud.visible = layers.cloud;
    trajLine.visible = layers.trajectory;
    fusedBones.visible = fusedJoints.visible = layers.fused;
    perCamBones.visible = perCamJoints.visible = layers.perCamera;
    boardGroup.visible = layers.board;
    bboxGroup.visible = layers.bbox;
    frustaGroup.visible = layers.frusta;
    palmTriad.visible = layers.palm && palmTriad.userData.have;
    handBones.visible = layers.handFit;
    handJoints.visible = layers.handFit;
    const robotOn = layers.robot && !!retarget?.ready;
    robotGroup.visible = robotOn;
    // Solid URDF meshes replace the stick-figure links/joints when they are
    // loaded and the "robot: solid mesh" row is on; the base triad + light stay.
    const meshMode = robotOn && layers.robotMesh && robotMeshReady
      && !!retarget?.robot_visuals?.length;
    robotMeshGroup.visible = meshMode;
    for (const c of robotGroup.children) {
      if (c.userData.kind === 'robot-link' || c.userData.kind === 'robot-joint') {
        // Degenerate/fixed-frame edges are deliberately hidden by
        // updateRobotFrame(). Do not resurrect their untouched unit geometry
        // when switching from the solid mesh back to the stick figure.
        c.visible = !meshMode && !!c.userData.frameVisible;
      }
    }
    targetTrajLine.visible = layers.targetTrajectory && !!retarget?.ready;
    targetDot.visible = layers.targetTrajectory && !!retarget?.target_trajectory?.length;
    targetPoseFrame.visible = layers.targetTrajectory && !!retarget?.ready
      && !!targetPoseFrame.userData.have;
    achievedTrajLine.visible = layers.achievedTrajectory && !!retarget?.ready;
    achievedDot.visible = layers.achievedTrajectory && !!retarget?.achieved_trajectory?.length;
    achievedPoseFrame.visible = layers.achievedTrajectory && !!retarget?.ready
      && !!achievedPoseFrame.userData.have;
    updateLegend();
  }

  function applyWorldDisplay(m) {
    // m is a row-major 4x4 (rig -> display). three.js Matrix4.set() takes
    // row-major args, so this is a direct load.
    if (Array.isArray(m) && m.length === 4) {
      worldGroup.matrix.set(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3]);
    } else {
      worldGroup.matrix.identity();
    }
    worldGroup.matrixWorldNeedsUpdate = true;
  }

  function buildBoard() {
    clearGroup(boardGroup);
    const b = geo?.board;
    if (!b || !b.board_size || !b.square_size) return;
    const [cols, rows] = b.board_size;
    const sq = b.square_size;
    const w = cols * sq, h = rows * sq;
    // canonical_board_extrinsics re-centres the world origin on the board
    // centre, so the plane sits AT the origin (±½ square of slack from the
    // (n-1)/2 rounding — negligible next to a 0.4–0.5 m board).
    const geoPlane = new THREE.PlaneGeometry(w, h);
    const plane = new THREE.Mesh(
      geoPlane,
      new THREE.MeshBasicMaterial({ color: 0x20242c, transparent: true, opacity: 0.55,
        side: THREE.DoubleSide })
    );
    boardGroup.add(plane);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geoPlane),
      new THREE.LineBasicMaterial({ color: 0x5b7fff })
    );
    boardGroup.add(edges);
  }

  function buildBbox() {
    clearGroup(bboxGroup);
    const bb = geo?.workspace_bbox;
    if (!bb || bb.length !== 6) return;
    const [x0, x1, y0, y1, z0, z1] = bb;
    const box = new THREE.Box3(
      new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));
    bboxGroup.add(new THREE.Box3Helper(box, 0x394150));
  }

  function buildFrusta() {
    clearGroup(frustaGroup);
    const cams = geo?.cameras || {};
    Object.keys(cams).forEach((dev, i) => {
      const col = CAM_PALETTE[i % CAM_PALETTE.length];
      const o = cams[dev].pos, f = cams[dev].forward;
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...o),
          new THREE.Vector3(o[0] + f[0] * 0.15, o[1] + f[1] * 0.15, o[2] + f[2] * 0.15),
        ]),
        new THREE.LineBasicMaterial({ color: col }));
      frustaGroup.add(line);
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 10, 10),
        new THREE.MeshBasicMaterial({ color: col }));
      dot.position.set(...o);
      frustaGroup.add(dot);
    });
  }

  function buildTrajectory() {
    const T = geo?.wrist_traj || [];
    const segments = [];
    const finite = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
    for (let i = 1; i < T.length; i++) {
      if (!finite(T[i - 1]) || !finite(T[i])) continue;
      segments.push(...T[i - 1], ...T[i]);
    }
    const arr = new Float32Array(segments);
    trajLine.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    trajLine.geometry.setDrawRange(0, arr.length / 3);
    if (arr.length) trajLine.geometry.computeBoundingSphere();
  }

  function setLineTrajectory(line, points) {
    const segments = [];
    const finite = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
    for (let i = 1; i < (points?.length || 0); i++) {
      if (finite(points[i - 1]) && finite(points[i])) segments.push(...points[i - 1], ...points[i]);
    }
    const arr = new Float32Array(segments);
    line.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    line.geometry.setDrawRange(0, arr.length / 3);
    if (arr.length) line.geometry.computeBoundingSphere();
  }

  function placeCylinder(mesh, a, b, radius = 0.012) {
    const start = new THREE.Vector3().fromArray(a), end = new THREE.Vector3().fromArray(b);
    const delta = new THREE.Vector3().subVectors(end, start);
    const length = delta.length();
    mesh.visible = Number.isFinite(length) && length > 1e-7;
    if (!mesh.visible) return;
    mesh.position.addVectors(start, end).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.multiplyScalar(1 / length));
    mesh.scale.set(radius, length, radius);
  }

  function placePoseFrame(group, position, rotation) {
    const havePosition = Array.isArray(position) && position.length === 3
      && position.every(Number.isFinite);
    const haveRotation = Array.isArray(rotation) && rotation.length === 3
      && rotation.every(row => Array.isArray(row) && row.length === 3
        && row.every(Number.isFinite));
    group.userData.have = havePosition && haveRotation;
    if (!group.userData.have) return;
    group.position.set(...position);
    group.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(
      rotation[0][0], rotation[0][1], rotation[0][2], 0,
      rotation[1][0], rotation[1][1], rotation[1][2], 0,
      rotation[2][0], rotation[2][1], rotation[2][2], 0,
      0, 0, 0, 1));
  }

  function disposeSubtree(obj) {
    obj.traverse(o => {
      o.geometry?.dispose?.();
      const m = o.material;
      if (Array.isArray(m)) m.forEach(x => x?.dispose?.());
      else m?.dispose?.();
    });
  }

  function clearRobotMeshes() {
    robotMeshToken++;
    robotMeshReady = false;
    while (robotMeshGroup.children.length) {
      const c = robotMeshGroup.children.pop();
      disposeSubtree(c);
    }
  }

  // Load the URDF visual meshes and stamp each with its parent joint + the
  // fixed joint←geometry offset, so updateRobotFrame() only has to compose
  // world←joint (per frame) with that offset.
  function loadRobotVisuals() {
    clearRobotMeshes();
    const visuals = retarget?.robot_visuals || [];
    if (!visuals.length) { applyLayerVisibility(); return; }
    const token = robotMeshToken;
    let pending = visuals.length;
    const done = () => {
      if (token !== robotMeshToken) return;
      if (--pending <= 0) { robotMeshReady = true; updateRobotFrame(frame); applyLayerVisibility(); }
    };
    for (const v of visuals) {
      const local = rowMajorMatrix4(v.placement || [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
      const s = (v.scale || [1, 1, 1]).map(x => x || 1);
      local.multiply(new THREE.Matrix4().makeScale(s[0], s[1], s[2]));
      // DAE carries its own materials — keep them. STL/OBJ/primitives don't, so
      // paint those with the URDF's meshColor (falling back to a neutral grey).
      const daeMesh = /\.dae($|\?)/i.test(v.mesh_path || '');
      const attach = node => {
        if (token !== robotMeshToken) { disposeSubtree(node); return; }
        if (daeMesh) {
          // ColladaLoader rotates Z-up assets into three's Y-up and scales by
          // the asset's <unit>. Our scene is Z-up (URDF-native, matching
          // Pinocchio), so drop the rotation but fold the unit into `local`.
          const unit = node.scale.x || 1;
          node.rotation.set(0, 0, 0);
          node.scale.setScalar(1);
          if (unit !== 1) local.multiply(new THREE.Matrix4().makeScale(unit, unit, unit));
        }
        node.matrixAutoUpdate = false;
        node.userData.kind = 'robot-visual';
        node.userData.parentJoint = v.parent_joint | 0;
        node.userData.local = local;
        if (!daeMesh) {
          const c = Array.isArray(v.color) && v.color[3] !== 0 ? v.color : [0.73, 0.77, 0.82, 1];
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(c[0], c[1], c[2]), roughness: 0.6, metalness: 0.05,
            transparent: (c[3] ?? 1) < 1, opacity: c[3] ?? 1 });
          node.traverse(o => { if (o.isMesh) o.material = mat; });
        }
        robotMeshGroup.add(node);
      };
      if (v.mesh_path) {
        const url = '/api/pipeline/retarget/mesh/'
          + v.mesh_path.split('/').map(encodeURIComponent).join('/');
        loadRobotMesh(url).then(attach)
          .catch(e => log && log('robot mesh ' + v.name + ': ' + e, 'warn'))
          .finally(done);
      } else if (v.primitive) {
        const p = v.primitive;
        let geo = null;
        if (p.type === 'box') geo = new THREE.BoxGeometry(...(p.size || [0.05, 0.05, 0.05]));
        else if (p.type === 'sphere') geo = new THREE.SphereGeometry(p.radius || 0.02, 16, 12);
        else if (p.type === 'cylinder') {
          geo = new THREE.CylinderGeometry(
            p.radius || 0.02, p.radius || 0.02, p.length || 0.05, 16);
          // hpp-fcl/URDF cylinders are Z-aligned; Three.js builds them on Y.
          geo.rotateX(Math.PI / 2);
        }
        if (geo) attach(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x9aa4b2, roughness: 0.7 })));
        done();
      } else {
        done();
      }
    }
  }

  function rebuildRobot() {
    clearRobotMeshes();
    clearGroup(robotGroup);
    robotGroup.add(robotMeshGroup);
    if (!retarget?.ready) return;
    const linkMaterial = new THREE.MeshStandardMaterial({ color: 0x8aa4ba, roughness: 0.65 });
    const gripperMaterial = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.55 });
    const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xdbeafe, roughness: 0.45 });
    const gripperJointMaterial = new THREE.MeshStandardMaterial({ color: 0xfde68a, roughness: 0.45 });
    for (const [index, _edge] of (retarget.link_edges || []).entries()) {
      const part = retarget.link_groups?.[index] || 'robot';
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(1, 1, 1, 12),
        part === 'gripper' ? gripperMaterial : linkMaterial,
      );
      mesh.userData.kind = 'robot-link';
      mesh.userData.part = part;
      mesh.userData.frameVisible = false;
      mesh.visible = false;
      robotGroup.add(mesh);
    }
    const count = retarget.link_positions?.[0]?.length || 0;
    for (let i = 0; i < count; i++) {
      const part = retarget.point_groups?.[i] || 'robot';
      const radius = part === 'gripper' ? 0.009 : 0.018;
      const joint = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 12, 10),
        part === 'gripper' ? gripperJointMaterial : jointMaterial,
      );
      joint.userData.kind = 'robot-joint';
      joint.userData.part = part;
      joint.userData.frameVisible = false;
      joint.visible = false;
      robotGroup.add(joint);
    }
    const baseAxes = fatAxes(0.10, 0.004);
    baseAxes.userData.kind = 'robot-base';
    if (retarget.base_transform) {
      baseAxes.matrixAutoUpdate = false;
      baseAxes.matrix.copy(rowMajorMatrix4(retarget.base_transform));
      baseAxes.matrixWorldNeedsUpdate = true;
    } else {
      baseAxes.position.set(...retarget.base_position);
    }
    robotGroup.add(baseAxes);
    robotGroup.add(new THREE.HemisphereLight(0xffffff, 0x111827, 1.7));
    loadRobotVisuals();
  }

  function updateRobotFrame(i) {
    if (!retarget?.ready) return;
    const index = Math.max(0, Math.min(i, retarget.n_frames - 1));
    const points = retarget.link_positions?.[index] || [];
    const links = robotGroup.children.filter(child => child.userData.kind === 'robot-link');
    const joints = robotGroup.children.filter(child => child.userData.kind === 'robot-joint');
    const finitePoint = point => Array.isArray(point) && point.length === 3
      && point.every(Number.isFinite);
    (retarget.link_edges || []).forEach(([a, b], k) => {
      const link = links[k];
      if (!link) return;
      if (finitePoint(points[a]) && finitePoint(points[b])) {
        placeCylinder(link, points[a], points[b], link.userData.part === 'gripper' ? 0.006 : 0.012);
      } else link.visible = false;
      link.userData.frameVisible = link.visible;
    });
    joints.forEach((joint, k) => {
      joint.userData.frameVisible = finitePoint(points[k]);
      joint.visible = joint.userData.frameVisible;
      if (joint.userData.frameVisible) joint.position.set(...points[k]);
    });
    if (robotMeshReady && retarget.robot_joint_placements?.length) {
      const jp = retarget.robot_joint_placements[
        Math.min(index, retarget.robot_joint_placements.length - 1)];
      for (const node of robotMeshGroup.children) {
        const m = jp?.[node.userData.parentJoint];
        node.visible = !!m;
        if (!m) continue;
        node.matrix.copy(rowMajorMatrix4(m)).multiply(node.userData.local);
        node.matrixWorldNeedsUpdate = true;
      }
    }
    const target = retarget.target_trajectory?.[index];
    const achieved = retarget.achieved_trajectory?.[index];
    if (target) targetDot.position.set(...target);
    if (achieved) achievedDot.position.set(...achieved);
    placePoseFrame(targetPoseFrame, target, retarget.target_rotation?.[index]);
    placePoseFrame(achievedPoseFrame, achieved, retarget.achieved_rotation?.[index]);
  }

  function setRetargetData(data) {
    retarget = data?.ready ? data : null;
    rebuildRobot();
    setLineTrajectory(targetTrajLine, retarget?.target_trajectory || []);
    setLineTrajectory(achievedTrajLine, retarget?.achieved_trajectory || []);
    updateRobotFrame(frame);
    applyLayerVisibility();
  }

  function frameCamera() {
    // frame on the board + workspace bbox, never on cloud bounds
    const box = new THREE.Box3();
    const b = geo?.board;
    if (b?.board_size && b?.square_size) {
      const hw = b.board_size[0] * b.square_size / 2;
      const hh = b.board_size[1] * b.square_size / 2;
      box.expandByPoint(new THREE.Vector3(-hw, -hh, 0));
      box.expandByPoint(new THREE.Vector3(hw, hh, 0));
    }
    const bb = geo?.workspace_bbox;
    if (bb?.length === 6) {
      box.expandByPoint(new THREE.Vector3(bb[0], bb[2], bb[4]));
      box.expandByPoint(new THREE.Vector3(bb[1], bb[3], bb[5]));
    }
    if (box.isEmpty()) box.expandByPoint(new THREE.Vector3(0.5, 0.5, 0.5));
    const c = box.getCenter(new THREE.Vector3());
    const span = Math.max(box.getSize(new THREE.Vector3()).length(), 0.4);
    controls.target.copy(c);
    camera.position.set(c.x + span * 0.7, c.y - span * 0.9, c.z + span * 0.7);
    camera.near = span / 200;
    camera.far = span * 60;
    camera.updateProjectionMatrix();
    controls.update();
  }

  // ── cloud frames ──────────────────────────────────────────────────────
  function fetchCloud(i) {
    if (cloudCache.has(i)) return cloudCache.get(i);
    const p = fetch(`/api/pipeline/episode/${epId}/cloud/${i}`)
      .then(r => { if (!r.ok) throw new Error('cloud ' + i + ': ' + r.status); return r.arrayBuffer(); })
      .then(buf => {
        const n = new DataView(buf).getInt32(0, true);
        const xyz = new Float32Array(buf, 4, n * 3);
        const rgbU8 = new Uint8Array(buf, 4 + n * 12, n * 3);
        return { n, xyz, rgbU8 };
      }).catch(e => { cloudCache.delete(i); throw e; });
    cloudCache.set(i, p);
    if (cloudCache.size > CACHE_CAP) {
      cloudCache.delete(cloudCache.keys().next().value);
    }
    return p;
  }

  function paintCloud({ n, xyz, rgbU8 }) {
    const step = Math.max(1, stride | 0);
    const m = Math.ceil(n / step);
    const need = m * 3;
    let replaced = false;
    if (cloudPos.length < need) {
      cloudPos = new Float32Array(need);
      cloudCol = new Uint8Array(need);
      replaced = true;
    }
    let zmin = Infinity, zmax = -Infinity;
    if (colorMode === 'height') {
      for (let k = 0; k < n; k += step) { const z = xyz[k * 3 + 2]; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
    }
    let j = 0;
    for (let k = 0; k < n; k += step, j++) {
      cloudPos[j * 3] = xyz[k * 3]; cloudPos[j * 3 + 1] = xyz[k * 3 + 1]; cloudPos[j * 3 + 2] = xyz[k * 3 + 2];
      if (colorMode === 'height') {
        const t = zmax > zmin ? (xyz[k * 3 + 2] - zmin) / (zmax - zmin) : 0.5;
        cloudCol[j * 3] = Math.round(255 * t);
        cloudCol[j * 3 + 1] = Math.round(255 * (0.4 + 0.4 * (1 - Math.abs(t - 0.5) * 2)));
        cloudCol[j * 3 + 2] = Math.round(255 * (1 - t));
      } else {
        cloudCol[j * 3] = rgbU8[k * 3];
        cloudCol[j * 3 + 1] = rgbU8[k * 3 + 1];
        cloudCol[j * 3 + 2] = rgbU8[k * 3 + 2];
      }
    }
    const g = cloud.geometry;
    if (replaced || g.getAttribute('position')?.array !== cloudPos) {
      g.setAttribute('position', new THREE.BufferAttribute(cloudPos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(cloudCol, 3, true));
    } else {
      g.getAttribute('position').needsUpdate = true;
      g.getAttribute('color').needsUpdate = true;
    }
    g.setDrawRange(0, j);
  }

  function fetchFrameGeo(i) {
    if (fgCache.has(i)) return fgCache.get(i);
    const variant = encodeURIComponent(variantId);
    const p = api('GET', `/api/pipeline/episode/${epId}/geometry?frame=${i}&variant=${variant}`)
      .catch(e => { fgCache.delete(i); throw e; });
    fgCache.set(i, p);
    if (fgCache.size > CACHE_CAP) fgCache.delete(fgCache.keys().next().value);
    return p;
  }

  function clearCloud() {
    cloud.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    cloud.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
  }

  // ── per-frame skeletons ───────────────────────────────────────────────
  function updateHandFit(caps) {
    if (!Array.isArray(caps) || caps.length < 16) {
      handBones.count = handJoints.count = 0;
      return;
    }
    // Capsule 0 is palm; then 3 phalanges for thumb/index/middle/ring/pinky.
    // Rebuild the 21 MediaPipe joints, then draw them like every other hand.
    const joints = [caps[0][0]];
    for (let start = 1; start < 16; start += 3) {
      joints.push(caps[start][0], caps[start][1], caps[start + 1][1], caps[start + 2][1]);
    }
    drawCapsuleHand(handBones, handJoints, joints, HAND_BONE_R, HAND_JOINT_R);
  }

  function updateFrameGeometry(fg) {
    // Hidden diagnostic layers must be computationally hidden too — per-camera
    // skeletons are off by default, so skip the packing entirely when hidden.
    if (layers.perCamera) {
      const per = fg?.per_camera || {};
      let bOff = 0, jOff = 0;
      Object.keys(per).slice(0, MAX_SKEL_CAMS).forEach((dev, i) => {
        const r = packCapsuleHand(
          perCamBones, perCamJoints, bOff, jOff, per[dev].points,
          HAND_BONE_R, HAND_JOINT_R, CAM_COLORS[i % CAM_COLORS.length]);
        bOff += r.nb; jOff += r.nj;
      });
      perCamBones.count = bOff; perCamJoints.count = jOff;
      perCamBones.instanceMatrix.needsUpdate = true;
      perCamJoints.instanceMatrix.needsUpdate = true;
      if (perCamBones.instanceColor) perCamBones.instanceColor.needsUpdate = true;
      if (perCamJoints.instanceColor) perCamJoints.instanceColor.needsUpdate = true;
    } else {
      perCamBones.count = perCamJoints.count = 0;
    }

    // Fused skeleton → IK: the amber capsule hand.
    if (layers.fused) {
      drawCapsuleHand(fusedBones, fusedJoints, fg?.fused_skeleton, HAND_BONE_R, HAND_JOINT_R);
    } else {
      fusedBones.count = fusedJoints.count = 0;
    }

    // fitted capsule hand: fg.hand_capsules = C×[[ax,ay,az],[bx,by,bz]] world
    if (layers.handFit) updateHandFit(fg?.hand_capsules);
    else handBones.count = handJoints.count = 0;

    // palm frame triad, from the summary geometry
    const T = geo?.wrist_traj, R = geo?.palm_rot;
    const fi = Number.isInteger(fg?.frame) ? fg.frame : frame;
    const origin = T?.[fi], rotation = R?.[fi];
    const have = !!(
      Array.isArray(origin) && origin.length === 3 && origin.every(Number.isFinite)
      && Array.isArray(rotation) && rotation.length === 9 && rotation.every(Number.isFinite)
      && fg?.frame_valid !== false
    );
    palmTriad.userData.have = have;
    if (have) {
      const o = origin, m = rotation;
      palmTriad.position.set(o[0], o[1], o[2]);
      palmTriad.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(
        m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1));
    }
    updateRobotFrame(fi);
    applyLayerVisibility();
  }

  // ── public API ────────────────────────────────────────────────────────
  async function loadEpisode(id, list, variant = 'active') {
    pause();
    const serial = ++loadSerial;
    const episodeChanged = id !== epId;
    loadingEpisode = true;
    if (episodeChanged) {
      cloudCache.clear();
      cmeta = null;
      setRetargetData(null);
      clearCloud();
    }
    epId = id;
    variantId = variant || 'active';
    if (Array.isArray(list)) { episodes = list; epIndex = list.findIndex(e => (e.id || e) === id); }
    fgCache.clear();
    if (!id) {
      clearCloud(); geo = null; loadingEpisode = false;
      return { hasCloud: false };
    }
    try {
      geo = await api('GET', `/api/pipeline/episode/${id}/geometry?variant=${encodeURIComponent(variantId)}`);
      applyWorldDisplay(geo && geo.t_world_display);
    } catch (e) { geo = null; log && log('scene: ' + e, 'error'); }
    if (serial !== loadSerial) return { hasCloud: false };
    buildBoard(); buildBbox(); buildFrusta(); buildTrajectory(); frameCamera();
    if (episodeChanged || !cmeta) {
      try { cmeta = await api('GET', `/api/pipeline/episode/${id}/cloud`); }
      catch { cmeta = null; clearCloud(); }
    }
    if (serial !== loadSerial) return { hasCloud: false };
    frame = 0;
    await setFrame(0);
    if (serial === loadSerial) loadingEpisode = false;
    return { hasCloud: !!cmeta, geo, cmeta };
  }

  async function setFrame(i) {
    const n = nFrames();
    frame = n ? Math.max(0, Math.min(i, n - 1)) : 0;
    const want = frame, episode = epId, sourceVariant = variantId;
    const cloudPromise = cmeta && epId
      ? fetchCloud(want).catch(e => { log && log('' + e, 'error'); return null; })
      : Promise.resolve(null);
    const geometryPromise = epId
      ? fetchFrameGeo(want).catch(e => { log && log('' + e, 'error'); return null; })
      : Promise.resolve(null);
    const [cloudFrame, frameGeometry] = await Promise.all([cloudPromise, geometryPromise]);
    if (want !== frame || episode !== epId || sourceVariant !== variantId || disposed) return false;
    if (cloudFrame) paintCloud(cloudFrame);
    applyLayerVisibility();
    updateFrameGeometry(frameGeometry);
    frameCb && frameCb(frame, n);

    // Geometry is tiny enough for a short runway.  Point clouds are not: four
    // concurrent cloud reads produced a visible wait/burst cycle (several
    // cached frames painted back-to-back, then another stall).  Keep exactly
    // one cloud in flight so playback degrades to a slower steady cadence when
    // storage/decoding cannot sustain the recorded FPS.
    for (let d = 1; d <= 3; d++) {
      const ahead = want + d;
      if (ahead >= n) break;
      fetchFrameGeo(ahead).catch(() => {});
    }
    const nextCloud = want + 1;
    if (cmeta && nextCloud < cmeta.n_frames) fetchCloud(nextCloud).catch(() => {});
    return true;
  }

  function play() {
    if (playing || loadingEpisode) return;
    const n = nFrames();
    if (n < 2) return;
    playing = true;
    const serial = ++playSerial;
    let nextDue = performance.now() + 1000 / fps();
    const advance = async now => {
      if (!playing || serial !== playSerial || disposed) return;
      // The small tolerance makes 29.9 fps land on every second 60 Hz vsync
      // instead of accidentally falling through to every third one.
      if (playPending || now + 1 < nextDue) {
        playTimer = requestAnimationFrame(advance);
        return;
      }
      playPending = true;
      await setFrame((frame + 1) % nFrames());
      playPending = false;
      if (!playing || serial !== playSerial || disposed) return;
      // Never catch up by applying multiple states between two browser paints.
      // A slow frame lowers playback speed instead of looking like a 3-frame jump.
      const interval = 1000 / fps();
      nextDue += interval;
      if (nextDue < performance.now() + 1) nextDue = performance.now() + interval;
      playTimer = requestAnimationFrame(advance);
    };
    playTimer = requestAnimationFrame(advance);
  }
  function pause() {
    playing = false;
    playSerial++;
    if (playTimer) cancelAnimationFrame(playTimer);
    playTimer = 0;
    playPending = false;
  }
  function stop() { pause(); setFrame(0); }
  function togglePlay() { playing ? pause() : play(); return playing; }
  function skipSeconds(dir) {
    setFrame(frame + Math.round(dir * 5 * fps()));
  }
  function step(d) { setFrame(frame + d); }
  function nextEpisode(d) {
    if (!episodes.length) return null;
    epIndex = (epIndex + d + episodes.length) % episodes.length;
    const e = episodes[epIndex];
    const id = e.id || e;
    loadEpisode(id, episodes, 'active');
    return id;
  }

  // These layers are packed on demand in updateFrameGeometry, so switching one
  // back on needs a frame re-pack, not just a visibility flip.
  const LAZY_LAYERS = ['perCamera', 'handFit', 'fused'];
  function setLayer(name, on) {
    const needsRefresh = !!on && !layers[name] && LAZY_LAYERS.includes(name);
    layers[name] = !!on;
    applyLayerVisibility();
    if (needsRefresh) setFrame(frame);
    layerCb && layerCb({ ...layers });
  }
  function setLayers(obj) {
    const needsRefresh = LAZY_LAYERS.some(k => !layers[k] && obj?.[k]);
    layers = { ...layers, ...obj };
    applyLayerVisibility();
    if (needsRefresh) setFrame(frame);
    layerCb && layerCb({ ...layers });
  }
  function onLayerChange(cb) { layerCb = cb; }
  function setColorMode(m) { colorMode = m; applyLayerVisibility(); if (cmeta) setFrame(frame); }
  function setStride(n) { stride = Math.max(1, n | 0); applyLayerVisibility(); if (cmeta) setFrame(frame); }
  function onFrame(cb) { frameCb = cb; }

  function dispose() {
    disposed = true;
    robotMeshToken++;   // strand any in-flight mesh loads
    pause();
    if (raf) cancelAnimationFrame(raf);
    ro.disconnect();
    cloudCache.clear(); fgCache.clear();
    controls.dispose();
    scene.traverse(o => {
      o.geometry?.dispose?.();
      if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
      else o.material?.dispose?.();
    });
    renderer.dispose();
    renderer.forceContextLoss?.();
    renderer.domElement.remove();
    legendEl.remove();
  }

  return {
    loadEpisode, setFrame, step, play, pause, stop, togglePlay, setRetargetData,
    skipSeconds, nextEpisode, setLayer, setLayers, setColorMode, setStride,
    onFrame, onLayerChange, dispose,
    get frame() { return frame; },
    get nFrames() { return nFrames(); },
    get fps() { return fps(); },
    get playing() { return playing; },
    get hasCloud() { return !!cmeta; },
    get meta() { return { geo, cmeta, variantId }; },
    get layerState() { return { ...layers }; },
  };
}
