// scene3d.js — the shared three.js scene for the Viewer and Extract tabs.
// One episode at a time: the ChArUco world (board on the Z=0 plane at the
// origin), the per-frame coloured point cloud, per-camera lifted hand skeletons,
// the fused+smoothed skeleton that goes to IK, the wrist trajectory, a palm
// triad + gripper marker, and camera frusta. Orbit / wheel-zoom / right-drag pan.
//
// create(canvasEl, {api, log}) -> a controller the tab drives.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

// MediaPipe / RTMPose 21-point hand topology.
const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const CAM_PALETTE = [0xe6194b, 0x3cb44b, 0x4363d8, 0xf58231, 0x911eb4, 0x46f0f0];

const DEFAULT_LAYERS = {
  cloud: true, perCamera: false, fused: true, trajectory: true,
  palm: true, frusta: true, board: true, bbox: false, handFit: false,
};

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
  function fatAxes(len = 0.15, radius = 0.003) {
    const g = new THREE.Group();
    const arm = (color, ax) => {
      const geo = new THREE.CylinderGeometry(radius, radius, len, 12);
      geo.translate(0, len / 2, 0);            // base at origin, tip at +len
      if (ax === 'x') geo.rotateZ(-Math.PI / 2);
      if (ax === 'z') geo.rotateX(Math.PI / 2);
      return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color }));
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

  // Everything data-driven lives in the RIG (reference-camera) frame; the
  // world anchor's T_world_display is applied here, for presentation only.
  const worldGroup = new THREE.Group();
  worldGroup.matrixAutoUpdate = false;
  scene.add(worldGroup);

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

  const fusedSkel = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xffd166, linewidth: 2 })
  );
  const fusedPositions = new Float32Array(HAND_EDGES.length * 6);
  const fusedPositionAttr = new THREE.BufferAttribute(fusedPositions, 3);
  fusedSkel.geometry.setAttribute('position', fusedPositionAttr);
  fusedSkel.geometry.setDrawRange(0, 0);
  fusedSkel.frustumCulled = false;
  worldGroup.add(fusedSkel);

  const camSkelGroup = new THREE.Group();
  worldGroup.add(camSkelGroup);

  // Fitted hand: translucent cylinders are used because WebGL ignores line
  // widths. Reconstructing the MediaPipe joints from capsule endpoints lets us
  // draw the complete 21-joint topology, including the palm cross-links.
  const handBones = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(1, 1, 1, 10),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.42,
      depthWrite: false }),
    HAND_EDGES.length
  );
  const handJoints = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 12, 9),
    new THREE.MeshBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.72,
      depthWrite: false }),
    21
  );
  handBones.count = handJoints.count = 0;
  handBones.frustumCulled = handJoints.frustumCulled = false;
  worldGroup.add(handBones, handJoints);

  const palmTriad = new THREE.AxesHelper(0.05);
  palmTriad.visible = false;
  worldGroup.add(palmTriad);
  const gripDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.012, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0x4ade80 })
  );
  gripDot.visible = false;
  worldGroup.add(gripDot);

  // ── state ─────────────────────────────────────────────────────────────
  let geo = null, cmeta = null, epId = null, variantId = 'active', episodes = [], epIndex = -1;
  let frame = 0, playing = false, playTimer = 0, playSerial = 0, playPending = false;
  let colorMode = initColor || 'rgb', stride = initStride || 1;
  let layers = { ...DEFAULT_LAYERS, ...(initLayers || {}) };
  let frameCb = null;
  const cloudCache = new Map();     // frame -> Promise<{xyz, rgb}>
  const fgCache = new Map();        // frame -> Promise<geometry?frame= payload>
  const CACHE_CAP = 80;
  let loadSerial = 0, loadingEpisode = false;
  let cloudPos = new Float32Array(0), cloudCol = new Uint8Array(0);
  let raf = 0, disposed = false;

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
  function nFrames() { return cmeta?.n_frames || geo?.n_frames || 0; }

  function clearGroup(g) {
    while (g.children.length) {
      const c = g.children.pop();
      c.geometry?.dispose?.();
      if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
      else c.material?.dispose?.();
    }
  }

  function applyLayerVisibility() {
    cloud.visible = layers.cloud;
    trajLine.visible = layers.trajectory;
    fusedSkel.visible = layers.fused;
    camSkelGroup.visible = layers.perCamera;
    boardGroup.visible = layers.board;
    bboxGroup.visible = layers.bbox;
    frustaGroup.visible = layers.frusta;
    palmTriad.visible = layers.palm && palmTriad.userData.have;
    gripDot.visible = layers.palm && gripDot.userData.have;
    handBones.visible = layers.handFit;
    handJoints.visible = layers.handFit;
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
  function skelPositions(pts) {
    // pts: 21x3 (values may be null). Returns Float32Array of edge endpoints.
    const seg = [];
    for (const [a, b] of HAND_EDGES) {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb || pa[0] == null || pb[0] == null) continue;
      seg.push(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);
    }
    return new Float32Array(seg);
  }

  function updateHandFit(caps) {
    if (!Array.isArray(caps) || caps.length < 16) {
      handBones.count = handJoints.count = 0;
      return;
    }

    // Capsule 0 is palm; then 3 phalanges for thumb/index/middle/ring/pinky.
    const joints = [caps[0][0]];
    for (let start = 1; start < 16; start += 3) {
      joints.push(caps[start][0], caps[start][1], caps[start + 1][1], caps[start + 2][1]);
    }
    const up = new THREE.Vector3(0, 1, 0);
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    const mid = new THREE.Vector3(), direction = new THREE.Vector3();
    const rotation = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    let boneCount = 0;
    for (const [ia, ib] of HAND_EDGES) {
      a.fromArray(joints[ia]); b.fromArray(joints[ib]);
      direction.subVectors(b, a);
      const length = direction.length();
      if (!Number.isFinite(length) || length < 1e-7) continue;
      mid.addVectors(a, b).multiplyScalar(0.5);
      rotation.setFromUnitVectors(up, direction.multiplyScalar(1 / length));
      scale.set(0.0038, length, 0.0038);
      matrix.compose(mid, rotation, scale);
      handBones.setMatrixAt(boneCount++, matrix);
    }
    handBones.count = boneCount;
    handBones.instanceMatrix.needsUpdate = true;

    let jointCount = 0;
    rotation.identity(); scale.setScalar(0.0052);
    for (const p of joints) {
      a.fromArray(p);
      if (!Number.isFinite(a.x + a.y + a.z)) continue;
      matrix.compose(a, rotation, scale);
      handJoints.setMatrixAt(jointCount++, matrix);
    }
    handJoints.count = jointCount;
    handJoints.instanceMatrix.needsUpdate = true;
  }

  function updateFrameGeometry(fg) {
    // Hidden diagnostic layers must be computationally hidden too. Previously
    // these Three.js objects were destroyed and rebuilt on every frame even
    // though per-camera skeletons are off by default.
    if (layers.perCamera) {
      clearGroup(camSkelGroup);
      const per = fg?.per_camera || {};
      Object.keys(per).forEach((dev, i) => {
        const arr = skelPositions(per[dev].points);
        if (!arr.length) return;
        const ls = new THREE.LineSegments(
          new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(arr, 3)),
          new THREE.LineBasicMaterial({ color: CAM_PALETTE[i % CAM_PALETTE.length] }));
        camSkelGroup.add(ls);
      });
    }

    // Keep one small GPU buffer for the fused skeleton instead of replacing
    // its BufferAttribute (and WebGL buffer) on every frame.
    const fArr = fg?.fused_skeleton ? skelPositions(fg.fused_skeleton) : new Float32Array(0);
    fusedPositions.fill(0);
    fusedPositions.set(fArr.subarray(0, fusedPositions.length));
    fusedPositionAttr.needsUpdate = true;
    fusedSkel.geometry.setDrawRange(0, fArr.length / 3);

    // fitted capsule hand: fg.hand_capsules = C×[[ax,ay,az],[bx,by,bz]] world
    if (layers.handFit) updateHandFit(fg?.hand_capsules);
    else handBones.count = handJoints.count = 0;

    // palm triad + gripper marker from the summary geometry
    const T = geo?.wrist_traj, R = geo?.palm_rot;
    const fi = Number.isInteger(fg?.frame) ? fg.frame : frame;
    const origin = T?.[fi], rotation = R?.[fi];
    const have = !!(
      Array.isArray(origin) && origin.length === 3 && origin.every(Number.isFinite)
      && Array.isArray(rotation) && rotation.length === 9 && rotation.every(Number.isFinite)
      && fg?.frame_valid !== false
    );
    palmTriad.userData.have = gripDot.userData.have = have;
    if (have) {
      const o = origin, m = rotation;
      palmTriad.position.set(o[0], o[1], o[2]);
      palmTriad.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(
        m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1));
      gripDot.position.set(o[0], o[1], o[2] + 0.03);
      gripDot.material.color.set(fg?.gripper ? 0xf87171 : 0x4ade80);
    }
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

  function setLayer(name, on) {
    const needsRefresh = !!on && !layers[name] && (name === 'perCamera' || name === 'handFit');
    layers[name] = !!on;
    applyLayerVisibility();
    if (needsRefresh) setFrame(frame);
  }
  function setLayers(obj) {
    const needsRefresh = (!layers.perCamera && obj?.perCamera)
      || (!layers.handFit && obj?.handFit);
    layers = { ...layers, ...obj };
    applyLayerVisibility();
    if (needsRefresh) setFrame(frame);
  }
  function setColorMode(m) { colorMode = m; applyLayerVisibility(); if (cmeta) setFrame(frame); }
  function setStride(n) { stride = Math.max(1, n | 0); applyLayerVisibility(); if (cmeta) setFrame(frame); }
  function onFrame(cb) { frameCb = cb; }

  function dispose() {
    disposed = true;
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
  }

  return {
    loadEpisode, setFrame, step, play, pause, stop, togglePlay,
    skipSeconds, nextEpisode, setLayer, setLayers, setColorMode, setStride,
    onFrame, dispose,
    get frame() { return frame; },
    get nFrames() { return nFrames(); },
    get fps() { return fps(); },
    get playing() { return playing; },
    get hasCloud() { return !!cmeta; },
    get meta() { return { geo, cmeta, variantId }; },
    get layerState() { return { ...layers }; },
  };
}
