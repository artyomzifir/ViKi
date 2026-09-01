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
  cloud: true, perCamera: true, fused: true, trajectory: true,
  palm: true, frusta: true, board: true, bbox: false, handFit: true,
};

export function create(canvasEl, { api, log, layers: initLayers, colorMode: initColor, stride: initStride }) {
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
  const frustaGroup = new THREE.Group();
  scene.add(frustaGroup);

  // ── dynamic ───────────────────────────────────────────────────────────
  const cloud = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.006, vertexColors: true, sizeAttenuation: true })
  );
  cloud.frustumCulled = false;
  scene.add(cloud);

  const trajLine = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x9aa4b2 })
  );
  scene.add(trajLine);

  const fusedSkel = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xffd166, linewidth: 2 })
  );
  scene.add(fusedSkel);

  const camSkelGroup = new THREE.Group();
  scene.add(camSkelGroup);

  // fitted capsule hand (PERCEPTION_HAND_FIT): thin bones + fatter, darker
  // joint blobs sized to the calibrated capsule radius.
  const handBones = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x7dd3fc })
  );
  scene.add(handBones);
  const handJoints = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0x0369a1 }),
    64
  );
  handJoints.count = 0;
  handJoints.frustumCulled = false;
  scene.add(handJoints);

  const palmTriad = new THREE.AxesHelper(0.05);
  palmTriad.visible = false;
  scene.add(palmTriad);
  const gripDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.012, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0x4ade80 })
  );
  gripDot.visible = false;
  scene.add(gripDot);

  // ── state ─────────────────────────────────────────────────────────────
  let geo = null, cmeta = null, epId = null, episodes = [], epIndex = -1;
  let frame = 0, playing = false, playTimer = 0;
  let colorMode = initColor || 'rgb', stride = initStride || 1;
  let layers = { ...DEFAULT_LAYERS, ...(initLayers || {}) };
  let frameCb = null;
  const cloudCache = new Map();     // frame -> Promise<{xyz, rgb}>
  const fgCache = new Map();        // frame -> Promise<geometry?frame= payload>
  const CACHE_CAP = 80;
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
    handBones.visible = handJoints.visible = layers.handFit;
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
    const arr = new Float32Array(T.length * 3);
    T.forEach((p, i) => { arr[i * 3] = p[0]; arr[i * 3 + 1] = p[1]; arr[i * 3 + 2] = p[2]; });
    trajLine.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    trajLine.geometry.setDrawRange(0, T.length);
    trajLine.geometry.computeBoundingSphere();
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
        const xyz = new Float32Array(buf.slice(4, 4 + n * 12));
        const rgbU8 = new Uint8Array(buf, 4 + n * 12, n * 3);
        return { n, xyz, rgbU8 };
      });
    cloudCache.set(i, p);
    if (cloudCache.size > CACHE_CAP) cloudCache.delete(cloudCache.keys().next().value);
    return p;
  }

  function paintCloud({ n, xyz, rgbU8 }) {
    const step = Math.max(1, stride | 0);
    const m = Math.ceil(n / step);
    const pos = new Float32Array(m * 3);
    const col = new Float32Array(m * 3);
    let zmin = Infinity, zmax = -Infinity;
    if (colorMode === 'height') {
      for (let k = 0; k < n; k += step) { const z = xyz[k * 3 + 2]; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
    }
    let j = 0;
    for (let k = 0; k < n; k += step, j++) {
      pos[j * 3] = xyz[k * 3]; pos[j * 3 + 1] = xyz[k * 3 + 1]; pos[j * 3 + 2] = xyz[k * 3 + 2];
      if (colorMode === 'height') {
        const t = zmax > zmin ? (xyz[k * 3 + 2] - zmin) / (zmax - zmin) : 0.5;
        col[j * 3] = t; col[j * 3 + 1] = 0.4 + 0.4 * (1 - Math.abs(t - 0.5) * 2); col[j * 3 + 2] = 1 - t;
      } else {
        col[j * 3] = rgbU8[k * 3] / 255;
        col[j * 3 + 1] = rgbU8[k * 3 + 1] / 255;
        col[j * 3 + 2] = rgbU8[k * 3 + 2] / 255;
      }
    }
    const g = cloud.geometry;
    g.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, j * 3), 3));
    g.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, j * 3), 3));
    g.setDrawRange(0, j);
    g.computeBoundingSphere();
  }

  function fetchFrameGeo(i) {
    if (fgCache.has(i)) return fgCache.get(i);
    const p = api('GET', `/api/pipeline/episode/${epId}/geometry?frame=${i}`);
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

  const _m4 = new THREE.Matrix4();
  function updateHandFit(caps) {
    if (!Array.isArray(caps) || !caps.length) {
      handBones.geometry.setDrawRange(0, 0);
      handJoints.count = 0;
      return;
    }
    const radii = geo?.hand_capsule_radii || [];
    const seg = new Float32Array(caps.length * 6);
    let ji = 0;
    for (let c = 0; c < caps.length; c++) {
      const a = caps[c][0], b = caps[c][1];
      seg.set([a[0], a[1], a[2], b[0], b[1], b[2]], c * 6);
      const r = Math.max(radii[c] || 0.008, 0.004);
      for (const p of (caps[c])) {
        if (ji >= handJoints.count && ji >= 64) break;
        _m4.makeScale(r, r, r).setPosition(p[0], p[1], p[2]);
        handJoints.setMatrixAt(ji++, _m4);
      }
    }
    handBones.geometry.setAttribute('position', new THREE.BufferAttribute(seg, 3));
    handBones.geometry.setDrawRange(0, seg.length / 3);
    handBones.geometry.computeBoundingSphere();
    handJoints.count = Math.min(ji, 64);
    handJoints.instanceMatrix.needsUpdate = true;
  }

  function updateFrameGeometry(fg) {
    // per-camera skeletons
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

    // fused skeleton
    const fArr = fg?.fused_skeleton ? skelPositions(fg.fused_skeleton) : new Float32Array(0);
    fusedSkel.geometry.setAttribute('position', new THREE.BufferAttribute(fArr, 3));
    fusedSkel.geometry.setDrawRange(0, fArr.length / 3);
    fusedSkel.geometry.computeBoundingSphere();

    // fitted capsule hand: fg.hand_capsules = C×[[ax,ay,az],[bx,by,bz]] world
    updateHandFit(fg?.hand_capsules);

    // palm triad + gripper marker from the summary geometry
    const T = geo?.wrist_traj, R = geo?.palm_rot;
    const have = !!(T && R && T[frame] && R[frame]);
    palmTriad.userData.have = gripDot.userData.have = have;
    if (have) {
      const o = T[frame], m = R[frame];
      palmTriad.position.set(o[0], o[1], o[2]);
      palmTriad.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(
        m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1));
      gripDot.position.set(o[0], o[1], o[2] + 0.03);
      gripDot.material.color.set(fg?.gripper ? 0xf87171 : 0x4ade80);
    }
    applyLayerVisibility();
  }

  // ── public API ────────────────────────────────────────────────────────
  async function loadEpisode(id, list) {
    epId = id;
    if (Array.isArray(list)) { episodes = list; epIndex = list.findIndex(e => (e.id || e) === id); }
    cloudCache.clear(); fgCache.clear();
    cmeta = null;
    if (!id) { clearCloud(); geo = null; return { hasCloud: false }; }
    try {
      geo = await api('GET', `/api/pipeline/episode/${id}/geometry`);
    } catch (e) { geo = null; log && log('scene: ' + e, 'error'); }
    buildBoard(); buildBbox(); buildFrusta(); buildTrajectory(); frameCamera();
    try { cmeta = await api('GET', `/api/pipeline/episode/${id}/cloud`); }
    catch { cmeta = null; clearCloud(); }
    frame = 0;
    await setFrame(0);
    return { hasCloud: !!cmeta, geo, cmeta };
  }

  async function setFrame(i) {
    const n = nFrames();
    frame = n ? Math.max(0, Math.min(i, n - 1)) : 0;
    // cloud
    if (cmeta && epId) {
      for (let d = 0; d <= 3; d++) if (frame + d < cmeta.n_frames) fetchCloud(frame + d);
      const want = frame;
      fetchCloud(frame).then(c => { if (want === frame && !disposed) paintCloud(c); })
        .catch(e => log && log('' + e, 'error'));
    }
    // per-frame geometry (skeletons + gripper), cached
    if (epId) {
      const want = frame;
      try {
        const fg = await fetchFrameGeo(frame);
        if (want === frame && !disposed) updateFrameGeometry(fg);
      } catch { updateFrameGeometry(null); }
    }
    frameCb && frameCb(frame, n);
  }

  function play() {
    if (playing) return;
    const n = nFrames();
    if (n < 2) return;
    playing = true;
    playTimer = setInterval(() => setFrame((frame + 1) % nFrames()),
      Math.max(20, 1000 / fps()));
  }
  function pause() { playing = false; clearInterval(playTimer); playTimer = 0; }
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
    loadEpisode(id, episodes);
    return id;
  }

  function setLayer(name, on) { layers[name] = !!on; applyLayerVisibility(); }
  function setLayers(obj) { layers = { ...layers, ...obj }; applyLayerVisibility(); }
  function setColorMode(m) { colorMode = m; if (cmeta) fetchCloud(frame).then(paintCloud); }
  function setStride(n) { stride = Math.max(1, n | 0); if (cmeta) fetchCloud(frame).then(paintCloud); }
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
    get meta() { return { geo, cmeta }; },
    get layerState() { return { ...layers }; },
  };
}
