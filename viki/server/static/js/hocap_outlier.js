import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const params = new URLSearchParams(location.search);
const dataUrl = params.get('data');
const summary = document.getElementById('summary');
const frameSlider = document.getElementById('frame');
const metrics = document.getElementById('metrics');
const chart = document.getElementById('chart');
let review;
let frame = 0;
let playing = false;

function makePoseGroup(points, color, size) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
  const group = new THREE.Group();
  group.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color, size, sizeAttenuation: true })));
  group.add(new THREE.AxesHelper(0.09));
  group.matrixAutoUpdate = false;
  return group;
}

function setPose(group, matrix) {
  group.visible = !!matrix;
  if (matrix) {
    group.matrix.set(...matrix.flat());
    group.updateMatrixWorld(true);
  }
}

function nearestImageFrame(value) {
  return review.image_frames.reduce((best, current) =>
    Math.abs(current - value) < Math.abs(best - value) ? current : best);
}

function drawChart() {
  const width = chart.clientWidth;
  const height = chart.clientHeight;
  const scale = devicePixelRatio || 1;
  chart.width = Math.round(width * scale);
  chart.height = Math.round(height * scale);
  const context = chart.getContext('2d');
  context.scale(scale, scale);
  const values = review.rotation_error_deg;
  context.strokeStyle = '#ff6356';
  context.lineWidth = 1.5;
  context.beginPath();
  values.forEach((value, index) => {
    if (value === null) return;
    const x = index / (values.length - 1) * width;
    const y = height - 8 - value / 180 * (height - 18);
    if (index && values[index - 1] !== null) context.lineTo(x, y);
    else context.moveTo(x, y);
  });
  context.stroke();
  context.strokeStyle = '#dfe8f2';
  context.beginPath();
  const x = frame / (values.length - 1) * width;
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
  context.fillStyle = '#bbc9d8';
  context.font = '12px system-ui';
  context.fillText('180°', 8, 14);
  context.fillText('0°', 8, height - 5);
}

function startScene() {
  const mount = document.getElementById('scene');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#0d1218');
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  mount.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const predicted = makePoseGroup(review.points_gt_local, '#ff6356', 0.006);
  const truth = makePoseGroup(review.points_gt_local, '#67d9ef', 0.006);
  scene.add(predicted, truth);
  const positions = review.gt_pose.filter(Boolean).map(pose => new THREE.Vector3(
    pose[0][3], pose[1][3], pose[2][3]));
  const bounds = new THREE.Box3().setFromPoints(positions);
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(0.25, bounds.getSize(new THREE.Vector3()).length());
  const grid = new THREE.GridHelper(radius * 3, 12, '#547084', '#293746');
  grid.rotation.x = Math.PI / 2;
  grid.position.z = bounds.min.z - 0.12;
  scene.add(grid);
  camera.position.set(center.x + radius, center.y - radius * 1.2, center.z + radius * 0.8);
  controls.target.copy(center);
  controls.update();
  function resize() {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();
  function render() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();
  return { predicted, truth };
}

async function main() {
  if (!dataUrl) throw new Error('Missing ?data=/static/.../review.json');
  const response = await fetch(dataUrl);
  if (!response.ok) throw new Error(`Review data: HTTP ${response.status}`);
  review = await response.json();
  const scene = startScene();
  frameSlider.max = review.frame_count - 1;
  summary.textContent = `${review.episode} · ${review.object} · mean rotation error ${review.mean_rotation_deg.toFixed(1)}° · peak frame ${review.peak_frame}`;
  function show(next) {
    frame = Math.max(0, Math.min(review.frame_count - 1, Number(next)));
    frameSlider.value = frame;
    setPose(scene.predicted, review.predicted_pose_aligned[frame]);
    setPose(scene.truth, review.gt_pose[frame]);
    const rotation = review.rotation_error_deg[frame];
    const translation = review.translation_error_mm[frame];
    metrics.textContent = `frame ${frame}/${review.frame_count - 1} · rotation ${rotation === null ? '—' : rotation.toFixed(1) + '°'} · translation ${translation === null ? '—' : translation.toFixed(1) + ' mm'}`;
    const imageFrame = nearestImageFrame(frame);
    review.camera_ids.forEach((cameraId, index) => {
      document.getElementById(`camera-${index}`).src = `${new URL(dataUrl, location.href).pathname.replace(/review\.json$/, '')}frames/${cameraId}-${String(imageFrame).padStart(6, '0')}.jpg`;
      document.getElementById(`camera-label-${index}`).textContent = `Camera ${cameraId} · RGB frame ${imageFrame}`;
    });
    drawChart();
  }
  frameSlider.addEventListener('input', () => show(frameSlider.value));
  chart.addEventListener('click', event => show(Math.round(event.offsetX / chart.clientWidth * (review.frame_count - 1))));
  document.getElementById('anchor').onclick = () => show(review.anchor_frame);
  document.getElementById('before').onclick = () => show(Math.max(0, review.peak_frame - 20));
  document.getElementById('peak').onclick = () => show(review.peak_frame);
  document.getElementById('play').onclick = event => {
    playing = !playing;
    event.currentTarget.textContent = playing ? '■ Pause' : '▶ Play';
  };
  setInterval(() => { if (playing) show((frame + 1) % review.frame_count); }, 100);
  window.addEventListener('resize', drawChart);
  show(review.peak_frame);
}

main().catch(error => { summary.textContent = `Could not load review: ${error.message}`; });
