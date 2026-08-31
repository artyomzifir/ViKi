// Server configuration modal. Curated typed widgets for the keys that matter
// day-to-day; everything else stays editable as raw JSON under "Advanced".
// Single source of truth is CFG (the full object); curated widgets write through
// to it and the Advanced textarea mirrors it.
import { api, log, initializeFrontendConfig } from './core.js';

let CFG = {};
let dirty = false;

// group | key | label | widget: number | int | toggle | select | csvInt | text
const CURATED = [
  ['Capture', 'DEFAULT_FPS', 'Default FPS', 'int'],
  ['Capture', 'DEFAULT_COLOR_WIDTH', 'Colour width', 'int'],
  ['Capture', 'DEFAULT_COLOR_HEIGHT', 'Colour height', 'int'],
  ['Capture', 'DEFAULT_DEPTH_MODE', 'Depth mode', 'select', 'KINECT_DEPTH_MODES'],
  ['Capture', 'JPEG_QUALITY', 'JPEG quality', 'int'],
  ['Capture', 'RECORD_DEPTH', 'Save depth .npy', 'toggle'],
  ['Capture', 'FRAME_BUFFER_SIZE', 'Ring buffer size', 'int'],
  ['Calibration', 'CALIB_BOARD_TYPE', 'Board type', 'select', ['chess', 'aruco']],
  ['Calibration', 'CALIB_ARUCO_BOARD_SIZE', 'ChArUco board [cols,rows]', 'csvInt'],
  ['Calibration', 'CALIB_ARUCO_SQUARE_SIZE', 'ChArUco square (m)', 'number'],
  ['Calibration', 'CALIB_ARUCO_MARKER_SIZE', 'ChArUco marker (m)', 'number'],
  ['Calibration', 'CALIB_ARUCO_DICT', 'ArUco dict id', 'int'],
  ['Calibration', 'CALIB_CHESS_BOARD_SIZE', 'Chess board [cols,rows]', 'csvInt'],
  ['Calibration', 'CALIB_CHESS_SQUARE_SIZE', 'Chess square (m)', 'number'],
  ['Recording', 'RECORDING_DURATION', 'Default duration (s)', 'int'],
  ['Recording', 'RECORDING_FPS', 'Recording FPS', 'int'],
  ['Cloud', 'CLOUD_STRIDE', 'Depth pixel stride', 'int'],
  ['Cloud', 'CLOUD_VOXEL_M', 'Voxel leaf size (m)', 'number'],
  ['Cloud', 'CLOUD_WORKSPACE_BBOX', 'Workspace AABB [x0,x1,y0,y1,z0,z1]', 'csvFloat'],
  ['Cloud', 'CLOUD_MAX_POINTS_PER_FRAME', 'Max points / frame', 'int'],
];

function widgetHTML([group, key, label, widget, opt]) {
  if (!(key in CFG)) return '';
  const v = CFG[key];
  let control;
  if (widget === 'toggle') {
    control = `<input type="checkbox" data-key="${key}" data-w="toggle" ${v ? 'checked' : ''}>`;
  } else if (widget === 'select') {
    const opts = Array.isArray(opt) ? opt : (CFG[opt] || []);
    control = `<select data-key="${key}" data-w="select">${
      opts.map(o => `<option ${String(o) === String(v) ? 'selected' : ''}>${o}</option>`).join('')
    }</select>`;
  } else if (widget === 'csvInt' || widget === 'csvFloat') {
    control = `<input type="text" data-key="${key}" data-w="${widget}" value="${(v || []).join(', ')}">`;
  } else if (widget === 'number' || widget === 'int') {
    control = `<input type="number" data-key="${key}" data-w="${widget}"
      ${widget === 'number' ? 'step="0.001"' : ''} value="${v}">`;
  } else {
    control = `<input type="text" data-key="${key}" data-w="text" value="${v}">`;
  }
  return `<div class="cfg-row"><label>${label}</label>${control}</div>`;
}

function render() {
  const body = document.querySelector('#config-modal .cfg-body');
  if (!body) return;
  const groups = [...new Set(CURATED.map(r => r[0]))];
  const curated = groups.map(g => `
    <div class="cfg-group">
      <div class="cfg-group-title">${g}</div>
      ${CURATED.filter(r => r[0] === g).map(widgetHTML).join('')}
    </div>`).join('');
  body.innerHTML = `
    ${curated}
    <details class="cfg-advanced">
      <summary>Advanced (raw JSON)</summary>
      <textarea id="cfg-json" spellcheck="false"></textarea>
    </details>`;
  body.querySelector('#cfg-json').value = JSON.stringify(CFG, null, 2);
  body.querySelectorAll('[data-key]').forEach(el => {
    el.addEventListener('change', onWidgetChange);
  });
  body.querySelector('#cfg-json').addEventListener('blur', onJsonBlur);
  setDirty(dirty);
}

function onWidgetChange(e) {
  const el = e.target;
  const key = el.dataset.key;
  const w = el.dataset.w;
  let val;
  if (w === 'toggle') val = el.checked;
  else if (w === 'int') val = parseInt(el.value, 10);
  else if (w === 'number') val = parseFloat(el.value);
  else if (w === 'csvInt') val = el.value.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !Number.isNaN(n));
  else if (w === 'csvFloat') val = el.value.split(',').map(s => parseFloat(s.trim())).filter(n => !Number.isNaN(n));
  else val = el.value;
  CFG[key] = val;
  const ta = document.getElementById('cfg-json');
  if (ta) ta.value = JSON.stringify(CFG, null, 2);
  setDirty(true);
}

function onJsonBlur(e) {
  try {
    CFG = JSON.parse(e.target.value);
    setDirty(true);
    render();
  } catch {
    log('Config JSON is invalid — not applied', 'error');
  }
}

function setDirty(on) {
  dirty = on;
  const b = document.querySelector('#config-modal .cfg-banner');
  if (b) b.hidden = !on;
}

export async function open() {
  try {
    CFG = await api('GET', '/api/config');
    dirty = false;
  } catch (e) {
    log('Failed to load config: ' + e, 'error');
    return;
  }
  const modal = document.getElementById('config-modal');
  if (modal) modal.hidden = false;
  render();
}

export function close() {
  const modal = document.getElementById('config-modal');
  if (modal) modal.hidden = true;
}

export function toggle() {
  const modal = document.getElementById('config-modal');
  if (!modal) return;
  if (modal.hidden) open(); else close();
}

export async function save() {
  const ta = document.getElementById('cfg-json');
  if (ta) {
    try { CFG = JSON.parse(ta.value); }
    catch { log('Config JSON is invalid — not saved', 'error'); return; }
  }
  try {
    await api('POST', '/api/config', CFG);
    log('Configuration saved — restart to apply', 'ok');
    setDirty(true);
    // Refresh the in-memory frontend config so tabs pick up new defaults, and
    // ask the router to re-mount the current tab.
    await initializeFrontendConfig(CFG);
    document.dispatchEvent(new CustomEvent('config:saved'));
  } catch (e) {
    log('Failed to save config: ' + e, 'error');
  }
}

// Two-click confirm on a modal button: first click arms it for 3 s, second runs.
const _armed = {};
function twoStep(action, run) {
  const btn = document.querySelector(`#config-modal [data-action="${action}"]`);
  if (_armed[action]) {
    clearTimeout(_armed[action]);
    delete _armed[action];
    if (btn) { btn.textContent = btn.dataset.label; btn.classList.remove('armed'); }
    run();
    return;
  }
  if (btn) {
    btn.dataset.label = btn.dataset.label || btn.textContent;
    btn.textContent = 'Click again to confirm';
    btn.classList.add('armed');
  }
  _armed[action] = setTimeout(() => {
    delete _armed[action];
    if (btn) { btn.textContent = btn.dataset.label; btn.classList.remove('armed'); }
  }, 3000);
}

export function reset() {
  twoStep('cfgReset', async () => {
    try {
      await api('POST', '/api/config/reset');
      log('Configuration reset to defaults — restart to apply', 'ok');
      await open();
      setDirty(true);
    } catch (e) { log('Failed to reset config: ' + e, 'error'); }
  });
}

export function restart() {
  twoStep('cfgRestart', async () => {
    try {
      await api('POST', '/api/restart');
      log('Restarting server… please wait.', 'ok');
    } catch (e) { log('Restart request failed: ' + e, 'error'); }
  });
}

// Let a tab that edited a shared key refresh an open modal.
export function refresh() {
  const modal = document.getElementById('config-modal');
  if (modal && !modal.hidden) open();
}
