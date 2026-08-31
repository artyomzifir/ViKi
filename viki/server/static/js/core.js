// Core shared foundation: HTTP helper, logger, device state, frontend config.
// FRONTEND_CONFIG / CAMERA_CONFIG are exported as live bindings — importers see
// the values assigned by initializeFrontendConfig().

export let FRONTEND_CONFIG = {};
export let CAMERA_CONFIG = {};

export const state = {}; // device_id -> { running, type, infoInterval }

// Per-page-session settings that survive tab switches (a tab reads its widgets
// from here on mount and writes back on change). NOT persisted to the server —
// that's the Config modal's job; this just stops params snapping back to
// defaults every time you leave and re-enter a tab.
export const session = {};

export function sessionGet(key, fallback) {
  return key in session ? session[key] : fallback;
}
export function sessionSet(key, value) {
  session[key] = value;
  return value;
}
// merge a partial into an object-valued session key
export function sessionPatch(key, partial) {
  session[key] = { ...(session[key] || {}), ...partial };
  return session[key];
}

// ── Logging ────────────────────────────────────────────────────────────────
// One rolling buffer. The newest entry shows in the header's one-line slot
// (#log-line); the full list renders into the popover (#log-popover .log-list).
// log(msg, cls) keeps its old signature so every call site is untouched.

const LOG = [];
const LOG_CAP = 200;

function fmt(entry) {
  return `[${entry.time}] ${entry.msg}`;
}

function renderLog() {
  const line = document.getElementById('log-line');
  if (line) {
    const last = LOG[LOG.length - 1];
    line.textContent = last ? fmt(last) : '';
    line.className = 'log-line ' + (last ? last.cls : '');
  }
  const list = document.querySelector('#log-popover .log-list');
  if (list) {
    list.innerHTML = LOG.slice().reverse()
      .map(e => `<div class="entry ${e.cls}">${fmt(e)}</div>`)
      .join('');
  }
}

export function log(msg, cls = '') {
  LOG.push({ time: new Date().toLocaleTimeString(), msg: String(msg), cls });
  if (LOG.length > LOG_CAP) LOG.shift();
  renderLog();
}

// Called once by main.js after the shell DOM exists, so the first paint of the
// header/popover reflects whatever was logged during boot.
export function mountLog() {
  renderLog();
}

export async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) {
    throw new Error(data.detail || 'API request failed');
  }
  return data;
}

export async function initializeFrontendConfig(config) {
  FRONTEND_CONFIG = {
    cameras: {
      realsense: {
        defaultRes: `${config.DEFAULT_COLOR_WIDTH}x${config.DEFAULT_COLOR_HEIGHT}`,
        defaultFps: config.DEFAULT_FPS,
      },
      kinect: {
        defaultRes: `${config.DEFAULT_COLOR_WIDTH}x${config.DEFAULT_COLOR_HEIGHT}`,
        defaultFps: config.DEFAULT_FPS,
        defaultDepth: config.DEFAULT_DEPTH_MODE,
      }
    },
    calibration: {
      chess: {
        boardSize: config.CALIB_CHESS_BOARD_SIZE,
        squareSize: config.CALIB_CHESS_SQUARE_SIZE,
      },
      aruco: {
        boardSize: config.CALIB_ARUCO_BOARD_SIZE,
        squareSize: config.CALIB_ARUCO_SQUARE_SIZE,
        markerSize: config.CALIB_ARUCO_MARKER_SIZE,
        defaultDict: 'DICT_5X5_50', // This is hardcoded as a string for UI, mapping to ID 4
      }
    },
    recording: {
      duration: config.RECORDING_DURATION,
      fps: config.RECORDING_FPS,
    },
    cloud: {
      stride: config.CLOUD_STRIDE ?? 1,
      voxel: config.CLOUD_VOXEL_M ?? 0.005,
      maxPoints: config.CLOUD_MAX_POINTS_PER_FRAME ?? 40000,
      bbox: config.CLOUD_WORKSPACE_BBOX ?? [-0.6, 0.6, -0.6, 0.6, -0.2, 1.2],
    },
  };

  CAMERA_CONFIG = {
    realsense: {
      resolutions: config.REALSENSE_RESOLUTIONS,
      defaultRes: FRONTEND_CONFIG.cameras.realsense.defaultRes,
      fps: config.REALSENSE_FPS,
      defaultFps: FRONTEND_CONFIG.cameras.realsense.defaultFps,
      depthModes: null,
    },
    kinect: {
      resolutions: config.KINECT_RESOLUTIONS,
      defaultRes: FRONTEND_CONFIG.cameras.kinect.defaultRes,
      fps: config.KINECT_FPS,
      defaultFps: FRONTEND_CONFIG.cameras.kinect.defaultFps,
      depthModes: config.KINECT_DEPTH_MODES,
      defaultDepth: FRONTEND_CONFIG.cameras.kinect.defaultDepth,
      depthModeMaxFps: config.KINECT_DEPTH_MODE_MAX_FPS,
    },
  };
}
