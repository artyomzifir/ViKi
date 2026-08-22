// Camera calibration: board params, per-device streams, capture, extrinsics.
import { api, log, state, FRONTEND_CONFIG } from './core.js';
import { setCameraExtrinsics, setCalibBoard, setCalibCameras, captureBaseDepth } from './skeleton.js';

const ARUCO_DICTS = [
  'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
  'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
  'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
  'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
  'DICT_ARUCO_ORIGINAL'
];

let calibPollInterval = null;

export function populateArucoDicts() {
  const select = document.getElementById("aruco-dict");
  select.innerHTML = "";
  ARUCO_DICTS.forEach(name => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name == FRONTEND_CONFIG.calibration.aruco.defaultDict) opt.selected = true;
    select.appendChild(opt);
  });
}

export async function toggleBoardFields() {
  const boardType = document.getElementById("board-type").value;
  const arucoFields = document.getElementById("aruco-fields");
  arucoFields.style.display = boardType === 'aruco' ? 'block' : 'none';

  if (boardType === 'chess') {
    document.getElementById('board-width').value = FRONTEND_CONFIG.calibration.chess.boardSize[0];
    document.getElementById('board-height').value = FRONTEND_CONFIG.calibration.chess.boardSize[1];
    document.getElementById('square-size').value = FRONTEND_CONFIG.calibration.chess.squareSize;
  } else {
    document.getElementById('board-width').value = FRONTEND_CONFIG.calibration.aruco.boardSize[0];
    document.getElementById('board-height').value = FRONTEND_CONFIG.calibration.aruco.boardSize[1];
    document.getElementById('square-size').value = FRONTEND_CONFIG.calibration.aruco.squareSize;
    document.getElementById('marker-size').value = FRONTEND_CONFIG.calibration.aruco.markerSize;
  }
  await syncBoardParameters();
  await startCalibrationSession(true);
}

export async function syncBoardParameters() {
  const boardType = document.getElementById("board-type").value;
  const width = parseInt(document.getElementById("board-width").value, 10);
  const height = parseInt(document.getElementById("board-height").value, 10);
  const squareSize = parseFloat(document.getElementById("square-size").value);

  const params = {
    board_size: [width, height],
    square_size: squareSize
  };

  if (boardType === 'aruco') {
    params.marker_size = parseFloat(document.getElementById('marker-size').value);
    params.aruco_dict = document.getElementById('aruco-dict').value;
  }

  try {
    await api('POST', `/api/calibration/sync?board_type=${boardType}`, params);
  } catch (e) {
    log('Parameter sync failed: ' + e, 'error');
  }
}

async function resetCalibration() {
  await api('POST', `/api/calibration/reset`);
}

function populateCalibrationStreams() {
  const container = document.getElementById('calib-streams');
  container.innerHTML = '';
  const deviceIds = Object.keys(state);
  if (deviceIds.length === 0) {
    container.innerHTML = '<div style="color:var(--muted);padding:20px;">No cameras detected</div>';
    return;
  }

  container.className = 'calib-streams';
  deviceIds.forEach(id => {
    const wrapper = document.createElement('div');
    wrapper.className = 'calib-stream-wrapper';
    wrapper.dataset.deviceId = id;

    const imgContainer = document.createElement('div');
    imgContainer.className = 'calib-img-container';

    const img = document.createElement('img');
    img.src = `/api/calibration/${id}/stream?t=${Date.now()}`;
    imgContainer.appendChild(img);

    const label = document.createElement('span');
    label.className = 'calib-stream-label';
    label.innerHTML = `${id} <span class="calib-count" id="calib-count-${id}">(0 samples)</span>`;
    imgContainer.appendChild(label);

    wrapper.appendChild(imgContainer);
    container.appendChild(wrapper);
  });
}

function clearCalibrationStreams() {
  const container = document.getElementById('calib-streams');
  container.querySelectorAll('img').forEach(img => img.src = '');
  container.innerHTML = '';
}

async function updateCalibStatus() {
  const deviceIds = Object.keys(state);
  await Promise.all(deviceIds.map(async (id) => {
    const countEl = document.getElementById(`calib-count-${id}`);
    if (!countEl) return;
    try {
      const data = await api('GET', `/api/calibration/status/${id}?t=${Date.now()}`);

      if (data.started) {
        countEl.textContent = `(${data.samples_count} samples)`;
      } else {
        countEl.textContent = `(0 samples)`;
      }
    } catch (e) {
      countEl.textContent = `(error)`;
      log(`Status check for ${id} failed`, 'error');
    }
  }));
}

export function toggleCalibration() {
  const panel = document.getElementById('calib-panel');
  const isVisible = panel.style.display === 'block';
  panel.style.display = isVisible ? 'none' : 'block';

  if (!isVisible) {
    populateCalibrationStreams();
    startCalibrationSession(false);
    updateCalibStatus();
    calibPollInterval = setInterval(updateCalibStatus, 1000);
  } else {
    if (calibPollInterval) {
      clearInterval(calibPollInterval);
      calibPollInterval = null;
    }
    clearCalibrationStreams();
    resetCalibration();
    document.getElementById('calib-streams').innerHTML = '';
  }
}

export async function captureSample() {
  log('Capturing calibration sample...');
  try {
    updateCalibStatus();
    await api('POST', '/api/calibration/capture');
    log('Sample captured successfully', 'ok');
    updateCalibStatus();
  } catch (e) {
    log(`Capture failed: ${e}`, 'error');
  }
}

async function startCalibrationSession(reset = true) {
  if (reset) {
    await resetCalibration();
  }
  const boardType = document.getElementById('board-type').value;
  log(`Starting calibration session (${boardType} board)...`);

  const width = parseInt(document.getElementById('board-width').value, 10);
  const height = parseInt(document.getElementById('board-height').value, 10);
  const squareSize = parseFloat(document.getElementById('square-size').value);

  let params, endpoint;
  if (boardType === 'chess') {
    params = {
      board_size: [width, height],
      square_size: squareSize
    };
    endpoint = (id) => `/api/calibration/start/${id}`;
  } else { // aruco
    const markerSize = parseFloat(document.getElementById('marker-size').value);
    const arucoDict = document.getElementById('aruco-dict').value;
    params = {
      board_size: [width, height],
      square_size: squareSize,
      marker_size: markerSize,
      aruco_dict: arucoDict
    };
    endpoint = (id) => `/api/calibration/start/aruco/${id}`;
  }

  for (const id of Object.keys(state)) {
    try {
      await api('POST', endpoint(id) + "?mode=manual", params);
    } catch (e) {
      log(`Calibration session start failed: ${e}`, 'error');
    }
  }
  updateCalibStatus();
}

// Defined for parity with the original UI; no button currently triggers it.
async function intrinsicsCalibration() {
  log('Intrinsics calibration started...');
  for (const id of Object.keys(state)) {
    try {
      await api('POST', `/api/calibration/intrinsics/${id}`);
      log(`Intrinsics calibration for ${id} successful`, 'ok');
    } catch (e) {
      log(`Intrinsics calibration for ${id} failed: ${e}`, 'error');
    }
  }
}

export async function extrinsicsCalibration() {
  log(`Extrinsics calibration started...`);
  try {
    const res = await api('POST', '/api/calibration/extrinsics');
    const extrinsics = {};
    res.forEach(extr => {
      extrinsics[extr.device_id] = { rvec: extr.rvec, tvec: extr.tvec };
    });
    setCameraExtrinsics(extrinsics);

    // Fetch viz data for the 3D skeleton panel (board + camera frames)
    try {
      const viz = await api('GET', '/api/calibration/viz');
      setCalibBoard(viz.board);
      setCalibCameras(viz.cameras);
    } catch { /* non-critical */ }

    // Snapshot the now-empty scene as static background depth for every camera
    // so skeleton estimation can subtract it. Run after extrinsics so the
    // board (if still in view) is out of the way.
    await Promise.all(Object.keys(state).map(async (id) => {
      try {
        await captureBaseDepth(id);
      } catch (e) {
        log(`Base depth capture for ${id} failed: ${e}`, 'error');
      }
    }));

    log('Extrinsics calibration successful', 'ok');
  } catch (e) {
    log(`Extrinsics calibration failed: ${e}`, 'error');
  }
}

export async function clearCalibration() {
  for (const id of Object.keys(state)) {
    await api('POST', `/api/calibration/clear/${id}`);
  }
  log('Calibration samples cleared');
  updateCalibStatus();
}
