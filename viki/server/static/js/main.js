// Entry point: wire event delegation (data-action = click, data-change = change),
// then bootstrap the app. Handlers live in feature modules; this file only maps
// declarative attributes to them and extracts arguments from the element.
import { api, log, initializeFrontendConfig } from './core.js';
import * as config from './config.js';
import * as cameras from './cameras.js';
import * as calibration from './calibration.js';
import * as skeleton from './skeleton.js';
import * as process from './process.js';
import * as robotviz from './robotviz.js';

// data-action -> handler (click). Handlers read any args from the element dataset.
const CLICK_ACTIONS = {
  // cameras / toolbar
  scanDevices: () => cameras.scanDevices(),
  startAll: () => cameras.startAll(),
  stopAll: () => cameras.stopAll(),
  startCamera: el => cameras.startCamera(el.dataset.id),
  stopCamera: el => cameras.stopCamera(el.dataset.id),
  startRGBDRecording: () => cameras.startRGBDRecording(),
  // config
  toggleConfig: () => config.toggleConfig(),
  toggleConfigHelp: () => config.toggleConfigHelp(),
  loadConfig: () => config.loadConfig(),
  resetConfig: () => config.resetConfig(),
  saveConfig: () => config.saveConfig(),
  restartServer: () => config.restartServer(),
  // calibration
  toggleCalibration: () => calibration.toggleCalibration(),
  captureSample: () => calibration.captureSample(),
  extrinsicsCalibration: () => calibration.extrinsicsCalibration(),
  clearCalibration: () => calibration.clearCalibration(),
  // skeleton
  toggleSkeleton: () => skeleton.toggleSkeleton(),
  toggleSkelView: () => skeleton.toggleSkelView(),
  toggleEstimation: el => skeleton.toggleEstimation(el.dataset.enable === 'true'),
  toggleRecording: el => skeleton.toggleRecording(el.dataset.enable === 'true'),
  toggleCalibOverlay: () => skeleton.toggleCalibOverlay(),
  toggleDepthDebug: () => skeleton.toggleDepthDebug(),
  // process
  toggleProcess: () => process.toggleProcess(),
  setProcessMode: el => process.setProcessMode(el.dataset.mode),
  loadProcessRecordings: () => process.loadProcessRecordings(),
  processPrevPage: () => process.processPrevPage(),
  processNextPage: () => process.processNextPage(),
  processSmoothSelected: () => process.processSmoothSelected(),
  selectProcessRec: el => process.selectProcessRec(el.dataset.filename, el),
  // process - dataset mode
  loadPreparedRecordings: () => process.loadPreparedRecordings(),
  selectPreparedRec: el => process.selectPreparedRec(el.dataset.filename, el),
  selectDsCLNRec: el => process.selectDsCLNRec(el.dataset.filename, el),
  convertDataset: () => process.convertDataset(),
  loadVizOutputs: () => process.loadVizOutputs(),
  selectVizOutput: el => process.selectVizOutput(el.dataset.filename, el),
  // robot viz (redirect to process in dataset mode)
  toggleRobotViz: () => robotviz.toggleRobotViz(),
};

// data-change -> handler (change on inputs/selects).
const CHANGE_ACTIONS = {
  updateFpsForDepthMode: el => cameras.updateFpsForDepthMode(el.dataset.id),
  toggleBoardFields: () => calibration.toggleBoardFields(),
  syncBoardParameters: () => calibration.syncBoardParameters(),
  updateSkelVizCam: () => skeleton.updateSkelVizCam(),
  toggleFollowEE: () => skeleton.toggleFollowEE(),
  // smoothing viz config changes
  applySmoothConfig: () => process.applySmoothConfig(),
  // robot viz config changes (now in process panel)
  applyVizConfig: () => process.applyVizConfig(),
};

document.addEventListener('click', e => {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const fn = CLICK_ACTIONS[el.dataset.action];
  if (fn) fn(el, e);
});

document.addEventListener('change', e => {
  const el = e.target.closest('[data-change]');
  if (!el) return;
  const fn = CHANGE_ACTIONS[el.dataset.change];
  if (fn) fn(el, e);
});

async function init() {
  try {
    const cfg = await api('GET', '/api/config');
    await initializeFrontendConfig(cfg);
    log('Configuration loaded from server', 'ok');
  } catch (e) {
    log('Failed to load initial config: ' + e, 'error');
    // Fallback to some minimal defaults if API fails
    initializeFrontendConfig({
      DEFAULT_FPS: 15,
      DEFAULT_COLOR_WIDTH: 1280,
      DEFAULT_COLOR_HEIGHT: 720,
      DEFAULT_DEPTH_MODE: 'NFOV_UNBINNED',
      SKELETON_DEPTH_SAMP_RADIUS: 15,
      SKELETON_DEPTH_BASE_DIR: 'data/depth_bases/',
      SKELETON_ENABLE_DEPTH_VALIDATION: true,
      SKELETON_DEPTH_SUBTRACT_THRESHOLD: 0.01,
      HAND_TO_DETECT: 'right',
      CALIB_MODE: 'manual',
      CALIB_BOARD_TYPE: 'aruco',
      BONE_LENGTHS: {},
      BONE_TOLERANCE: 0.2,
      CALIB_CHESS_BOARD_SIZE: [8, 6],
      CALIB_CHESS_SQUARE_SIZE: 0.025,
      CALIB_ARUCO_BOARD_SIZE: [10, 8],
      CALIB_ARUCO_SQUARE_SIZE: 0.05,
      CALIB_ARUCO_MARKER_SIZE: 0.035,
      CALIB_ARUCO_DICT: 4,
      RECORDING_DURATION: 10.0,
      RECORDING_FPS: 15,
      REALSENSE_RESOLUTIONS: ['640x480', '1280x720', '1920x1080'],
      REALSENSE_FPS: [15, 30],
      KINECT_RESOLUTIONS: ['1280x720', '1920x1080', '2048x1536'],
      KINECT_FPS: [5, 15, 30],
      KINECT_DEPTH_MODES: ['NFOV_UNBINNED', 'NFOV_2X2BINNED', 'WFOV_UNBINNED', 'WFOV_2X2BINNED'],
      KINECT_DEPTH_MODE_MAX_FPS: { 'WFOV_UNBINNED': 15 }
    });
  }

  cameras.scanDevices();
  calibration.populateArucoDicts();
  calibration.toggleBoardFields();
}

init();
