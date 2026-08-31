// Entry point: the tab router + the persistent top-bar wiring. Each tab module
// exposes mount(viewEl) / unmount(); switching tabs replaces #view entirely.
import { api, log, mountLog, initializeFrontendConfig } from './core.js';
import * as cameras from './cameras.js';
import * as calibration from './calibration.js';
import * as record from './record.js';
import * as perception from './perception.js';
import * as viewer from './viewer.js';
import * as configModal from './config.js';
import { makeStub } from './tabs_stub.js';

const TABS = {
  calibration: { label: 'Calibration', mod: calibration },
  record: { label: 'Record', mod: record },
  extract: { label: 'Extract', mod: perception },
  prepare: { label: 'Prepare', mod: makeStub('Prepare') },
  retarget: { label: 'Retarget', mod: makeStub('Retarget') },
  replay: { label: 'Replay', mod: makeStub('Replay') },
  export: { label: 'Export', mod: makeStub('Export') },
  viewer: { label: 'Viewer', mod: viewer },
};
const DEFAULT_TAB = 'calibration';

let current = null;

function show(name) {
  if (!TABS[name]) name = DEFAULT_TAB;
  if (current === name) return;
  const view = document.getElementById('view');
  TABS[current]?.mod.unmount?.();
  view.innerHTML = '';
  current = name;
  location.hash = name;
  document.querySelectorAll('#tabbar [data-tab]').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  TABS[name].mod.mount(view);
}

// Re-mount the current tab (e.g. after a config save changed its defaults).
function remount() {
  if (!current) return;
  const view = document.getElementById('view');
  TABS[current].mod.unmount?.();
  view.innerHTML = '';
  TABS[current].mod.mount(view);
}
document.addEventListener('config:saved', remount);

// ── persistent top-bar actions (delegated) ────────────────────────────────

const CLICK_ACTIONS = {
  scanDevices: () => cameras.scanDevices(),
  startAll: () => cameras.startAll(),
  stopAll: () => cameras.stopAll(),
  showTab: el => show(el.dataset.tab),
  toggleLog: () => {
    const p = document.getElementById('log-popover');
    if (p) p.hidden = !p.hidden;
  },
  toggleConfig: () => configModal.toggle(),
  cfgSave: () => configModal.save(),
  cfgReset: () => configModal.reset(),
  cfgRestart: () => configModal.restart(),
  cfgClose: () => configModal.close(),
};

document.addEventListener('click', e => {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const fn = CLICK_ACTIONS[el.dataset.action];
  if (fn) fn(el, e);
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    configModal.close();
    const p = document.getElementById('log-popover');
    if (p) p.hidden = true;
  }
});

// close popover / modal when clicking their backdrop
document.getElementById('config-modal')?.addEventListener('click', e => {
  if (e.target.id === 'config-modal') configModal.close();
});

// ── boot ─────────────────────────────────────────────────────────────────

async function init() {
  mountLog();
  try {
    const cfg = await api('GET', '/api/config');
    await initializeFrontendConfig(cfg);
    log('Configuration loaded from server', 'ok');
  } catch (e) {
    log('Failed to load initial config: ' + e, 'error');
    await initializeFrontendConfig({
      DEFAULT_FPS: 15, DEFAULT_COLOR_WIDTH: 1280, DEFAULT_COLOR_HEIGHT: 720,
      DEFAULT_DEPTH_MODE: 'NFOV_UNBINNED', HAND_TO_DETECT: 'right',
      CALIB_BOARD_TYPE: 'aruco', CALIB_CHESS_BOARD_SIZE: [8, 6],
      CALIB_CHESS_SQUARE_SIZE: 0.025, CALIB_ARUCO_BOARD_SIZE: [8, 10],
      CALIB_ARUCO_SQUARE_SIZE: 0.05, CALIB_ARUCO_MARKER_SIZE: 0.035,
      CALIB_ARUCO_DICT: 4, RECORDING_DURATION: 10, RECORDING_FPS: 15,
      REALSENSE_RESOLUTIONS: ['640x480', '1280x720', '1920x1080'],
      REALSENSE_FPS: [15, 30],
      KINECT_RESOLUTIONS: ['1280x720', '1920x1080', '2048x1536'],
      KINECT_FPS: [5, 15, 30],
      KINECT_DEPTH_MODES: ['NFOV_UNBINNED', 'NFOV_2X2BINNED', 'WFOV_UNBINNED', 'WFOV_2X2BINNED'],
      KINECT_DEPTH_MODE_MAX_FPS: { WFOV_UNBINNED: 15 },
    });
  }

  await cameras.scanDevices();
  cameras.startStatusPoll();
  startSysmonPoll();
  show(location.hash?.slice(1) || DEFAULT_TAB);
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
}

// ── host load monitor (top bar) ──────────────────────────────────────────

const _gib = b => (b / 1073741824).toFixed(1);

// <50% green · 50–90% yellow · >90% red  (same bands for °C)
function tier(v, lo = 50, hi = 90) {
  if (v == null || Number.isNaN(v)) return '';
  return v < lo ? 'ok' : v < hi ? 'warn' : 'bad';
}

function renderSysmon(s) {
  const el = document.getElementById('sysmon');
  if (!el) return;
  if (!s) { el.textContent = ''; return; }
  const seg = [];
  if (s.cpu_percent != null)
    seg.push(`CPU <span class="${tier(s.cpu_percent)}">${s.cpu_percent.toFixed(0)}%</span>`);
  if (s.mem)
    seg.push(`RAM <span class="${tier(s.mem.percent)}">${_gib(s.mem.used)}/${_gib(s.mem.total)}G</span>`);
  if (s.gpu && s.gpu[0]) {
    const g = s.gpu[0];
    const memPct = g.mem_total ? 100 * g.mem_used / g.mem_total : null;
    seg.push(`GPU <span class="${tier(g.util)}">${g.util.toFixed(0)}%</span> `
      + `<span class="${tier(memPct)}">${_gib(g.mem_used)}/${_gib(g.mem_total)}G</span>`
      + (g.temp ? ` <span class="${tier(g.temp)}">${g.temp.toFixed(0)}°</span>` : ''));
  } else {
    seg.push('GPU n/a');
  }
  el.innerHTML = seg.join('&nbsp;&nbsp;·&nbsp;&nbsp;');
}

function setServerDot(ok) {
  const dot = document.getElementById('server-dot');
  if (dot) dot.className = ok ? 'dot green' : 'dot red blink';
}

function startSysmonPoll() {
  const tick = async () => {
    try { await api('GET', '/api/health'); setServerDot(true); }
    catch { setServerDot(false); }
    try { renderSysmon(await api('GET', '/api/system/stats')); }
    catch { /* keep last reading */ }
  };
  tick();
  setInterval(tick, 2500);
}

init();
