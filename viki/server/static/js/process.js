import { api, log } from './core.js';

let processPage = 0;
let processMode = 'smooth';
let processSelectedRec = null;
let processDsSelectedRec = null;
let processJobs = {};
let processJobsPollInterval = null;
let processJobPolls = {};

let smoothStreamUrl = null;
let vizStreamUrl = null;
let smoothPreviousFilename = null;
let vizPreviousFilename = null;

export function toggleProcess() {
  const panel = document.getElementById('process-panel');
  const visible = panel.style.display === 'block';
  panel.style.display = visible ? 'none' : 'block';
  if (!visible) {
    loadProcessRecordings();
    loadPreparedRecordings();
    pollProcessJobs();
  } else {
    stopSmoothStream();
    stopVizStream();
    clearProcessJobsPoll();
    Object.values(processJobPolls).forEach(clearInterval);
    processJobPolls = {};
  }
}

export function setProcessMode(mode) {
  processMode = mode;

  document.getElementById('btn-process-mode-smooth').className = mode === 'smooth' ? 'primary' : '';
  document.getElementById('btn-process-mode-dataset').className = mode === 'dataset' ? 'primary' : '';

  stopSmoothStream();
  stopVizStream();

  if (mode === 'smooth') {
    document.getElementById('process-smooth-controls').style.display = '';
    document.getElementById('process-dataset-controls').style.display = 'none';
    document.getElementById('process-smooth-viz-controls').style.display = '';
    document.getElementById('process-dataset-viz-controls').style.display = 'none';

    document.getElementById('process-stream').innerHTML =
      '<div style="color:var(--muted);padding:20px;text-align:center;">Select a recording to view</div>';
    document.getElementById('process-status').textContent = '';
    loadProcessRecordings();
    loadPreparedRecordings();
  } else {
    document.getElementById('process-smooth-controls').style.display = 'none';
    document.getElementById('process-dataset-controls').style.display = 'flex';
    document.getElementById('process-smooth-viz-controls').style.display = 'none';
    document.getElementById('process-dataset-viz-controls').style.display = 'flex';

    document.getElementById('process-stream').innerHTML =
      '<div style="color:var(--muted);padding:20px;text-align:center;">Select an output to view</div>';
    document.getElementById('process-ds-status').textContent = '';
    loadPreparedDSCLNList();
    loadVizOutputs();
  }
}

export async function loadProcessRecordings() {
  const statusEl = document.getElementById('process-status');
  const listEl = document.getElementById('process-rec-list');
  statusEl.textContent = 'Loading...';
  try {
    const data = await api('GET', `/api/optimization/recordings?page=${processPage}&limit=10`);
    const recs = data.recordings || [];
    listEl.innerHTML = recs.length === 0
      ? '<div style="padding:8px;color:var(--muted);">No recordings found</div>'
      : recs.map(f =>
        `<div style="padding:6px 8px;cursor:pointer;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;"
                    data-action="selectProcessRec" data-filename="${f}">
                <input type="radio" name="process-rec" value="${f}" style="accent-color:var(--accent);">
                <span>${f}</span>
              </div>`
      ).join('');
    document.getElementById('btn-process-prev').disabled = processPage === 0;
    document.getElementById('btn-process-next').disabled = recs.length < 10;
    document.getElementById('process-page-info').textContent = `Page ${processPage + 1}`;
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = `Failed to load: ${e}`;
  }
}

export function selectProcessRec(filename, el) {
  processSelectedRec = filename;
  document.querySelectorAll('#process-rec-list div').forEach(d => d.style.background = '');
  if (el) el.style.background = 'var(--surface)';
  document.getElementById('btn-process-smooth').disabled = false;
}

export function processPrevPage() {
  if (processPage > 0) { processPage--; loadProcessRecordings(); }
}

export function processNextPage() {
  processPage++;
  loadProcessRecordings();
}

function pollProcessJobs() {
  if (processJobsPollInterval) clearInterval(processJobsPollInterval);
  processJobsPollInterval = setInterval(renderProcessJobs, 2000);
  renderProcessJobs();
}

function clearProcessJobsPoll() {
  if (processJobsPollInterval) {
    clearInterval(processJobsPollInterval);
    processJobsPollInterval = null;
  }
}

async function renderProcessJobs() {
  try {
    const data = await api('GET', '/api/dataset/optimize/jobs');
    const container = document.getElementById('process-conversion-status');
    data.jobs.forEach(j => { processJobs[j.job_id] = j; });
    const entries = Object.values(processJobs).slice(0, 20);
    if (entries.length === 0) {
      container.style.display = 'none';
      return;
    }
    container.style.display = 'block';
    container.innerHTML = entries.map(j => {
      const icons = { queued: '⏳', running: '⟳', completed: '✅', failed: '❌' };
      const colors = { queued: 'var(--muted)', running: 'var(--yellow)', completed: 'var(--green)', failed: 'var(--red)' };
      return `<div style="padding:2px 0;color:${colors[j.status] || 'var(--muted)'}">${icons[j.status] || '?'} ${j.filename} → ${j.robot} [${j.status}]</div>`;
    }).join('');
  } catch (e) { /* ignore */ }
}

export async function processSmoothSelected() {
  if (!processSelectedRec) return;
  const statusEl = document.getElementById('process-status');
  const btn = document.getElementById('btn-process-smooth');
  btn.disabled = true;

  const winLen = parseInt(document.getElementById('process-win-len').value) || 7;
  const poly = parseInt(document.getElementById('process-poly').value) || 2;
  statusEl.textContent = `Smoothing ${processSelectedRec}...`;
  try {
    const res = await api('POST', '/api/optimization/smooth', { filename: processSelectedRec, window_length: winLen, polyorder: poly });
    statusEl.textContent = `✅ ${res.path}`;
    log(`Smoothed: ${res.path}`, 'ok');
    const plotName = res.path.split('/').pop() || res.path.split('\\').pop();
    loadPreparedRecordings();
    startSmoothStream(plotName);
  } catch (e) {
    statusEl.textContent = `❌ ${e}`;
    log(`Smoothing failed: ${e}`, 'error');
  }
  btn.disabled = false;
}

export async function loadPreparedRecordings() {
  const listEl = document.getElementById('process-cln-list');
  const infoEl = document.getElementById('process-cln-info');
  infoEl.textContent = 'Loading...';
  try {
    const data = await api('GET', '/api/optimization/smoothed-recordings?page=0&limit=50');
    const files = data.recordings || [];
    listEl.innerHTML = files.length === 0
      ? '<div style="padding:8px;color:var(--muted);">No prepared recordings</div>'
      : files.map(f =>
        `<div style="padding:6px 8px;cursor:pointer;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;"
                    data-action="selectPreparedRec" data-filename="${f}">
                <span>${f}</span>
              </div>`
      ).join('');
    infoEl.textContent = `${files.length} prepared`;
  } catch (e) {
    infoEl.textContent = `Failed: ${e}`;
    listEl.innerHTML = '<div style="padding:8px;color:var(--red);">Failed to load</div>';
  }
}

export function selectPreparedRec(filename, el) {
  document.querySelectorAll('#process-cln-list div').forEach(d => d.style.background = '');
  if (el) el.style.background = 'var(--surface)';
  smoothPreviousFilename = filename;
  startSmoothStream(filename);
}

export async function loadPreparedDSCLNList() {
  const listEl = document.getElementById('process-ds-cln-list');
  const infoEl = document.getElementById('process-ds-page-info');
  infoEl.textContent = 'Loading...';
  try {
    const data = await api('GET', '/api/optimization/smoothed-recordings?page=0&limit=50');
    const files = data.recordings || [];
    listEl.innerHTML = files.length === 0
      ? '<div style="padding:8px;color:var(--muted);">No prepared recordings</div>'
      : files.map(f =>
        `<div style="padding:6px 8px;cursor:pointer;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;"
                    data-action="selectDsCLNRec" data-filename="${f}">
                <input type="radio" name="process-ds-cln" value="${f}" style="accent-color:var(--accent);">
                <span>${f}</span>
              </div>`
      ).join('');
    infoEl.textContent = `${files.length} file(s)`;
  } catch (e) {
    infoEl.textContent = `Failed: ${e}`;
    listEl.innerHTML = '<div style="padding:8px;color:var(--red);">Failed to load</div>';
  }
}

export function selectDsCLNRec(filename, el) {
  processDsSelectedRec = filename;
  document.querySelectorAll('#process-ds-cln-list div').forEach(d => d.style.background = '');
  if (el) el.style.background = 'var(--surface)';
  document.getElementById('btn-process-ds-convert').disabled = false;
}

export async function loadVizOutputs() {
  const listEl = document.getElementById('process-viz-output-list');
  const infoEl = document.getElementById('process-viz-info');
  infoEl.textContent = 'Loading...';
  try {
    const data = await api('GET', '/api/dataset/outputs');
    const files = data.outputs || [];
    listEl.innerHTML = files.length === 0
      ? '<div style="padding:8px;color:var(--muted);">No outputs found</div>'
      : files.map(f =>
        `<div style="padding:6px 8px;cursor:pointer;border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;"
                    data-action="selectVizOutput" data-filename="${f}">
                <span>${f}</span>
              </div>`
      ).join('');
    infoEl.textContent = `${files.length} output(s)`;
  } catch (e) {
    infoEl.textContent = `Failed: ${e}`;
    listEl.innerHTML = '<div style="padding:8px;color:var(--red);">Failed to load outputs</div>';
  }
}

export function selectVizOutput(filename, el) {
  document.querySelectorAll('#process-viz-output-list div').forEach(d => d.style.background = '');
  if (el) el.style.background = 'var(--surface)';
  vizPreviousFilename = filename;
  startVizStream(filename);
}

export async function convertDataset() {
  if (!processDsSelectedRec) return;
  const statusEl = document.getElementById('process-ds-status');
  const btn = document.getElementById('btn-process-ds-convert');
  btn.disabled = true;

  const robot = document.getElementById('process-ds-robot').value;
  const baseX = document.getElementById('process-ds-base-x').value || '0';
  const baseY = document.getElementById('process-ds-base-y').value || '0';
  const baseZ = document.getElementById('process-ds-base-z').value || '0';
  const targX = document.getElementById('process-ds-target-x').value || '0';
  const targY = document.getElementById('process-ds-target-y').value || '0';
  const targZ = document.getElementById('process-ds-target-z').value || '0';
  const scale = parseFloat(document.getElementById('process-ds-scale').value) || 1.0;

  const baseOffset = `${baseX},${baseY},${baseZ}`;
  const targetOffset = `${targX},${targY},${targZ}`;

  statusEl.textContent = `Queueing ${processDsSelectedRec} for robot ${robot}...`;
  try {
    const res = await api('POST', '/api/dataset/optimize', {
      filename: processDsSelectedRec,
      robot,
      base_offset: baseOffset,
      target_offset: targetOffset,
      trajectory_scale: scale,
    });
    log(`Dataset conversion queued: ${processDsSelectedRec} -> ${robot} (job: ${res.job_id})`, 'ok');
    statusEl.innerHTML = `⏳ Job queued — <span id="ds-job-status-${res.job_id}" style="color:var(--yellow);">pending</span>`;
    processJobs[res.job_id] = { job_id: res.job_id, filename: processDsSelectedRec, robot, status: 'queued' };
    renderProcessJobs();

    const poll = setInterval(async () => {
      try {
        const j = await api('GET', `/api/dataset/optimize/status/${res.job_id}`);
        processJobs[res.job_id] = j;
        renderProcessJobs();
        const span = document.getElementById(`ds-job-status-${res.job_id}`);
        if (span) {
          if (j.status === 'running') { span.innerHTML = '<span style="color:var(--yellow);">⟳ converting...</span>'; }
          else if (j.status === 'completed') {
            span.innerHTML = '<span style="color:var(--green);">✅ done</span>';
            clearInterval(poll);
            delete processJobPolls[res.job_id];
            loadVizOutputs();
          } else if (j.status === 'failed') {
            span.innerHTML = `<span style="color:var(--red);">❌ ${j.error || 'failed'}</span>`;
            clearInterval(poll);
            delete processJobPolls[res.job_id];
          }
        }
      } catch (e) { clearInterval(poll); }
    }, 1000);
    processJobPolls[res.job_id] = poll;
  } catch (e) {
    statusEl.textContent = `❌ ${e}`;
    log(`Dataset conversion failed: ${e}`, 'error');
  }
  btn.disabled = false;
}

function buildSmoothConfig() {
  const getChecked = id => document.getElementById(id)?.checked ?? true;
  const centerOn = document.querySelector('input[name="smooth-center"]:checked')?.value ?? 'world';
  const axesLength = parseFloat(document.getElementById('smooth-axes-length')?.value) || 1.0;
  return {
    show_raw: getChecked('smooth-toggle-raw'),
    show_smooth: getChecked('smooth-toggle-smooth'),
    axes_length: axesLength,
    center_on: centerOn,
  };
}

function startSmoothStream(filename) {
  if (!filename) return;
  const cfg = buildSmoothConfig();
  const params = new URLSearchParams({ filename, ...cfg });
  const url = `/api/optimization/smooth-stream?${params}&t=${Date.now()}`;

  const streamDiv = document.getElementById('process-stream');
  if (smoothStreamUrl) {
    const oldImg = streamDiv.querySelector('img');
    if (oldImg) oldImg.src = '';
  }
  smoothStreamUrl = url;
  streamDiv.innerHTML = `<img src="${url}" alt="smoothing comparison">`;
}

export function applySmoothConfig() {
  if (smoothPreviousFilename) startSmoothStream(smoothPreviousFilename);
}

function stopSmoothStream() {
  const img = document.querySelector('#process-stream img');
  if (img && smoothStreamUrl) img.src = '';
  smoothStreamUrl = null;
  smoothPreviousFilename = null;
}

function buildVizConfig() {
  const getChecked = id => document.getElementById(id)?.checked ?? true;
  const centerOn = document.querySelector('input[name="viz-center"]:checked')?.value ?? 'world';
  const axesLength = parseFloat(document.getElementById('viz-axes-length')?.value) || 2.0;
  return {
    center_on: centerOn,
    axes_length: axesLength,
    show_cameras: getChecked('viz-toggle-cameras'),
    show_board: getChecked('viz-toggle-board'),
    show_neutral_ee: getChecked('viz-toggle-neutral'),
    show_human_trail: getChecked('viz-toggle-human-trail'),
    show_robot_trail: getChecked('viz-toggle-robot-trail'),
    show_base_to_ee: getChecked('viz-toggle-base-to-ee'),
    show_debug_overlay: getChecked('viz-toggle-debug'),
    show_reach_sphere: getChecked('viz-toggle-reach'),
    show_fk_arm: getChecked('viz-toggle-fk'),
    show_ee_target: getChecked('viz-toggle-ee-target'),
  };
}

function startVizStream(filename) {
  if (!filename) return;
  const cfg = buildVizConfig();
  const params = new URLSearchParams({ filename, ...cfg });
  const url = `/api/dataset/viz-stream?${params}&t=${Date.now()}`;

  const streamDiv = document.getElementById('process-stream');
  if (vizStreamUrl) {
    const oldImg = streamDiv.querySelector('img');
    if (oldImg) oldImg.src = '';
  }
  vizStreamUrl = url;
  streamDiv.innerHTML = `<img src="${url}" alt="robot trajectory">`;
}

export function applyVizConfig() {
  if (vizPreviousFilename) startVizStream(vizPreviousFilename);
}

function stopVizStream() {
  const img = document.querySelector('#process-stream img');
  if (img && vizStreamUrl) img.src = '';
  vizStreamUrl = null;
  vizPreviousFilename = null;
}
