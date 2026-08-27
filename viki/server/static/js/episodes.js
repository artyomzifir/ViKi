// Episodes panel: list episodes, run offline stages as jobs, edit labels,
// record a new scene. Selecting an episode loads it into the 3-D viewer.
import { api, log } from './core.js';
import { loadEpisode } from './viewer.js';

let selected = null;   // { id, path, ... }
const STAGES = ['extract', 'prepare', 'retarget', 'replay'];

export function togglePanel() {
  for (const id of ['episodes-panel', 'viewer-panel']) {
    const p = document.getElementById(id);
    p.style.display = p.style.display === 'block' ? 'none' : 'block';
  }
  if (document.getElementById('episodes-panel').style.display === 'block') refresh();
}

export async function refresh() {
  let data;
  try { data = await api('GET', '/api/pipeline/episodes'); }
  catch (e) { log('episodes: ' + e, 'error'); return; }
  const box = document.getElementById('episode-list');
  box.innerHTML = '';
  for (const ep of data.episodes) {
    const row = document.createElement('div');
    row.className = 'episode-row' + (selected?.id === ep.id ? ' sel' : '');
    const badges = STAGES.map(s => {
      const done = ep.stages?.[s]?.done;
      return `<span class="badge ${done ? 'ok' : ''}">${s[0].toUpperCase()}</span>`;
    }).join('');
    row.innerHTML = `<span class="ep-id">${ep.id}</span>
      <span class="ep-task">${ep.task || '<i>unlabelled</i>'}</span>
      <span class="ep-badges">${badges}</span>`;
    row.onclick = () => select(ep);
    box.appendChild(row);
  }
}

function select(ep) {
  selected = ep;
  refresh();
  document.getElementById('episode-detail').style.display = 'flex';
  document.getElementById('ed-id').textContent = ep.id;
  loadLabels(ep);
  loadEpisode(ep.id);
}

async function loadLabels(ep) {
  try {
    const l = await api('GET', `/api/label?episode=${encodeURIComponent(ep.path)}`);
    document.getElementById('lbl-task').value = l.task || '';
    document.getElementById('lbl-hand').value = l.hand || 'right';
    document.getElementById('lbl-outcome').value = l.outcome || 'unrated';
  } catch (e) { /* no labels yet */ }
}

export async function saveLabels() {
  if (!selected) return;
  const body = {
    task: document.getElementById('lbl-task').value,
    hand: document.getElementById('lbl-hand').value,
    outcome: document.getElementById('lbl-outcome').value,
  };
  try {
    await api('POST', `/api/label?episode=${encodeURIComponent(selected.path)}`, body);
    log('labels saved', 'ok');
    refresh();
  } catch (e) { log('save labels: ' + e, 'error'); }
}

const STAGE_PATH = {
  extract: '/api/pipeline/extract', prepare: '/api/pipeline/prepare',
  retarget: '/api/pipeline/retarget', replay: '/api/replay',
};
const JOB_BASE = { replay: '/api/replay/jobs', record: '/api/record/jobs' };

export async function runStage(el) {
  if (!selected) return;
  await runOne(el.dataset.stage);
}

async function runOne(stage) {
  const body = stage === 'replay'
    ? { episode: selected.path, driver: 'dryrun' }
    : { episode: selected.path };
  let job_id;
  try {
    ({ job_id } = await api('POST', STAGE_PATH[stage], body));
    log(`${stage} started (${job_id})`);
  } catch (e) { log(`${stage}: ` + e, 'error'); throw e; }
  const ok = await waitJob(stage, job_id);
  refresh();
  if (selected) loadEpisode(selected.id);
  if (!ok) throw new Error(`${stage} failed`);
}

function waitJob(stage, jobId) {
  const base = JOB_BASE[stage] || '/api/pipeline/jobs';
  return new Promise(resolve => {
    const t = setInterval(async () => {
      let j;
      try { j = await api('GET', `${base}/${jobId}`); } catch { return; }
      if (j.status === 'running') return;
      clearInterval(t);
      if (j.status === 'done') { log(`${stage} done`, 'ok'); resolve(true); }
      else { log(`${stage} failed: ${j.error}`, 'error'); resolve(false); }
    }, 1500);
  });
}

function pollJob(stage, jobId) { waitJob(stage, jobId).then(() => refresh()); }

export async function runAll() {
  if (!selected) return;
  for (const s of STAGES) {
    try { await runOne(s); } catch { break; }
  }
}

export async function recordScene() {
  const seconds = +document.getElementById('rec-seconds').value || 10;
  const task = document.getElementById('rec-task').value || '';
  try {
    const { job_id } = await api('POST', '/api/record/start', { seconds, task, fps: 15 });
    log(`recording ${seconds}s (${job_id})`);
    pollJob('record', job_id);
  } catch (e) { log('record: ' + e, 'error'); }
}
