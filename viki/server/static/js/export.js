// Export tab: retargeted episodes -> a dataset on disk.
//
// Two formats behind one button. The trajectory bundle is the default because
// it is the one that completes here: the LeRobot writer needs optional
// dependencies (lerobot, torch) that are not in the image, and it also demands
// a replay verdict that no episode has while replay is a stub. The tab says so
// rather than letting the user discover it from a failed job.
import { api, log, sessionGet, sessionPatch } from './core.js';
import { renderList } from './episodes.js';

const SESSION_KEY = 'export-v1';
let root = null;
let epList = [];
let selected = new Set();
let watching = null;          // job id we are waiting on

export function mount(view) {
  const S = { format: 'trajectory', out: 'data/datasets/exported', name: '', ...sessionGet(SESSION_KEY, {}) };
  selected = new Set();
  watching = null;
  root = document.createElement('div');
  root.className = 'export-tab';
  root.innerHTML = `
    <aside class="exp-runcol">
      <div class="calib-sec-title">1 · Episodes</div>
      <div class="cfg-row"><label>Dataset</label><select data-role="dataset"></select></div>
      <div class="exp-eps" data-role="eps"></div>
      <label class="perc-all"><input type="checkbox" data-role="all"> select all exportable</label>
      <div class="hint" data-role="eligible">—</div>
    </aside>

    <section class="exp-main">
      <section class="calib-sec">
        <div class="calib-sec-title">2 · Format</div>
        <div class="cfg-row"><label>Format</label>
          <select data-role="format">
            <option value="trajectory"${S.format === 'trajectory' ? ' selected' : ''}>trajectory bundle · npz + manifest</option>
            <option value="lerobot"${S.format === 'lerobot' ? ' selected' : ''}>LeRobot dataset · needs viki[export]</option>
          </select></div>
        <div class="hint" data-role="format-meta"></div>

        <div class="calib-sec-title">3 · Output</div>
        <div class="cfg-row"><label>Directory</label>
          <input data-role="out" value="${S.out}"></div>
        <div class="cfg-row"><label>Name</label>
          <input data-role="name" value="${S.name}" placeholder="defaults to the directory name"></div>

        <button class="primary" data-role="run">Export</button>
        <div class="hint">An episode is exportable once retarget has written its plan.h5.</div>
      </section>

      <section class="calib-sec" data-role="result-sec" hidden>
        <div class="calib-sec-title">Result</div>
        <div data-role="result"></div>
      </section>
    </section>`;
  view.appendChild(root);

  root.querySelector('[data-role="dataset"]').addEventListener('change', loadEpisodes);
  root.querySelector('[data-role="format"]').addEventListener('change', () => { persist(); syncFormat(); });
  root.querySelector('[data-role="out"]').addEventListener('change', persist);
  root.querySelector('[data-role="name"]').addEventListener('change', persist);
  root.querySelector('[data-role="all"]').addEventListener('change', onSelectAll);
  root.querySelector('[data-role="eps"]').addEventListener('change', onPick);
  root.querySelector('[data-role="run"]').addEventListener('click', run);
  document.addEventListener('jobs:updated', onJobsUpdated);

  syncFormat();
  loadDatasets();
}

export function unmount() {
  document.removeEventListener('jobs:updated', onJobsUpdated);
  root = null;
}

function persist() {
  sessionPatch(SESSION_KEY, {
    format: root.querySelector('[data-role="format"]').value,
    out: root.querySelector('[data-role="out"]').value.trim(),
    name: root.querySelector('[data-role="name"]').value.trim(),
  });
}

function syncFormat() {
  const fmt = root.querySelector('[data-role="format"]').value;
  root.querySelector('[data-role="format-meta"]').innerHTML = fmt === 'lerobot'
    ? 'video + frames for policy training. Needs the <code>lerobot</code> package '
      + 'and episodes that have been replayed and screened — neither is available yet, '
      + 'so this will report a clean error.'
    : 'joint trajectory, tool pose, gripper command and the hand pose it came from, '
      + 'as <code>.npz</code> plus a manifest. No optional dependencies. '
      + '<b>Nothing is screened</b> — the manifest records what actually ran per episode.';
}

function exportable(ep) { return !!ep.has?.plan; }

async function loadDatasets() {
  const sel = root.querySelector('[data-role="dataset"]');
  try {
    const { datasets } = await api('GET', '/api/datasets');
    sel.innerHTML = (datasets || []).map(d =>
      `<option value="${d.name}">${d.name} (${d.episodes})</option>`).join('');
  } catch (err) {
    log(`export: cannot list datasets: ${err}`, 'err');
  }
  await loadEpisodes();
}

async function loadEpisodes() {
  const dataset = root.querySelector('[data-role="dataset"]').value;
  if (!dataset) { epList = []; render(); return; }
  try {
    ({ episodes: epList } = await api('GET', `/api/datasets/${encodeURIComponent(dataset)}/episodes`));
  } catch (err) {
    epList = [];
    log(`export: cannot list episodes: ${err}`, 'err');
  }
  selected = new Set([...selected].filter(id => epList.some(e => e.id === id)));
  render();
}

function render() {
  renderList(root.querySelector('[data-role="eps"]'), epList, {
    select: true, selected, emptyText: 'no episodes in this dataset',
  });
  const ok = epList.filter(exportable).length;
  const pickedBad = [...selected].filter(id => !exportable(epList.find(e => e.id === id) || {}));
  root.querySelector('[data-role="eligible"]').innerHTML =
    `${ok} of ${epList.length} have a plan.h5 · ${selected.size} selected`
    + (pickedBad.length ? ` · <b>${pickedBad.length} selected without a plan will be skipped</b>` : '');
}

function onPick(e) {
  const box = e.target.closest('.ep-sel');
  if (!box) return;
  if (box.checked) selected.add(box.dataset.ep); else selected.delete(box.dataset.ep);
  render();
}

function onSelectAll(e) {
  selected = e.target.checked ? new Set(epList.filter(exportable).map(ep => ep.id)) : new Set();
  render();
}

function paths() {
  return epList.filter(ep => selected.has(ep.id)).map(ep => ep.path);
}

async function run() {
  const chosen = paths();
  if (!chosen.length) { log('export: select at least one episode', 'warn'); return; }
  const out = root.querySelector('[data-role="out"]').value.trim();
  if (!out) { log('export: set an output directory', 'warn'); return; }
  persist();
  try {
    const { job_id } = await api('POST', '/api/export', {
      episodes: chosen,
      out_dir: out,
      format: root.querySelector('[data-role="format"]').value,
      name: root.querySelector('[data-role="name"]').value.trim() || null,
    });
    watching = job_id;
    showResult('<span class="hint">queued…</span>');
    log(`export: queued ${chosen.length} episode(s) -> ${out}`);
  } catch (err) {
    log(`export failed to queue: ${err}`, 'err');
  }
}

function onJobsUpdated(e) {
  if (!root || !watching) return;
  const job = (e.detail || []).find(j => j.id === watching);
  if (!job) return;
  if (job.status === 'running') { showResult('<span class="hint">writing…</span>'); return; }
  if (job.status === 'error') {
    watching = null;
    showResult(`<div class="err">export failed: ${job.error || 'unknown error'}</div>`);
    return;
  }
  if (job.status !== 'done') return;
  watching = null;
  const r = job.result || {};
  if (!r.episode_count) { showResult(`<div>written to <code>${r.out_dir || '?'}</code></div>`); return; }
  const skipped = (r.skipped || []).map(s =>
    `<li><code>${s.episode_id}</code> — ${s.why}</li>`).join('');
  showResult(`
    <div><b>${r.episode_count}</b> episode(s), <b>${r.total_frames}</b> frames
         ${r.robots?.length ? `· ${r.robots.join(', ')}` : ''}</div>
    <div class="hint">written to <code>${r.out_dir}</code></div>
    ${skipped ? `<div class="hint">skipped:</div><ul class="hint">${skipped}</ul>` : ''}
    <div class="hint" style="margin-top:6px">${r.screening || ''}</div>`);
  log(`export: ${r.episode_count} episode(s), ${r.total_frames} frames -> ${r.out_dir}`);
}

function showResult(html) {
  root.querySelector('[data-role="result-sec"]').hidden = false;
  root.querySelector('[data-role="result"]').innerHTML = html;
}
