// Global background-job widget in the header, between the host-load monitor and
// the log line. Same idea as the log widget: the job running *now* shows inline
// in #job-line, the whole list drops down into #job-popover. The Extract and
// Retarget tabs no longer render their own queue — they call refreshJobs() after
// submitting and listen for the `jobs:updated` event for tab-specific reactions.
import { api, log } from './core.js';

let JOBS = [];
let timer = 0;
let inflight = false;

const KIND_LABEL = {
  perceive: 'perceive', extract: 'extract', prepare: 'prepare', cloud: 'cloud',
  download: 'download', retarget: 'retarget', export: 'export', record: 'record',
};

// Running first, then the next queued one — that's "what's in work now".
function activeJob() {
  return JOBS.find(j => j.status === 'running')
    || JOBS.find(j => j.status === 'queued')
    || null;
}

function jobName(j) {
  return j.episode || KIND_LABEL[j.kind] || j.kind || 'job';
}

function jobText(j) {
  const p = j.progress || {};
  if (j.status === 'queued') return `${jobName(j)} · queued #${j.queue_pos ?? '?'}`;
  if (j.status === 'running') {
    const where = p.total ? `${p.frame || 0}/${p.total}`
      : (p.frame != null ? `frame ${p.frame}` : '');
    return `${jobName(j)} · ${(p.stage || j.kind || 'run')} ${where}`.trim();
  }
  return `${jobName(j)} · ${j.status}`;
}

function pct(j) {
  const p = j.progress || {};
  if (j.status === 'done') return 100;
  if (j.status === 'running' && p.total) return Math.round(100 * (p.frame || 0) / p.total);
  return 0;
}

function render() {
  const line = document.getElementById('job-line');
  const toggle = document.getElementById('job-toggle');
  const running = JOBS.filter(j => j.status === 'running').length;
  const queued = JOBS.filter(j => j.status === 'queued').length;
  const active = activeJob();

  if (line) {
    if (active) {
      const extra = queued && active.status === 'running' ? `  +${queued} queued` : '';
      line.innerHTML =
        `<span class="job-cls ${active.status}">${jobText(active)}${extra}</span>`
        + `<span class="job-mini"><i style="width:${pct(active)}%"></i></span>`;
    } else {
      const err = JOBS.find(j => j.status === 'error');
      line.textContent = err ? `${jobName(err)} · failed` : (JOBS.length ? 'jobs idle' : '');
    }
  }
  if (toggle) toggle.classList.toggle('busy', running > 0 || queued > 0);

  const list = document.querySelector('#job-popover .job-list');
  if (list) {
    list.innerHTML = JOBS.length
      ? JOBS.slice(0, 40).map(j => `
        <div class="job-row ${j.status}">
          <span class="job-row-ep">${jobName(j)}</span>
          <span class="job-row-st">${jobText(j)}</span>
          <span class="perc-bar"><i style="width:${pct(j)}%"></i></span>
          ${j.status === 'queued' ? `<button data-job-cancel="${j.id}">✕</button>` : '<span></span>'}
        </div>`).join('')
      : '<div class="hint" style="padding:6px 4px">no jobs</div>';
  }
}

export async function refreshJobs() {
  if (inflight) return;
  inflight = true;
  try {
    const { jobs } = await api('GET', '/api/pipeline/jobs');
    JOBS = Array.isArray(jobs) ? jobs : [];
    render();
    document.dispatchEvent(new CustomEvent('jobs:updated', { detail: JOBS }));
  } catch { /* keep the last good render */ }
  finally { inflight = false; }
}

export function mountJobs() {
  render();
  refreshJobs();
  clearInterval(timer);
  timer = setInterval(refreshJobs, 1500);
  // Cancel a queued job straight from the dropdown (delegated, bound once).
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-job-cancel]');
    if (!b) return;
    api('DELETE', `/api/pipeline/jobs/${b.dataset.jobCancel}`)
      .then(refreshJobs)
      .catch(err => log('cancel: ' + err, 'error'));
  });
}
