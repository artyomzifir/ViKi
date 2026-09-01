// Shared episode-list rendering for the Record and Extract tabs. One row shape
// (pretty date · pipeline-stage badges · task/meta) with opt-in per-row controls:
//   select  — a checkbox        (Extract: multi-select for the perceive queue)
//   view    — a "view" button   (Extract: open the episode in the scene3d viewer)
//   manage  — inline edit / del (Record: the episode file-manager)
//
// The row carries both data-path (Record's meta PATCH / DELETE key) and data-id
// (Extract's episode id for the queue + viewer). Callers keep their own click
// handlers; this module only builds markup.

export const STAGES = [
  ['raw', 'RAW', 'raw/ — recorded colour + depth frames'],
  ['rec', 'REC', 'rec.npz — extracted 3-D hand landmarks'],
  ['cln', 'CLN', 'cln.npz — fused + smoothed trajectory'],
  ['plan', 'PLN', 'plan.h5 — retargeted robot joint plan'],
  ['replay', 'RPL', 'replay.h5 — physical replay states'],
];

export function esc(s) {
  return String(s ?? '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

// "2026-08-31_13-06-43" → "2026-08-31 · 13:06:43"; anything else passes through.
export function prettyId(id) {
  const m = /^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$/.exec(String(id || ''));
  return m ? `${m[1]} · ${m[2]}:${m[3]}:${m[4]}` : id;
}

function badgesHTML(ep) {
  return STAGES.map(([key, label, tip]) =>
    `<span class="badge ${ep.has?.[key] ? 'ok' : ''}" title="${tip}${ep.has?.[key] ? '' : ' (not done)'}">${label}</span>`
  ).join('');
}

function editingRowHTML(ep) {
  const hand = (ep.hand || 'right').toLowerCase();
  return `<div class="episode-row editing" data-path="${ep.path}" data-id="${ep.id}">
    <span class="ep-when" title="${ep.id}">${prettyId(ep.id)}</span>
    <input class="ep-edit-name" data-role="ep-name" placeholder="name / task" value="${esc(ep.task)}">
    <input class="ep-edit-demo" data-role="ep-demo" placeholder="demonstrator" value="${esc(ep.demonstrator)}">
    <select class="ep-edit-hand" data-role="ep-hand">
      <option value="right"${hand === 'right' ? ' selected' : ''}>right</option>
      <option value="left"${hand === 'left' ? ' selected' : ''}>left</option>
    </select>
    <div class="ep-edit-acts">
      <button data-role="ep-edit-save">save</button>
      <button data-role="ep-edit-cancel">cancel</button>
    </div>
  </div>`;
}

// A post-capture cloud-build bar shown on one row (Record: the take just
// recorded). cp = { id, status, pct, label }.
function cloudBarHTML(cp) {
  const pct = cp.status === 'done' ? 100 : (cp.status === 'running' ? (cp.pct || 0) : 0);
  return `<span class="ep-cloud ${cp.status || ''}">
    <span class="perc-bar"><i style="width:${pct}%"></i></span>
    <span class="ep-cloud-lbl">${esc(cp.label || '')}</span>
  </span>`;
}

// o: { select, selected:Set, view, manage, activeId, editingPath,
//      confirmDeletePath, cloudProgress:{id,status,pct,label} }
export function rowHTML(ep, o = {}) {
  if (o.manage && o.editingPath === ep.path) return editingRowHTML(ep);

  const pick = o.select
    ? `<input type="checkbox" class="ep-sel" data-ep="${ep.id}"${o.selected?.has?.(ep.id) ? ' checked' : ''}> `
    : '';
  const meta = [
    ep.demonstrator,
    ep.hand,
    ep.duration_s != null ? `${ep.duration_s}s` : '',
    ep.fps ? `${ep.fps} fps` : '',
  ].filter(Boolean).join(' · ');

  let acts = '';
  if (o.view) acts += `<button data-view="${ep.id}" title="open in viewer">view</button>`;
  if (o.manage) {
    acts += o.confirmDeletePath === ep.path
      ? `<span class="hint">delete?</span>
         <button data-role="ep-delete-yes" class="danger">yes</button>
         <button data-role="ep-delete-no">no</button>`
      : `<button data-role="ep-edit">edit</button>
         <button data-role="ep-delete" class="danger">del</button>`;
  }

  const active = o.activeId != null && o.activeId === ep.id ? ' active' : '';
  return `<div class="episode-row${active}" data-path="${ep.path}" data-id="${ep.id}">
    <div class="ep-row-top">
      <span class="ep-when" title="${ep.id}">${pick}${prettyId(ep.id)}</span>
      <span class="ep-badges">${badgesHTML(ep)}</span>
    </div>
    <div class="ep-row-bot">
      <span class="ep-name">${ep.task ? esc(ep.task) : '<i>unnamed</i>'}</span>
      ${meta ? `<span class="ep-meta">${esc(meta)}</span>` : ''}
      ${acts ? `<span class="ep-acts">${acts}</span>` : ''}
      ${o.cloudProgress && o.cloudProgress.id === ep.id ? cloudBarHTML(o.cloudProgress) : ''}
    </div>
  </div>`;
}

export function renderList(box, episodes, o = {}) {
  if (!box) return;
  box.innerHTML = episodes?.length
    ? episodes.map(ep => rowHTML(ep, o)).join('')
    : `<div class="hint" style="padding:10px">${o.emptyText || 'no episodes yet'}</div>`;
}
