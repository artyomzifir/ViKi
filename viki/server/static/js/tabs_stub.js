// Placeholder tab for pipeline stages that aren't built yet. main.js maps the
// not-yet-implemented tab names to makeStub('<Label>'); each returns the same
// { mount, unmount } contract the real tab modules use.

export function makeStub(label) {
  return {
    mount(view) {
      view.innerHTML = `
        <div class="tab-stub">
          <h2>${label}</h2>
          <p>This stage isn't wired into the UI yet.</p>
          <p class="hint">Run it from the CLI: <code>viki ${label.toLowerCase()} &lt;episode&gt;</code></p>
        </div>`;
    },
    unmount() { },
  };
}
