# ViKi audit, 2026-10-25

See [the review and completion plan](../../../docs/2026-10-25-project-revision.md).

- `recomputed.json`: fresh scores from saved HO-Cap hand/object arrays and inspected plan provenance.
- `recompute.py`: reproduce with `docker compose run --rm -T --entrypoint python3 test - < paper/results/audit_2026_10_25/recompute.py`; source data must exist under the repository's `data/` mount. Prints JSON; does not overwrite inputs.
- `shadow_report.json`, `tesollo_report.json`: fresh, unassisted 190-frame MuJoCo replays. Both fail the current task gate.
- `*_trajectory.npz`, `*_scene.xml`: saved replay outputs; XML asset paths use the container's `/app/` layout and require external local models.
- `provenance.json`: source revision, 98 input hashes, comparison with archived reports. Only GUI-enabled flag differs; numerical results match exactly.
- `tests.txt`: final normal Docker suite output.
- `SHA256SUMS`: hashes for this audit package.

Replay commands (each writes only its named new output directory):

```sh
docker compose -f docker-compose.experiments.yml run --rm sim experiments/hocap_shadow_replay.py --output /app/rl-output/audit_20261025_shadow
docker compose -f docker-compose.experiments.yml run --rm sim experiments/hocap_tesollo_replay.py --output /app/rl-output/audit_20261025_tesollo
```

Use a different output directory to preserve these outputs. Default EGL works in the current image. An OSMesa attempt failed at import because its OpenGL backend was unavailable; it produced no simulation result. The first attempt to save recomputed scores under a container-owned parent failed at output creation; recomputation was repeated with stdout captured on the host. The first extra simulator-test invocation lacked the `scripts` mount; after adding it, all 68 experimental tests passed. These environment failures are not algorithmic failures.

No fresh camera extraction, segmentation, IK solve or physical-robot test was performed. Saved measurements are not an independent held-out success-rate study.
