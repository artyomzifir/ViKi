# HO-Cap G10_4 insertion-depth trial

Immutable raw `metrics_json` from three retarget `plan.h5` files and complete
MuJoCo replay JSON for 0, 10 and 15 mm insertion depth, plus three
15 mm friction/pressure sensitivity replays. `SHA256SUMS` guards all nine JSON files. The controls, values, interpretation and reproduction
steps are in [`docs/2026-10-15-hocap-insertion-depth-trial.md`](../../../../docs/2026-10-15-hocap-insertion-depth-trial.md).

The raw plans, cropped HO-Cap inputs and robot URDFs are gitignored and remain
on the local machine. Simulated 0.20 kg mass and rubber-pad collision boxes
are proxies, not HO-Cap ground truth or validated hardware behaviour.
