# HO-Cap retarget numerical snapshot

This directory preserves the completed 2026-10-14 hand-only UR10/Robotiq
2F-85 batch separately from the pre-retarget perception evidence. The
`retarget_batch_status.json` is a byte-for-byte copy of
`data/datasets/hocap_eightview/retarget_batch_status.json` at completion;
`2026-10-14-hocap-retarget-results.md` is generated from that copy and embeds
the complete unrounded JSON. `SHA256SUMS` guards both files.

All 28 entries are `complete`, meaning numerical plans were produced. It does
not mean physical execution, successful grasp, or collision avoidance for
people and objects. The table constraint is an unmeasured infinite plane at
`z=0` in `hocap_tag_1`. The 28 `plan.h5` files remain beside their source
episodes under ignored `data/`; they are **not** included in this snapshot.

To verify: `cd paper/results/hocap/retarget && sha256sum -c SHA256SUMS`.
To regenerate the diary from the archived status, run
`python3 scripts/build_hocap_retarget_report.py
paper/results/hocap/retarget/retarget_batch_status.json
paper/results/hocap/retarget/2026-10-14-hocap-retarget-results.md`
from the repository root, then re-check the hashes.
