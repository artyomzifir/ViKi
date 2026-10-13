# Contact-aware and placement-aware augmentation — 2026-10-12

## Objective and measurement

The existing released-stable-stack check remains a **hard gate**. A successful
MuJoCo replay receives a dimensionless quality penalty:

`2 × (red + green robot-contact normal impulse in N·s)
 + blue/green XY center error / 5 mm
 + blue upright error / 5°
 + blue/green relative rotation error / 10°`.

Failed replays are not ranked as successful. The relative rotation error is
computed modulo the cube's 24 rotational symmetries; a 90° turn of a cube is
not inherently wrong. MuJoCo now also logs contact duration, peak normal
force, penetration, contact geometry, final center/height error and tilt.
Contact force is **simulated**, not a calibrated real-world measurement.
These denominators are experimental priorities, not learned weights or a
physics law. We do not quietly redefine the earlier success rates.

The red support is contacted mainly by the right distal phalanx during
pinch closure; the green support has no robot contact in the tested scenes.
The pilot's red touch lasts about 0.75 s with 1.24 N·s normal impulse.
Its blue cube finishes 7.98 mm off the green center and about 17.6° rotated
relative to the green cube, although its top face is almost horizontal.
This shows why binary stack success alone was too permissive.

## Quality search and dataset

We generated **scratch** variants without changing the original pickup
target: small XY corrections and yaw during carry/placement, plus a separate
test of closure timing. Every candidate was rerun with the same held 50 Hz
controller; only physics-successful variants were ranked. The single-scene
pilot improved the penalty from **5.85 to 4.70**, center error from
**7.98 to 4.56 mm**, and relative rotation error from **17.6° to 12.9°**.
Changing closure timing did not eliminate red contact. Larger yaw changes
sometimes aligned the cube but lost the stack, so they were rejected.

On **18** physically accepted layouts from independent seed 2028, four
variants per layout (original plus three placement variants) were checked.
The selected plan remained physically successful on **18/18**; each had a
lower penalty than its own baseline. Mean center error fell from **9.39 to
4.09 mm**; mean relative rotation error from **17.56° to 14.38°**. Ten
selected plans use +5 mm/+5 mm placement and +5° carry yaw; eight use only
+5 mm/+5 mm placement. This is a *selection result on the same 18 scenes*,
not an independent generalization estimate or a newly trained policy.

**The red-contact problem is not solved.** All 18 selected plans still have
red robot contact, and their mean red-contact impulse is unchanged at
**1.222 N·s**. Green robot contact remains zero. The current perturbations
do not move the pickup target, and a penalty alone cannot remove a contact
shared by every candidate. Eliminating it requires a new pickup-clearance or
grasp-geometry experiment; the earlier instruction not to shift the grasp
was respected here.

## Artifacts and reproduction

The pilot is in `data/experiments/bc-v1/quality-search-001/`; the 18-layout
quality-selected plans and decision manifest are in
`data/experiments/bc-v1/quality-batch-001/`. Original recordings,
calibrations, canonical plans and the trained `layout_bc_v1.pt` checkpoint
are unchanged. The existing BC model has **not** been retrained on these
variants; its published 14/14 and 30/30 results refer to the old objective.

```bash
docker compose -f docker-compose.experiments.yml run --rm sim experiments/quality_search.py
docker compose -f docker-compose.experiments.yml run --rm sim experiments/quality_batch.py
experiments/run_viewer.sh --quality
```

The same quality penalty can later be used as a terminal/trajectory reward
in an RL environment. Plain behavioural cloning minimizes command error and
does not optimize this penalty without better demonstrations or a separate
optimization step.
