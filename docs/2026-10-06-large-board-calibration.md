# Large-board calibration: field observation (2026-10-06)

## Observation and scope

The operator reports that the doubled/shifted two-Kinect cloud was pronounced
with the small ChArUco board. After recalibrating with the large board, the
clouds look substantially better aligned and are visually usable for the
planned cube-transfer recordings. The earlier hand mismatch was about 1–2 cm
by eye; no independent ground-truth accuracy measurement was made for the new
cloud. This is a **field observation**, not a measured millimetre accuracy or
proof that board size was the only changed cause.

| Preset | Printed ChArUco board | Saved capture sets | Automatic validation |
|---|---|---:|---|
| `table-v1` | 11×7 squares, 25 mm square, 18.75 mm marker, `DICT_5X5_100` | 16 | green |
| `big-board` | 8×10 squares, 50 mm square, 35 mm marker, `DICT_5X5_50` | 12 | green |

The large print is 400×500 mm and is now the rig's default board geometry.
The saved `big-board` solve is dated 2026-10-06 08:03:38 UTC. Both automatic
reports are green despite the reported visual difference. Their nearest-
neighbour medians are 3.22 and 3.67 mm, respectively, and both reports warn
that this metric is likely flawed; it must **not** be used to claim that either
calibration has that accuracy. The two reports also concern different scenes,
so their other cloud scores are not a controlled A/B comparison.

## Working decision

For this two-Kinect rig, use the **large board** for new calibrations; do not
reuse the small-board `table-v1` preset for new demonstrations. The practical
failure was associated with calibration using the small board, and the large-
board recalibration restored visually acceptable cloud registration. Capture
board poses spanning the shared task area and hand-transfer height, with the
print visible in both cameras. Save the resulting preset with each recording;
recalibrate if either camera or the board-to-workspace geometry moves.

This supersedes the *operational* decision to treat the small-board calibration
as adequate for the task, but does not erase the measurements in
`2026-10-05-static-cubes-hand-depth-control.md`. That control found good rigid
table/cube agreement and separately identified unsupported near-range finger
depth under `NFOV_UNBINNED`. The new visual result does not establish that
the latter issue is gone or that board size alone caused every doubled finger.
Before calling the new dataset metrically validated, compare matched rigid
targets at tabletop **and hand-transfer heights**, check depth validity of the
moving fingers, and record independent error measures. The large-board
calibration is the present **working capture baseline**, not a certified
millimetre-accurate reconstruction.
