# License audit and remediation TODO

Status: open technical debt. Last reviewed: 2026-09-07.

ViKi's own source code is distributed under Apache-2.0, but that license does
not automatically cover third-party libraries, binary SDKs, model weights, or
robot-description assets. Before publishing a release or distributing the
Docker image, complete the actions below and record the exact dependency
versions that were reviewed.

## Required actions

- [ ] Remove `quadprog` from `pyproject.toml` unless it becomes an explicitly
  supported solver. It is GPL-2.0-or-later, is installed in the current image,
  and is not referenced by ViKi code. Retargeting currently selects OSQP.
- [ ] Decide whether LGPL-3.0 is acceptable for the product. The retargeting
  implementation imports `qpsolvers`, which is LGPL-3.0. If the dependency
  policy is strictly MIT/Apache-2.0, call the Apache-2.0 OSQP API directly and
  remove `qpsolvers`. Otherwise, document and satisfy the LGPL distribution
  requirements.
- [ ] Add a third-party notices bundle. At minimum, include the complete
  Three.js MIT license next to the vendored `three.module.js` and
  `OrbitControls.js`; the local `OrbitControls.js` copy currently has no license
  header of its own.
- [ ] Document proprietary runtime terms in distributed Docker images. The GPU
  image installs NVIDIA CUDA, cuDNN, cuBLAS, cuFFT, and cuRAND wheels governed
  by NVIDIA's license terms rather than MIT/Apache-2.0.
- [ ] Preserve and surface the Azure Kinect binary notices and EULA. The open
  source Sensor SDK is MIT, but the binary depth engine included in `libk4a` is
  proprietary, and the Dockerfile explicitly accepts its EULA.
- [ ] Introduce a robot-description allowlist with a recorded license per
  supported robot. The `robot_descriptions` Python loader is Apache-2.0, but
  every downloaded URDF and mesh set retains its own license. In particular,
  Universal Robots' UR8LONG, UR15, UR18, UR20, and UR30 meshes use custom
  Graphical Documentation terms that are not OSI-compliant.
- [ ] Apply the same allowlist to detachable grippers. Robotiq 2F-85 currently
  resolves from a BSD-2 source. Robotiq 2F-140, SCHUNK WSG-50 and SMC MHZ2
  variants must stay disabled until the exact URDF/mesh revision and
  redistribution terms are recorded; downloadable manufacturer CAD by itself
  is not permission to vendor converted meshes.
- [ ] Audit model weights separately from inference code. A framework being
  Apache-2.0 does not prove that a downloaded checkpoint or its training data
  has the same terms. Store the source URL, checksum, license, and permitted-use
  notes for every MediaPipe, RTMPose, and user-supplied MMPose artifact.
- [ ] Generate a version-locked SBOM/license report from the production image
  and review transitive Python and system packages. Commit the report or attach
  it to each release; do not rely only on the direct dependencies listed in
  `pyproject.toml`.

## Current findings

| Area | Component | License / terms | Current use |
|---|---|---|---|
| Frontend | Three.js r160 and OrbitControls | MIT | Runtime, vendored |
| Retarget | OSQP | Apache-2.0 | Selected QP backend |
| Retarget | qpsolvers | LGPL-3.0 | Runtime wrapper around OSQP |
| Retarget | quadprog | GPL-2.0-or-later | Installed, apparently unused |
| Perception | NVIDIA CUDA runtime libraries | NVIDIA proprietary terms | Installed in Docker for GPU inference |
| Capture | Azure Kinect depth engine | Microsoft proprietary terms / EULA | Included with the binary `libk4a` package |
| Robot assets | Supported UR and iiwa descriptions | Mostly BSD variants | Downloaded and cached at runtime |
| Robot assets | Selected newer UR meshes | Custom Universal Robots terms | Present in some description caches; not currently selectable |

BSD, ISC, PSF, and similar licenses also occur in the dependency tree. They are
not literally MIT or Apache-2.0, but they are generally permissive; their notice
and attribution requirements still need to be retained.

## Historical note

The removed React/Vite frontend had transitive build dependencies under
CC-BY-4.0, BSD, ISC, Python-2.0, and BlueOak-1.0.0. Those packages are not part
of the current plain HTML/CSS/JavaScript frontend, but this explains why an
older frontend license review did not report an MIT/Apache-only tree.

This file is an engineering checklist, not legal advice. Any public or
commercial distribution should receive a final review based on the exact
artifacts shipped.
