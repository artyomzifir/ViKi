# ViKi — Vision-based Kinematic Imitation

> Video-to-Kinematics for robotics: capture human demonstrations with RGB-D cameras, retarget motions to robots, and generate LeRobot datasets.

ViKi turns multi-view RGB-D video of a human doing a manipulation task into a robot-ready demonstration dataset — no teleoperation rig required.

**🚧 Work in progress.** The capture → skeleton → retarget pipeline runs end to end, but tracking accuracy, multi-camera sync, and the hand-fit model are under active tuning. Interfaces and defaults are still moving.

---

## How it works

```
Multi-view RGB-D capture   ← RealSense D435i + Azure Kinect DK (HW-synced)
        │
        ▼
3D hand skeleton           ← MediaPipe + multi-view triangulation + depth
        │
        ▼
Smoothing & fusion         ← cross-camera fuse, Savitzky-Golay, articulated hand model
        │
        ▼
Robot retargeting          ← object-relative IK via PINK / Pinocchio
        │
        ▼
LeRobot dataset            ← for ACT / Diffusion Policy training     [export: stub]
```

Human video is cheap and abundant, but naive retargeting from human to robot kinematics produces noisy, jerky trajectories. ViKi's optimisation stage exists to close that gap.

---

## Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for USB configuration, Docker setup, and multi-Kinect sync wiring.

```bash
sudo ./scripts/host_setup.sh   # once per host
docker compose up --build
# open http://localhost:8000
```

The web UI (`viki/server/static/`) is plain HTML/CSS/JS served directly by the FastAPI server.

---

## Development

```bash
docker compose run --rm test                          # full test suite
docker compose run --rm test tests/unit_tests/gripper  # a subset (args append to pytest)
docker compose run --rm cli <verb> ...                 # viki record/extract/prepare/retarget/replay/label/export/run
docker compose run --rm terminal                       # bash shell in the container
```

### Pipeline stages

| stage | package | status |
|---|---|---|
| record | [`viki/cameras`](viki/cameras/README.md) | done |
| calibrate | [`viki/calibration`](viki/calibration/README.md) | done |
| extract (skeleton) | [`viki/perception`](viki/perception/README.md) | in progress — accuracy tuning |
| prepare (fuse + smooth) | [`viki/prepare`](viki/prepare/README.md) | in progress |
| retarget (IK) | [`viki/retarget`](viki/retarget/README.md) | done |
| replay (hardware validation) | `viki/replay` | stub |
| export (LeRobot dataset) | `viki/export` | stub |

See [`viki/README.md`](viki/README.md) for the full package map and cross-cutting rules.

---

## License

See [LICENSE](LICENSE).
