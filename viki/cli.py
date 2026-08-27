"""
viki.cli
--------
Thin command-line surface over the pipeline stages. Same package API the web
server calls — the CLI just parses args and prints results.

    viki record  --task "pick cube" --seconds 10
    viki extract  <episode>
    viki prepare  <episode>
    viki retarget <episode> --robot ur3
    viki replay   <episode> [--driver dryrun|ur3]
    viki label    <episode> --task "..." --outcome good
    viki export   --out data/datasets/pick <episode>...
    viki run      <episode>            # extract -> prepare -> retarget -> replay
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from viki.contracts import Episode


def _episode(arg: str) -> Episode:
    return Episode(root=Path(arg))


# ─────────────────────────── stage handlers ────────────────────────────


def _cmd_record(a) -> None:
    from viki.cameras.manager import CameraManager
    from viki.cameras.record import SceneRecorder

    mgr = CameraManager()
    for dev in a.cameras or mgr.list_devices().get("realsense", []):
        mgr.start(dev)
    rec = SceneRecorder(
        mgr,
        episodes_dir=a.episodes_dir,
        meta={"task": a.task, "demonstrator": a.demonstrator, "hand": a.hand},
    )
    ep = rec.record(a.seconds, fps=a.fps)
    mgr.stop_all()
    print(ep.root)


def _cmd_extract(a) -> None:
    from viki.perception.extract import extract_episode

    print(extract_episode(_episode(a.episode), backend=a.backend, hand=a.hand))


def _cmd_prepare(a) -> None:
    from viki.prepare.run import prepare_episode

    print(prepare_episode(_episode(a.episode), a.window, a.polyorder))


def _cmd_retarget(a) -> None:
    from viki.retarget.run import retarget_episode

    print(retarget_episode(_episode(a.episode), robot=a.robot))


def _cmd_replay(a) -> None:
    from viki.replay import replay_episode

    print(replay_episode(_episode(a.episode), driver=a.driver, max_resolves=a.max_resolves))


def _cmd_label(a) -> None:
    from viki.contracts import EpisodeLabels
    from viki.labeling import load_labels, save_labels

    ep = _episode(a.episode)
    labels = load_labels(ep)
    if a.task is not None:
        labels = EpisodeLabels(
            task=a.task, hand=a.hand or labels.hand,
            segments=labels.segments, outcome=a.outcome or labels.outcome,
            notes=labels.notes,
        )
        save_labels(ep, labels)
    print(labels)


def _cmd_export(a) -> None:
    from viki.export import export_dataset

    print(export_dataset(a.episodes, a.out, fps=a.fps))


def _cmd_run(a) -> None:
    from viki.episode import stage_done
    from viki.perception.extract import extract_episode
    from viki.prepare.run import prepare_episode
    from viki.replay import replay_episode
    from viki.retarget.run import retarget_episode

    ep = _episode(a.episode)
    steps = [
        ("extract", lambda: extract_episode(ep, backend=a.backend)),
        ("prepare", lambda: prepare_episode(ep)),
        ("retarget", lambda: retarget_episode(ep, robot=a.robot)),
        ("replay", lambda: replay_episode(ep, driver=a.driver)),
    ]
    for name, fn in steps:
        if stage_done(ep, name) and not a.force:
            print(f"= {name}: already done, skipping")
            continue
        print(f"→ {name}")
        print(f"  {fn()}")
    print("label + export are manual: viki label <ep> …  then  viki export …")


# ─────────────────────────────── parser ────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="viki", description=__doc__.splitlines()[3])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="record a synced RGB-D scene into a new episode")
    pr.add_argument("--seconds", type=float, default=10.0)
    pr.add_argument("--fps", type=int, default=15)
    pr.add_argument("--task", default="")
    pr.add_argument("--demonstrator", default="")
    pr.add_argument("--hand", default="right", choices=["left", "right"])
    pr.add_argument("--cameras", nargs="*", help="device ids (default: all detected)")
    pr.add_argument("--episodes-dir", default="data/episodes")
    pr.set_defaults(func=_cmd_record)

    pe = sub.add_parser("extract", help="raw/ -> rec.npz")
    pe.add_argument("episode")
    pe.add_argument("--backend", default=None, help="pose backend (default: config)")
    pe.add_argument("--hand", default="right", choices=["left", "right"])
    pe.set_defaults(func=_cmd_extract)

    pp = sub.add_parser("prepare", help="rec.npz -> cln.npz")
    pp.add_argument("episode")
    pp.add_argument("--window", type=int, default=7)
    pp.add_argument("--polyorder", type=int, default=2)
    pp.set_defaults(func=_cmd_prepare)

    pt = sub.add_parser("retarget", help="cln.npz -> plan.h5")
    pt.add_argument("episode")
    pt.add_argument("--robot", default=None)
    pt.set_defaults(func=_cmd_retarget)

    prp = sub.add_parser("replay", help="plan.h5 -> replay.h5 (stub stage)")
    prp.add_argument("episode")
    prp.add_argument("--driver", default="dryrun", choices=["dryrun", "ur3"])
    prp.add_argument("--max-resolves", type=int, default=0)
    prp.set_defaults(func=_cmd_replay)

    pl = sub.add_parser("label", help="get/set episode labels")
    pl.add_argument("episode")
    pl.add_argument("--task", default=None)
    pl.add_argument("--hand", default=None, choices=["left", "right"])
    pl.add_argument("--outcome", default=None, choices=["good", "bad", "unrated"])
    pl.set_defaults(func=_cmd_label)

    px = sub.add_parser("export", help="labelled episodes -> LeRobot dataset")
    px.add_argument("episodes", nargs="+")
    px.add_argument("--out", required=True)
    px.add_argument("--fps", type=int, default=15)
    px.set_defaults(func=_cmd_export)

    prn = sub.add_parser("run", help="extract -> prepare -> retarget -> replay")
    prn.add_argument("episode")
    prn.add_argument("--robot", default=None)
    prn.add_argument("--backend", default=None)
    prn.add_argument("--driver", default="dryrun", choices=["dryrun", "ur3"])
    prn.add_argument("--force", action="store_true", help="rerun done stages")
    prn.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
