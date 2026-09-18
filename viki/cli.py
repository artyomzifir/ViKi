"""
viki.cli
--------
Thin command-line surface over the pipeline stages. Same package API the web
server calls — the CLI just parses args and prints results.

    viki record  --task "pick cube" --seconds 10
    viki perceive <episode>             # stable fused + hand_fit by default
    viki extract  <episode>
    viki cloud    <episode>            # raw/ -> cloud/ (viewer point cloud)
    viki auto-prompts <episode>        # calibrated foreground -> SAM prompts
    viki segment  <episode> ...        # prompted SAM 2.1 masks [+ RGB-D lift]
    viki object-model <episode>        # semantic cloud -> compact rigid models
    viki scene <episode>               # auto-prompts -> masks -> 3-D -> models
    viki prepare  <episode>
    viki geometry-fit <episode>        # clean cln -> anatomical A/B variants
    viki retarget <episode> --robot ur3
    viki replay   <episode> [--driver dryrun|ur3]
    viki label    <episode> --task "..." --outcome good
    viki export   --out data/datasets/pick <episode>...
    viki run      <episode>            # core pipeline; --scene-objects adds scene branch
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
    discovered = mgr.list_devices()
    devices = a.cameras or [
        *discovered.get("realsense", []), *discovered.get("kinect", []),
    ]
    try:
        kinect_ids = [device_id for device_id in devices if device_id.startswith("kinect_")]
        if len(discovered.get("kinect", [])) >= 2 and kinect_ids:
            mgr.start_configured_kinect_rig(fps=a.fps)
        else:
            for device_id in kinect_ids:
                mgr.start(device_id, fps=a.fps)
        for device_id in devices:
            if not device_id.startswith("kinect_"):
                mgr.start(device_id, fps=a.fps)
        rec = SceneRecorder(
            mgr,
            dataset=a.dataset,
            episodes_dir=None if a.dataset else a.episodes_dir,
            meta={"task": a.task, "demonstrator": a.demonstrator, "hand": a.hand},
        )
        ep = rec.record(a.seconds, fps=a.fps)
    finally:
        mgr.stop_all()
    print(ep.root)


def _cmd_extract(a) -> None:
    from viki.perception.extract import extract_episode

    print(extract_episode(_episode(a.episode), backend=a.backend, hand=a.hand))


def _cmd_perceive(a) -> None:
    from viki.perception.run import PerceiveOpts, perceive_episode

    print(perceive_episode(
        _episode(a.episode),
        PerceiveOpts(
            profile=a.profile,
            hand=a.hand,
            build_cloud=a.build_cloud,
            cloud_stride=a.cloud_stride,
        ),
    ))


def _cmd_cloud(a) -> None:
    from viki.perception.cloud import build_cloud

    print(build_cloud(_episode(a.episode)))


def _cmd_auto_prompts(a) -> None:
    from viki.perception import AutoPromptConfig, generate_auto_prompts

    cfg = AutoPromptConfig(
        frame=a.frame,
        object_count=a.objects,
        object_label=a.object_label,
        depth_stride=a.depth_stride,
        background_tolerance_mm=a.bg_tolerance_mm,
        min_object_saturation=a.min_saturation,
        max_object_extent_m=a.max_object_extent_mm / 1000.0,
    )
    print(generate_auto_prompts(
        _episode(a.episode),
        a.out,
        config=cfg,
    ))


def _cmd_segment(a) -> None:
    from viki.perception.scene import lift_segmented_episode, segment_episode

    def progress(**fields):
        camera = f" {fields['camera']}" if "camera" in fields else ""
        print(f"segment{camera}: {fields.get('frame', 0)}/{fields.get('total', 0)}")

    ep = _episode(a.episode)
    masks = segment_episode(
        ep,
        a.prompts,
        checkpoint=a.checkpoint,
        chunk_frames=a.chunk_frames,
        max_frames=a.max_frames,
        render_overlay=not a.no_overlay,
        report=progress,
    )
    print(f"masks: {masks}")
    if a.lift_3d:
        cloud = lift_segmented_episode(
            ep,
            masks,
            stride=a.depth_stride,
            voxel_m=a.voxel_m,
            report=progress,
        )
        print(f"semantic cloud: {cloud}")


def _cmd_segment_lift(a) -> None:
    from viki.perception.scene import lift_segmented_episode

    def progress(**fields):
        print(f"segment-lift: {fields.get('frame', 0)}/{fields.get('total', 0)}")

    ep = _episode(a.episode)
    masks = (
        Path(a.segmentation_dir)
        if a.segmentation_dir
        else ep.segmentation_dir
    )
    print(lift_segmented_episode(
        ep,
        masks,
        stride=a.depth_stride,
        voxel_m=a.voxel_m,
        report=progress,
    ))


def _cmd_object_model(a) -> None:
    from viki.perception.object_model import ObjectModelConfig
    from viki.perception.scene import build_object_models_episode

    def progress(**fields):
        print(
            f"{fields.get('stage', 'object-model')}: "
            f"{fields.get('frame', 0)}/{fields.get('total', 0)} "
            f"object={fields.get('object_id', '?')}"
        )

    ep = _episode(a.episode)
    segmentation = (
        Path(a.segmentation_dir)
        if a.segmentation_dir
        else ep.segmentation_dir
    )
    config = ObjectModelConfig(
        bootstrap_frames=a.bootstrap_frames,
        component_radius_m=a.component_radius_mm / 1000.0,
        inlier_distance_m=a.inlier_distance_mm / 1000.0,
    )
    print(build_object_models_episode(
        ep,
        segmentation,
        config=config,
        report=progress,
    ))


def _cmd_scene(a) -> None:
    from viki.perception.object_model import ObjectModelConfig
    from viki.perception.scene import ScenePerceptionOpts, scene_perception_episode

    def progress(**fields):
        camera = f" camera={fields['camera']}" if fields.get("camera") else ""
        obj = f" object={fields['object_id']}" if fields.get("object_id") else ""
        print(
            f"{fields.get('stage', 'scene')}: "
            f"{fields.get('frame', 0)}/{fields.get('total', 0)}{camera}{obj}"
        )

    opts = ScenePerceptionOpts(
        prompts=a.prompts,
        object_count=a.objects,
        object_label=a.object_label,
        checkpoint=a.checkpoint,
        chunk_frames=a.chunk_frames,
        render_overlay=not a.no_overlay,
        depth_stride=a.depth_stride,
        voxel_m=a.voxel_m,
        object_model=ObjectModelConfig(
            bootstrap_frames=a.bootstrap_frames,
            component_radius_m=a.component_radius_mm / 1000.0,
            inlier_distance_m=a.inlier_distance_mm / 1000.0,
        ),
    )
    print(scene_perception_episode(
        _episode(a.episode), opts, force=a.force, report=progress,
    ))


def _cmd_prepare(a) -> None:
    from viki.prepare.run import prepare_episode

    print(prepare_episode(
        _episode(a.episode), a.window, a.polyorder, profile=a.profile,
    ))


def _cmd_checkpoints(a) -> None:
    from viki.prepare.run import generate_stage_checkpoints

    outputs = generate_stage_checkpoints(
        _episode(a.episode), fusion_modes=tuple(a.fusion),
        window_length=a.window, polyorder=a.polyorder,
        interp_max_gap=a.interp_max_gap,
    )
    for mode, path in outputs.items():
        print(f"{mode}: {path}")


def _cmd_hand_fit(a) -> None:
    from viki.perception.hand_fit import refine_cln

    print(refine_cln(_episode(a.episode)))


def _cmd_geometry_fit(a) -> None:
    from viki.perception.articulated import (
        ArticulatedConfig,
        generate_articulated_variants,
        install_articulated_overlay,
    )

    ep = _episode(a.episode)
    result = generate_articulated_variants(
        ep,
        cfg=ArticulatedConfig(source_profile=a.source_profile),
    )
    print(f"projected: {result['projected']}")
    print(f"optimized: {result['optimized']}")
    print(f"report: {result['report']}")
    if a.install_overlay:
        overlay = install_articulated_overlay(
            ep,
            cfg=ArticulatedConfig(source_profile=a.source_profile),
            variant=a.install_overlay,
        )
        print(f"active hand-fit overlay: {overlay['overlay']} -> {overlay['path']}")


def _cmd_retarget(a) -> None:
    from viki.retarget.run import retarget_episode

    options = {
        key: value
        for key, value in {
            "gripper": a.gripper,
            "target_position_anchor": a.target_anchor,
            "base_position": a.base_position,
            "base_rpy_deg": a.base_rpy_deg,
            "adapter_translation_mm": a.adapter_offset_mm,
            "adapter_rpy_deg": a.adapter_rpy_deg,
            "adapter_radius_mm": a.adapter_radius_mm,
            "sequential_baseline": a.sequential_baseline,
        }.items()
        if value is not None
    }
    print(retarget_episode(_episode(a.episode), robot=a.robot, options=options))


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
    if a.format == "lerobot":
        from viki.export import export_dataset

        print(export_dataset(a.episodes, a.out, fps=a.fps))
    else:
        from viki.export import export_trajectories

        print(export_trajectories(a.episodes, a.out, name=a.name))


def _cmd_viz(a) -> None:
    from viki.render.skeleton3d import render_episode_figure

    ep = _episode(a.episode)
    fig = render_episode_figure(ep, stage=a.stage)
    out = a.out or str(ep.root / f"viz-{a.stage}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)


def _cmd_run(a) -> None:
    from viki.episode import stage_done
    from viki.perception.extract import extract_episode
    from viki.perception.profiles import DEFAULT_PERCEPTION_PROFILE
    from viki.prepare.run import prepare_episode
    from viki.replay import replay_episode
    from viki.retarget.run import retarget_episode

    ep = _episode(a.episode)
    steps = [
        ("extract", lambda: extract_episode(ep, backend=a.backend)),
        ("prepare", lambda: prepare_episode(ep, profile=DEFAULT_PERCEPTION_PROFILE)),
    ]
    if a.scene_objects > 0 or a.scene_prompts is not None:
        from viki.perception.scene import ScenePerceptionOpts, scene_perception_episode

        scene_options = ScenePerceptionOpts(
            prompts=a.scene_prompts,
            object_count=max(1, a.scene_objects),
            checkpoint=a.scene_checkpoint,
            chunk_frames=a.scene_chunk_frames,
        )
        steps.append((
            "scene",
            lambda: scene_perception_episode(ep, scene_options, force=a.force),
        ))
    steps.extend([
        ("retarget", lambda: retarget_episode(ep, robot=a.robot)),
        ("replay", lambda: replay_episode(ep, driver=a.driver)),
    ])
    for name, fn in steps:
        # The scene branch owns two resumable stages and validates their
        # artifacts internally; there is intentionally no synthetic `scene`
        # completion bit that could mask a missing segment or object model.
        if name != "scene" and stage_done(ep, name) and not a.force:
            print(f"= {name}: already done, skipping")
            continue
        print(f"→ {name}")
        print(f"  {fn()}")
    print("label + export are manual: viki label <ep> …  then  viki export …")


# ─────────────────────────────── parser ────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    from viki.perception.profiles import (
        CLEAN_LANDMARKS_V1,
        DEFAULT_PERCEPTION_PROFILE,
        FUSED_HAND_NO_EXTRAP_V1,
        FUSED_HAND_REPROJ8_V1,
        FUSED_HAND_REPROJ16_V1,
        STABLE_FUSED_HAND_V1,
        STABLE_FUSED_HAND_V2,
        STABLE_FUSED_HAND_V2_1,
        STABLE_FUSED_HAND_V3,
        STABLE_FUSED_HAND_V4,
    )

    p = argparse.ArgumentParser(prog="viki", description=__doc__.splitlines()[3])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="record a synced RGB-D scene into a new episode")
    pr.add_argument("--seconds", type=float, default=10.0)
    pr.add_argument("--fps", type=int, default=15)
    pr.add_argument("--task", default="")
    pr.add_argument("--demonstrator", default="")
    pr.add_argument("--hand", default="right", choices=["left", "right"])
    pr.add_argument("--cameras", nargs="*", help="device ids (default: all detected)")
    pr.add_argument("--dataset", default=None, help="dataset folder under data/datasets/")
    pr.add_argument("--episodes-dir", default="data/episodes")
    pr.set_defaults(func=_cmd_record)

    pe = sub.add_parser("extract", help="raw/ -> rec.npz")
    pe.add_argument("episode")
    pe.add_argument("--backend", default=None, help="pose backend (default: config)")
    pe.add_argument("--hand", default="right", choices=["left", "right"])
    pe.set_defaults(func=_cmd_extract)

    pper = sub.add_parser(
        "perceive",
        help="raw/ -> reproducible baseline rec.npz + joints3d.npz + cln.npz",
    )
    pper.add_argument("episode")
    pper.add_argument(
        "--profile",
        choices=[
            FUSED_HAND_NO_EXTRAP_V1,
            FUSED_HAND_REPROJ8_V1,
            FUSED_HAND_REPROJ16_V1,
            STABLE_FUSED_HAND_V4,
            STABLE_FUSED_HAND_V3,
            STABLE_FUSED_HAND_V2_1,
            STABLE_FUSED_HAND_V2,
            STABLE_FUSED_HAND_V1,
            CLEAN_LANDMARKS_V1,
        ],
        default=DEFAULT_PERCEPTION_PROFILE,
    )
    pper.add_argument("--hand", default="right", choices=["left", "right"])
    pper.add_argument("--build-cloud", action="store_true")
    pper.add_argument("--cloud-stride", type=int, default=1)
    pper.set_defaults(func=_cmd_perceive)

    pc = sub.add_parser("cloud", help="raw/ -> cloud/ (per-frame coloured point cloud)")
    pc.add_argument("episode")
    pc.set_defaults(func=_cmd_cloud)

    pap = sub.add_parser(
        "auto-prompts",
        help="calibrated background foreground -> automatic SAM prompt proposals",
    )
    pap.add_argument("episode")
    pap.add_argument("--out", default=None, help="output prompt JSON")
    pap.add_argument("--frame", type=int, default=0)
    pap.add_argument("--objects", type=int, default=3)
    pap.add_argument(
        "--object-label",
        choices=["manipulated_object", "other_object", "other_dynamic"],
        default="other_dynamic",
    )
    pap.add_argument("--depth-stride", type=int, default=2)
    pap.add_argument("--bg-tolerance-mm", type=float, default=50.0)
    pap.add_argument("--min-saturation", type=float, default=0.25)
    pap.add_argument("--max-object-extent-mm", type=float, default=250.0)
    pap.set_defaults(func=_cmd_auto_prompts)

    ps = sub.add_parser(
        "segment",
        help="experimental prompted SAM 2.1 masks and optional RGB-D lift",
    )
    ps.add_argument("episode")
    ps.add_argument("--prompts", required=True, help="viki_sam2_prompts_v1 JSON")
    ps.add_argument(
        "--checkpoint",
        default="models/sam2/sam2.1_hiera_small.pt",
    )
    ps.add_argument("--chunk-frames", type=int, default=100)
    ps.add_argument("--max-frames", type=int, default=None)
    ps.add_argument("--no-overlay", action="store_true")
    ps.add_argument("--lift-3d", action="store_true")
    ps.add_argument("--depth-stride", type=int, default=2)
    ps.add_argument("--voxel-m", type=float, default=0.004)
    ps.set_defaults(func=_cmd_segment)

    psl = sub.add_parser(
        "segment-lift",
        help="existing SAM masks -> calibrated semantic 3-D cloud",
    )
    psl.add_argument("episode")
    psl.add_argument("--segmentation-dir", default=None)
    psl.add_argument("--depth-stride", type=int, default=2)
    psl.add_argument("--voxel-m", type=float, default=0.004)
    psl.set_defaults(func=_cmd_segment_lift)

    pom = sub.add_parser(
        "object-model",
        help="experimental semantic cloud -> compact rigid object models",
    )
    pom.add_argument("episode")
    pom.add_argument("--segmentation-dir", default=None)
    pom.add_argument("--bootstrap-frames", type=int, default=60)
    pom.add_argument("--component-radius-mm", type=float, default=12.0)
    pom.add_argument("--inlier-distance-mm", type=float, default=12.0)
    pom.set_defaults(func=_cmd_object_model)

    pscene = sub.add_parser(
        "scene",
        help="scene RGB-D -> prompts -> SAM masks -> semantic cloud -> object models",
    )
    pscene.add_argument("episode")
    pscene.add_argument("--prompts", default=None, help="prompt JSON; default: derive from background")
    pscene.add_argument("--objects", type=int, default=1, help="non-operator objects for auto prompts")
    pscene.add_argument(
        "--object-label",
        choices=["manipulated_object", "other_object", "other_dynamic"],
        default="other_dynamic",
    )
    pscene.add_argument("--checkpoint", default="models/sam2/sam2.1_hiera_small.pt")
    pscene.add_argument("--chunk-frames", type=int, default=100)
    pscene.add_argument("--no-overlay", action="store_true")
    pscene.add_argument("--depth-stride", type=int, default=2)
    pscene.add_argument("--voxel-m", type=float, default=0.004)
    pscene.add_argument("--bootstrap-frames", type=int, default=60)
    pscene.add_argument("--component-radius-mm", type=float, default=12.0)
    pscene.add_argument("--inlier-distance-mm", type=float, default=12.0)
    pscene.add_argument("--force", action="store_true", help="rebuild completed scene stages")
    pscene.set_defaults(func=_cmd_scene)

    phf = sub.add_parser("hand-fit", help="batch-fit a capsule hand trajectory and append hand_fit_* to cln.npz")
    phf.add_argument("episode")
    phf.set_defaults(func=_cmd_hand_fit)

    pgf = sub.add_parser(
        "geometry-fit",
        help="build non-destructive landmark-only articulated hand variants",
    )
    pgf.add_argument("episode")
    pgf.add_argument(
        "--source-profile",
        default=CLEAN_LANDMARKS_V1,
        choices=[
            CLEAN_LANDMARKS_V1,
            FUSED_HAND_NO_EXTRAP_V1,
            STABLE_FUSED_HAND_V4,
            STABLE_FUSED_HAND_V3,
            STABLE_FUSED_HAND_V2_1,
            STABLE_FUSED_HAND_V2,
            STABLE_FUSED_HAND_V1,
        ],
        help="protected baseline used as immutable input",
    )
    pgf.add_argument(
        "--install-overlay",
        nargs="?",
        const="optimized",
        choices=["projected", "optimized"],
        help="also attach the candidate to active cln.npz for fused/hand-fit overlay",
    )
    pgf.set_defaults(func=_cmd_geometry_fit)

    pp = sub.add_parser("prepare", help="rec.npz -> cln.npz")
    pp.add_argument("episode")
    pp.add_argument("--window", type=int, default=7)
    pp.add_argument("--polyorder", type=int, default=2)
    pp.add_argument(
        "--profile",
        choices=[
            FUSED_HAND_NO_EXTRAP_V1,
            FUSED_HAND_REPROJ8_V1,
            FUSED_HAND_REPROJ16_V1,
            STABLE_FUSED_HAND_V4,
            STABLE_FUSED_HAND_V3,
            STABLE_FUSED_HAND_V2_1,
            STABLE_FUSED_HAND_V2,
            STABLE_FUSED_HAND_V1,
            CLEAN_LANDMARKS_V1,
        ],
        default=DEFAULT_PERCEPTION_PROFILE,
        help="locked reproducible recipe (overrides fusion/gap/SG/hand-fit config)",
    )
    pp.set_defaults(func=_cmd_prepare)

    pcp = sub.add_parser(
        "checkpoints",
        help="save non-destructive observed/filled/smoothed A/B prepare variants",
    )
    pcp.add_argument("episode")
    pcp.add_argument(
        "--fusion", nargs="+", default=["triangulate", "xyz_mean"],
        choices=["triangulate", "xyz_mean"],
    )
    pcp.add_argument("--interp-max-gap", type=int, default=None)
    pcp.add_argument("--window", type=int, default=7)
    pcp.add_argument("--polyorder", type=int, default=2)
    pcp.set_defaults(func=_cmd_checkpoints)

    pt = sub.add_parser("retarget", help="cln.npz -> plan.h5")
    pt.add_argument("episode")
    pt.add_argument("--robot", default=None)
    pt.add_argument("--gripper", default=None)
    pt.add_argument(
        "--target-anchor",
        choices=["pinch_center", "wrist"],
        default=None,
    )
    pt.add_argument("--base-position", nargs=3, type=float, default=None)
    pt.add_argument("--base-rpy-deg", nargs=3, type=float, default=None)
    pt.add_argument("--adapter-offset-mm", nargs=3, type=float, default=None)
    pt.add_argument("--adapter-rpy-deg", nargs=3, type=float, default=None)
    pt.add_argument("--adapter-radius-mm", type=float, default=None)
    pt.add_argument(
        "--sequential-baseline",
        action="store_true",
        default=None,
        help="also archive frame-wise IK + post-hoc smoothing for comparison",
    )
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

    px = sub.add_parser("export", help="retargeted episodes -> a dataset")
    px.add_argument("episodes", nargs="+")
    px.add_argument("--out", required=True)
    px.add_argument(
        "--format", choices=["trajectory", "lerobot"], default="trajectory",
        help="trajectory: self-contained npz bundle, no optional deps (default). "
             "lerobot: policy-training dataset with video; needs viki[export] and "
             "episodes that have been replayed and screened",
    )
    px.add_argument("--name", default=None, help="dataset name in the manifest")
    px.add_argument("--fps", type=int, default=15, help="lerobot format only")
    px.set_defaults(func=_cmd_export)

    pv = sub.add_parser("viz", help="headless 3-D figure of an episode (rec|cln)")
    pv.add_argument("episode")
    pv.add_argument("--stage", default="cln", choices=["rec", "cln"])
    pv.add_argument("--out", default=None, help="PNG path (default: <episode>/viz-<stage>.png)")
    pv.set_defaults(func=_cmd_viz)

    prn = sub.add_parser(
        "run",
        help="extract -> prepare -> [scene perception] -> retarget -> replay",
    )
    prn.add_argument("episode")
    prn.add_argument("--robot", default=None)
    prn.add_argument("--backend", default=None)
    prn.add_argument("--driver", default="dryrun", choices=["dryrun", "ur3"])
    prn.add_argument(
        "--scene-objects",
        type=int,
        default=0,
        help="also build scene/object artifacts for this many non-operator objects",
    )
    prn.add_argument(
        "--scene-prompts",
        default=None,
        help="prompt JSON for the scene branch (implies scene perception)",
    )
    prn.add_argument(
        "--scene-checkpoint",
        default="models/sam2/sam2.1_hiera_small.pt",
    )
    prn.add_argument("--scene-chunk-frames", type=int, default=100)
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
