"""Eight-view PressureVision++ diagnostic; learned hints, not force ground truth.

Use the pinned upstream code via PYTHONPATH and its published paper_29.pth.
Only RGB enters the network. HO-Cap GT joints/calibration provide a reproducible
hand crop and per-digit image regions; no robot simulation labels enter inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')
BONES = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
# Exactly upstream classes_to_scalar with paper.yml thresholds, including last bin.
PRESSURE_VALUES = np.array([0, .75, 1.5, 3, 6, 12, 24, 48, 96], np.float32)


def hand_crop(uv, width, height):
    uv = np.asarray(uv, float)
    if uv.shape != (21, 2) or not np.isfinite(uv).all():
        raise ValueError('finite 21-joint image coordinates required')
    lo, hi = uv.min(0), uv.max(0)
    center = (lo + hi) / 2
    radius = max(hi - lo) * .75  # upstream square bbox, scale 1.5
    x0, y0 = np.maximum(0, np.rint(center - radius)).astype(int)
    x1, y1 = np.minimum([width, height], np.rint(center + radius)).astype(int)
    if min(x1 - x0, y1 - y0) < 8:
        raise ValueError('hand crop too small/outside image')
    return int(x0), int(y0), int(x1), int(y1)


def digit_regions(uv, radii_px, shape=(448, 448)):
    """Assign overlapping projected capsule regions to the nearest normalized axis.

    This is an image ROI proxy, not visibility or true skin segmentation. Preserve
    all per-view evidence; fused hints must not be described as measured forces.
    """
    yy, xx = np.indices(shape, dtype=np.float32)
    grid = np.stack([xx, yy], -1)
    best = np.full(shape, np.inf, np.float32)
    labels = np.full(shape, -1, np.int8)
    for bone, start in enumerate(BONES):
        a, b = uv[start], uv[start + 1]
        delta = b - a
        t = np.clip(np.sum((grid - a) * delta, -1) / max(float(delta @ delta), 1e-8), 0, 1)
        dist = np.linalg.norm(grid - (a + t[..., None] * delta), axis=-1)
        norm = dist / max(float(radii_px[bone]), 1.)
        mask = (norm <= 1) & (norm < best)
        labels[mask] = bone // 3
        best[mask] = norm[mask]
    return labels


def fuse_views(values):
    """Mean of strongest two views; attenuates but does not reject a single spike."""
    values = np.asarray(values, float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[-1] != 5:
        raise ValueError('expected view x frame x five digits')
    return np.sort(values, axis=0)[-2:].mean(axis=0)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=Path('/app/data/datasets/hocap_demo_object_v4/g10_object2_190'))
    parser.add_argument('--output', type=Path, default=Path('/app/rl-output/pressure_hint_20261026'))
    parser.add_argument('--frames', type=int, default=190)
    args = parser.parse_args()
    ep, out = args.episode, args.output
    cameras = sorted(p.name for p in (out / 'rgb').iterdir() if p.is_dir())
    if len(cameras) != 8:
        raise ValueError('this protocol requires all eight source cameras')
    intr = json.loads((ep / 'raw/intrinsics.json').read_text())
    ext = json.loads((ep / 'raw/extrinsics.json').read_text())
    with np.load(ep / 'gt/joints3d_world.npz') as g:
        joints = g['joints_world'][:args.frames].astype(float)
        if not g['valid'][:args.frames].all():
            raise ValueError('GT crop protocol requires valid source joints')
    with np.load(ep / 'cln.npz') as clean:
        radii = clean['hand_fit_capsule_radii'][1:].astype(float)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Published upstream full-module checkpoint; loaded in an isolated container.
    model = torch.load(out / 'paper_29.pth', map_location='cpu', weights_only=False).eval().to(device)
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    roi_mean = np.zeros((8, len(joints), 5), np.float32)
    roi_expected = np.zeros_like(roi_mean)
    roi_nonzero_probability = np.zeros_like(roi_mean)
    roi_p95 = np.zeros_like(roi_mean)
    roi_contact = np.zeros_like(roi_mean)
    aux = np.zeros((8, len(joints), 7), np.float32)
    bboxes = np.zeros((8, len(joints), 4), np.int32)
    previews = {k: [] for k in (20, 42, 60, 90, 120, 145, 175) if k < len(joints)}
    colors = ((0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 150, 0), (255, 0, 200))
    for ci, camera in enumerate(cameras):
        cam = intr[camera]['color']; ex = ext[camera]
        K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1.]])
        rvec, tvec = np.asarray(ex['rvec']), np.asarray(ex['tvec'])
        rotation = cv2.Rodrigues(rvec)[0]
        classes = np.zeros((len(joints), 448, 448), np.uint8)
        for frame, points in enumerate(joints):
            raw = cv2.imread(str(out / 'rgb' / camera / f'{frame:06}.jpg'))
            if raw is None:
                raise FileNotFoundError(f'{camera}/{frame:06}.jpg')
            uv = cv2.projectPoints(points, rvec, tvec, K, np.array(cam['dist_coeffs']))[0].reshape(21, 2)
            box = hand_crop(uv, raw.shape[1], raw.shape[0]); bboxes[ci, frame] = box
            x0, y0, x1, y1 = box
            crop = cv2.resize(raw[y0:y1, x0:x1], (448, 448))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred, heads = model(tensor)
                class_map = pred.argmax(1)[0].cpu().numpy().astype(np.uint8)
                probabilities = pred.softmax(1)
                expected_map = (probabilities * torch.as_tensor(PRESSURE_VALUES, device=device)[None, :, None, None]).sum(1)[0].cpu().numpy()
                nonzero_map = (1 - probabilities[0, 0]).cpu().numpy()
                aux[ci, frame] = heads['bottleneck_logits'][0].cpu().numpy()
            classes[frame] = class_map
            pressure = PRESSURE_VALUES[class_map]
            crop_uv = (uv - [x0, y0]) * [448 / (x1 - x0), 448 / (y1 - y0)]
            camera_points = points @ rotation.T + tvec
            bone_z = np.array([(camera_points[j, 2] + camera_points[j + 1, 2]) / 2 for j in BONES])
            image_scale = .5 * (448 / (x1 - x0) + 448 / (y1 - y0))
            radii_px = radii * .5 * (cam['fx'] + cam['fy']) / bone_z * image_scale
            regions = digit_regions(crop_uv, radii_px)
            for digit in range(5):
                values = pressure[regions == digit]
                if len(values):
                    roi_mean[ci, frame, digit] = values.mean()
                    roi_expected[ci, frame, digit] = expected_map[regions == digit].mean()
                    roi_nonzero_probability[ci, frame, digit] = nonzero_map[regions == digit].mean()
                    roi_p95[ci, frame, digit] = np.percentile(values, 95)
                    roi_contact[ci, frame, digit] = np.mean(values > 0)
            if frame in previews:
                heat = cv2.applyColorMap(np.uint8(np.clip(pressure / 32, 0, 1) * 255), cv2.COLORMAP_TURBO)
                heat[pressure == 0] = 0
                overlay = cv2.addWeighted(crop, .65, heat, .65, 0)
                for finger, start in enumerate((1, 5, 9, 13, 17)):
                    for j in range(start, start + 3):
                        cv2.line(overlay, tuple(np.rint(crop_uv[j]).astype(int)), tuple(np.rint(crop_uv[j + 1]).astype(int)), colors[finger], 2)
                tile = np.vstack([cv2.resize(crop, (224, 224)), cv2.resize(overlay, (224, 224))])
                cv2.putText(tile, camera, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
                previews[frame].append(tile)
            if frame % 60 == 0:
                print(camera, frame, 'aux', aux[ci, frame, :5].round(2), flush=True)
        np.savez_compressed(out / f'pressure_maps_{camera}.npz', classes=classes, class_kpa=PRESSURE_VALUES)
    for frame, tiles in previews.items():
        cv2.imwrite(str(out / f'views_pressure_{frame:03}.jpg'), np.hstack(tiles))
    fused_pressure = fuse_views(roi_mean)
    fused_contact = fuse_views(aux[:, :, :5])
    fused_expected = fuse_views(roi_expected)
    fused_probability = fuse_views(roi_nonzero_probability)
    # Spatial pressure magnitude times predicted digit participation; normalized
    # to relative weights only, with no claim of a calibrated newton conversion.
    importance = fused_pressure * fused_contact
    np.savez_compressed(out / 'pressure_hints.npz', cameras=cameras, digits=DIGITS,
                        roi_mean_kpa=roi_mean, roi_expected_kpa=roi_expected, roi_nonzero_probability=roi_nonzero_probability, roi_p95_kpa=roi_p95, roi_contact_fraction=roi_contact,
                        auxiliary_predictions=aux, bboxes=bboxes, fused_pressure_proxy_kpa=fused_pressure,
                        fused_digit_participation=fused_contact, fused_expected_pressure_proxy_kpa=fused_expected, fused_nonzero_probability=fused_probability, relative_importance=importance)
    report = {'protocol': 'PressureVision++ eight-view RGB pilot; HO-Cap GT crops/ROIs',
              'code_revision': (out / 'pressurevision2/REVISION').read_text().strip(),
              'checkpoint_sha256': hashlib.sha256((out / 'paper_29.pth').read_bytes()).hexdigest(),
              'cameras': cameras, 'frames': len(joints), 'device': device,
              'source_phase_frames': {'pre': [0, 20], 'carry': [42, 126], 'post': [160, 190]},
              'limits': ['network uses RGB only; depth/geometry may be used by downstream polisher',
                         'digit ROIs are projected capsule approximations, not skin masks',
                         'top-two view fusion is a fixed heuristic, not a trained multiview model',
                         'pressure predictions and weak contact head are hints, not sensor truth',
                         '2D pixel averages are not integrated force in newtons'],
              'phases': {}}
    for phase, (start, stop) in report['source_phase_frames'].items():
        if start >= len(joints): continue
        report['phases'][phase] = {'pressure_proxy_kpa_median': np.median(fused_pressure[start:stop], 0).tolist(),
                                  'digit_participation_median': np.median(fused_contact[start:stop], 0).tolist(),
                                  'expected_pressure_proxy_kpa_median': np.median(fused_expected[start:stop], 0).tolist(),
                                  'nonzero_probability_median': np.median(fused_probability[start:stop], 0).tolist(),
                                  'importance_median': np.median(importance[start:stop], 0).tolist()}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
