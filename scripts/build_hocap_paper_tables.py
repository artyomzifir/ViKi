"""Regenerate thesis HO-Cap detail tables from the protected JSON snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "paper/results/hocap/pre_retarget"
OUTPUT = ROOT / "paper/thesis/hocap_evaluation_tables.tex"
DESCRIPTORS = {
    "G02": ("whole milk carton", "cooked ham", "chocolate drink powder", "sauce bottle"),
    "G04": ("flapjack mix", "chocolate fudge", "herring fillets can", "crackers box"),
    "G05": ("vegetable oil spread", "body lotion", "peanut chocolate box", "sponge"),
    "G06": ("DVD", "pumpkin creamer", "yellow mustard", "dish brush"),
    "G07": ("game controller", "hot cocoa mix", "dandruff shampoo", "toy axe"),
    "G09": ("candy box", "toy car", "toothpaste box", "chocolate syrup bottle"),
    "G10": ("soup mix", "gel toothpaste", "chocolate biscuit sticks", "body wash"),
    "G11": ("coconut water", "baby powder", "baking soda", "chocolate bar"),
    "G15": ("men's body wash", "dandelion tea", "cooking spray", "joy controller"),
    "G16": ("toilet cleaner", "laundry detergent", "small coconut water", "fabric softener sheets"),
    "G18": ("green tea latte mix", "projector remote", "right shoe", "left shoe"),
    "G19": ("electric screwdriver", "blue spatula", "deodorant", "ping pong paddle"),
    "G20": ("cappuccino mix", "mustard bottle", "toilet cleaner", "dog toy bone"),
    "G21": ("moisturizing lotion", "playing cards", "pink spatula", "blue brush"),
    "G22": ("chocolate bar", "mayomust sauce", "soup mix", "gray spatula"),
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_snapshot() -> None:
    paths = set()
    for line in (SNAPSHOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = SNAPSHOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"snapshot hash mismatch: {path}")
        paths.add(name)
    if len(paths) != 114:
        raise ValueError(f"expected 114 archived JSON files, found {len(paths)}")


def _short_name(name: str) -> str:
    parts = name.split("_")
    return f"S{parts[1]}/{parts[3]}"


def build_tables() -> str:
    _verify_snapshot()
    folders = sorted((SNAPSHOT / "episodes").iterdir())
    if len(folders) != 28:
        raise ValueError(f"expected 28 episodes, found {len(folders)}")
    hand_rows = []
    object_rows = []
    for folder in folders:
        hand = _read(folder / "gt/evaluation.json")
        objects = _read(folder / "intermediates/segmentation/hocap-oracle-v1/evaluation.json")
        score = hand["prepare"]["smoothed_all"]
        if hand["episode"] != folder.name or len(objects["objects"]) != 4:
            raise ValueError(f"invalid episode: {folder}")
        hand_rows.append((folder.name, hand["frame_count"], score))
        for object_id, result in sorted(objects["objects"].items()):
            group, number = object_id.split("_")
            if not (group in DESCRIPTORS and 1 <= int(number) <= 4):
                raise ValueError(f"unknown object: {object_id}")
            object_rows.append((folder.name, object_id, result))
    if len(object_rows) != 112:
        raise ValueError("expected 112 object tracks")

    lines = [
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{Eight-view HO-Cap hand results by sequence; PCK values are percentages.}\label{tab:hocap-hand-detail}\\",
        r"\toprule Sequence & Frames & Matched & MPJPE & P95 & PCK@20 & PCK@50 \\",
        r"\midrule\endfirsthead",
        r"\toprule Sequence & Frames & Matched & MPJPE & P95 & PCK@20 & PCK@50 \\",
        r"\midrule\endhead",
    ]
    for name, frames, score in hand_rows:
        lines.append(
            f"{_short_name(name)} & {frames} & {score['matched_joints']} & "
            f"{score['mpjpe_mm']:.2f} & {score['p95_mm']:.2f} & "
            f"{score['pck_20mm'] * 100:.2f} & {score['pck_50mm'] * 100:.2f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    lines += [
        r"\begin{longtable}{llrrrrr}",
        r"\caption{Per-object relative-pose tracking after one first-frame SE(3) alignment. T and R are translation in mm and geodesic rotation in degrees.}\label{tab:hocap-object-detail}\\",
        r"\toprule Sequence & Object & Cov. (\%) & T mean & T P95 & R mean & R P95 \\",
        r"\midrule\endfirsthead",
        r"\toprule Sequence & Object & Cov. (\%) & T mean & T P95 & R mean & R P95 \\",
        r"\midrule\endhead",
    ]
    for name, object_id, result in object_rows:
        lines.append(
            f"{_short_name(name)} & {object_id.replace('_', r'\_')} & "
            f"{result['coverage'] * 100:.2f} & "
            f"{result['translation_mean_mm']:.2f} & {result['translation_p95_mm']:.2f} & "
            f"{result['rotation_mean_deg']:.2f} & {result['rotation_p95_deg']:.2f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    lines += [
        r"\begin{longtable}{lp{100mm}}",
        r"\caption{Official HO-Cap object identities for the evaluated groups; suffix order is \texttt{\_1} through \texttt{\_4}.}\label{tab:hocap-object-catalog}\\",
        r"\toprule Group & Official object descriptions in suffix order \\",
        r"\midrule\endfirsthead",
        r"\toprule Group & Official object descriptions in suffix order \\",
        r"\midrule\endhead",
    ]
    for group, labels in DESCRIPTORS.items():
        lines.append(f"{group} & {'; '.join(labels)} " + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = build_tables()
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != generated:
            raise SystemExit("HO-Cap thesis tables differ from archived metrics")
    else:
        OUTPUT.write_text(generated, encoding="utf-8")
        print(OUTPUT)


if __name__ == "__main__":
    main()
