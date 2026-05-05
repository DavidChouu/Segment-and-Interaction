#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def next_index(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    max_index = -1
    for child in root.iterdir():
        if child.is_dir():
            try:
                max_index = max(max_index, int(child.name))
            except ValueError:
                pass
    return max_index + 1


def build_candidates(base_scenario):
    candidates = [
        {
            "name": "more_displacement_x_0p03",
            "kind": "displacement_only",
            "yellow_translation": [0.0, 0.0, 2.0],
            "black_translation": [0.0, 0.0, 0.0],
            "yellow_velocity": [-0.03, 0.0, 0.0],
            "black_velocity": [0.03, 0.0, 0.0],
        },
        {
            "name": "more_displacement_x_0p04",
            "kind": "displacement_only",
            "yellow_translation": [0.0, 0.0, 2.0],
            "black_translation": [0.0, 0.0, 0.0],
            "yellow_velocity": [-0.04, 0.0, 0.0],
            "black_velocity": [0.04, 0.0, 0.0],
        },
        {
            "name": "alignment_with_z_push_soft",
            "kind": "alignment_only",
            "yellow_translation": [0.0, 0.0, 2.0],
            "black_translation": [0.0, 0.0, 0.0],
            "yellow_velocity": [-0.02, 0.0, -0.015],
            "black_velocity": [0.02, 0.0, 0.015],
        },
        {
            "name": "alignment_with_z_push_strong",
            "kind": "alignment_only",
            "yellow_translation": [0.0, 0.0, 2.0],
            "black_translation": [0.0, 0.0, 0.0],
            "yellow_velocity": [-0.02, 0.0, -0.025],
            "black_velocity": [0.02, 0.0, 0.025],
        },
        {
            "name": "combo_more_x_and_z",
            "kind": "combo",
            "yellow_translation": [0.0, 0.0, 2.0],
            "black_translation": [0.0, 0.0, 0.0],
            "yellow_velocity": [-0.03, 0.0, -0.02],
            "black_velocity": [0.03, 0.0, 0.02],
        },
        {
            "name": "combo_less_offset_more_x",
            "kind": "combo",
            "yellow_translation": [0.0, 0.0, 1.5],
            "black_translation": [0.0, 0.0, 0.0],
            "yellow_velocity": [-0.03, 0.0, -0.01],
            "black_velocity": [0.03, 0.0, 0.01],
        },
    ]

    built = []
    for candidate in candidates:
        payload = json.loads(json.dumps(base_scenario))
        for obj in payload["objects"]:
            if obj["id"] == "yellow_toy_car":
                obj["placement"]["translation"] = candidate["yellow_translation"]
            elif obj["id"] == "black_toy_car":
                obj["placement"]["translation"] = candidate["black_translation"]

        interactions = payload.get("interactions", [])
        if len(interactions) >= 2:
            interactions[0]["velocity"] = candidate["yellow_velocity"]
            interactions[1]["velocity"] = candidate["black_velocity"]

        built.append({"scenario": payload, **candidate})
    return built


def main():
    parser = argparse.ArgumentParser(
        description="Append focused forward-vs-alignment candidates for the new ply setup."
    )
    parser.add_argument(
        "--base-scenario",
        default="config/toy_car_2_first_frame_shared_layout_trial_car_st.json",
    )
    parser.add_argument(
        "--entry-script",
        default="interaction_simulation_shared_layout.py",
    )
    parser.add_argument(
        "--choose-root",
        default="choose_ply",
    )
    parser.add_argument(
        "--generated-config-dir",
        default="generated_configs/choose_ply_forward_vs_alignment",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest_forward_vs_alignment.csv",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--frame-dt",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
    )
    parser.add_argument(
        "--clean-generated",
        action="store_true",
    )
    args = parser.parse_args()

    root = Path.cwd()
    base_scenario = load_json(root / args.base_scenario)
    choose_root = root / args.choose_root
    generated_dir = root / args.generated_config_dir
    manifest_path = choose_root / args.manifest_name

    if args.clean_generated:
        shutil.rmtree(generated_dir, ignore_errors=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    choose_root.mkdir(parents=True, exist_ok=True)

    start = next_index(choose_root)
    candidates = build_candidates(base_scenario)

    with manifest_path.open("w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "folder",
            "name",
            "kind",
            "yellow_translation",
            "black_translation",
            "yellow_velocity",
            "black_velocity",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for offset, candidate in enumerate(candidates):
            folder = f"{start + offset:04d}"
            scenario_path = generated_dir / f"{folder}.json"
            save_json(scenario_path, candidate["scenario"])

            cmd = [
                args.python_bin,
                "tools/render_and_archive_video_preview.py",
                "--scenario",
                str(scenario_path),
                "--entry-script",
                args.entry_script,
                "--archive-root",
                str(choose_root),
                "--output-path",
                f"tmp_forward_align_{folder}",
                "--frame-num",
                str(args.frame_num),
                "--frame-dt",
                str(args.frame_dt),
                "--python-bin",
                args.python_bin,
                "--forward-only",
            ]
            subprocess.run(cmd, check=True)

            writer.writerow(
                {
                    "folder": folder,
                    "name": candidate["name"],
                    "kind": candidate["kind"],
                    "yellow_translation": json.dumps(candidate["yellow_translation"]),
                    "black_translation": json.dumps(candidate["black_translation"]),
                    "yellow_velocity": json.dumps(candidate["yellow_velocity"]),
                    "black_velocity": json.dumps(candidate["black_velocity"]),
                }
            )


if __name__ == "__main__":
    main()
