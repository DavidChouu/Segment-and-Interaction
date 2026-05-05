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


def build_candidates(base_scenario):
    candidates = [
        {
            "name": "same_lane_push_no_rebound",
            "simulation_overrides": {
                "E": 1.6e6,
                "yield_stress": 4.0e7,
                "grid_v_damping_scale": 0.998,
                "rpic_damping": 0.10,
            },
            "placements": {
                "yellow_toy_car": [0.0, 0.0, 0.8],
                "black_toy_car": [0.0, 0.0, 0.0],
            },
            "interactions": [
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [-0.050, 0.0, -0.010],
                    "start_time": 0.6,
                    "end_time": 5.45,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [0.050, 0.0, 0.010],
                    "start_time": 0.6,
                    "end_time": 5.45,
                },
            ],
        },
        {
            "name": "same_lane_push_tiny_release",
            "simulation_overrides": {
                "E": 1.2e6,
                "yield_stress": 3.0e7,
                "grid_v_damping_scale": 0.998,
                "rpic_damping": 0.12,
            },
            "placements": {
                "yellow_toy_car": [0.0, 0.0, 0.8],
                "black_toy_car": [0.0, 0.0, 0.0],
            },
            "interactions": [
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [-0.048, 0.0, -0.010],
                    "start_time": 0.6,
                    "end_time": 5.25,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [0.048, 0.0, 0.010],
                    "start_time": 0.6,
                    "end_time": 5.25,
                },
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [0.006, 0.0, 0.001],
                    "start_time": 5.35,
                    "end_time": 5.45,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [-0.006, 0.0, -0.001],
                    "start_time": 5.35,
                    "end_time": 5.45,
                },
            ],
        },
        {
            "name": "half_lane_offset_no_rebound",
            "simulation_overrides": {
                "E": 8.0e5,
                "yield_stress": 2.0e7,
                "grid_v_damping_scale": 0.996,
                "rpic_damping": 0.16,
            },
            "placements": {
                "yellow_toy_car": [0.0, 0.0, 1.2],
                "black_toy_car": [0.0, 0.0, 0.0],
            },
            "interactions": [
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [-0.046, 0.0, -0.012],
                    "start_time": 0.6,
                    "end_time": 5.55,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [0.046, 0.0, 0.012],
                    "start_time": 0.6,
                    "end_time": 5.55,
                },
            ],
        },
        {
            "name": "half_lane_offset_tiny_release",
            "simulation_overrides": {
                "E": 4.5e5,
                "yield_stress": 1.2e7,
                "grid_v_damping_scale": 0.996,
                "rpic_damping": 0.20,
            },
            "placements": {
                "yellow_toy_car": [0.0, 0.0, 1.2],
                "black_toy_car": [0.0, 0.0, 0.0],
            },
            "interactions": [
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [-0.044, 0.0, -0.012],
                    "start_time": 0.6,
                    "end_time": 5.3,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [0.044, 0.0, 0.012],
                    "start_time": 0.6,
                    "end_time": 5.3,
                },
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [0.008, 0.0, 0.002],
                    "start_time": 5.4,
                    "end_time": 5.55,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [-0.008, 0.0, -0.002],
                    "start_time": 5.4,
                    "end_time": 5.55,
                },
            ],
        },
        {
            "name": "same_lane_fastest_small_release",
            "simulation_overrides": {
                "E": 2.0e6,
                "yield_stress": 5.0e7,
                "grid_v_damping_scale": 0.998,
                "rpic_damping": 0.08,
            },
            "placements": {
                "yellow_toy_car": [0.0, 0.0, 0.4],
                "black_toy_car": [0.0, 0.0, 0.0],
            },
            "interactions": [
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [-0.055, 0.0, -0.006],
                    "start_time": 0.6,
                    "end_time": 5.15,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [0.055, 0.0, 0.006],
                    "start_time": 0.6,
                    "end_time": 5.15,
                },
                {
                    "type": "object_translation",
                    "object_id": "yellow_toy_car",
                    "velocity": [0.010, 0.0, 0.001],
                    "start_time": 5.25,
                    "end_time": 5.45,
                },
                {
                    "type": "object_translation",
                    "object_id": "black_toy_car",
                    "velocity": [-0.010, 0.0, -0.001],
                    "start_time": 5.25,
                    "end_time": 5.45,
                },
            ],
        },
    ]

    built = []
    for candidate in candidates:
        payload = json.loads(json.dumps(base_scenario))
        payload["simulation_overrides"] = candidate["simulation_overrides"]
        payload["interactions"] = candidate["interactions"]
        placements = candidate.get("placements", {})
        for obj in payload["objects"]:
            if obj["id"] in placements:
                obj["placement"]["translation"] = placements[obj["id"]]
        built.append(
            {
                "name": candidate["name"],
                "scenario": payload,
                "simulation_overrides": candidate["simulation_overrides"],
                "interactions": candidate["interactions"],
                "placements": placements,
            }
        )
    return built


def main():
    parser = argparse.ArgumentParser(
        description="Render multiple new-ply collision candidates into choose_ply/ with 10-frame previews."
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
        default="generated_configs/choose_ply_candidates",
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
        "--clean",
        action="store_true",
    )
    args = parser.parse_args()

    root = Path.cwd()
    base_scenario = load_json(root / args.base_scenario)
    choose_root = root / args.choose_root
    generated_config_dir = root / args.generated_config_dir
    manifest_path = choose_root / "manifest.csv"

    if args.clean:
        shutil.rmtree(choose_root, ignore_errors=True)
        shutil.rmtree(generated_config_dir, ignore_errors=True)

    choose_root.mkdir(parents=True, exist_ok=True)
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(base_scenario)

    with manifest_path.open("w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "folder",
            "name",
            "yellow_translation",
            "black_translation",
            "E",
            "yield_stress",
            "grid_v_damping_scale",
            "rpic_damping",
            "approach_velocity",
            "approach_end",
            "rebound_velocity",
            "rebound_start",
            "rebound_end",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for index, candidate in enumerate(candidates):
            folder_name = f"{index:04d}"
            scenario_path = generated_config_dir / f"{folder_name}.json"
            save_json(scenario_path, candidate["scenario"])

            render_cmd = [
                args.python_bin,
                "tools/render_and_archive_video_preview.py",
                "--scenario",
                str(scenario_path),
                "--entry-script",
                args.entry_script,
                "--output-path",
                f"tmp_choose_ply_render_{folder_name}",
                "--archive-root",
                str(choose_root),
                "--frame-num",
                str(args.frame_num),
                "--frame-dt",
                str(args.frame_dt),
                "--python-bin",
                args.python_bin,
            ]
            subprocess.run(render_cmd, check=True)

            sim = candidate["simulation_overrides"]
            interactions = candidate["interactions"]
            rebound_velocity = ""
            rebound_start = ""
            rebound_end = ""
            if len(interactions) >= 4:
                rebound_velocity = interactions[2]["velocity"][0]
                rebound_start = interactions[2]["start_time"]
                rebound_end = interactions[2]["end_time"]
            writer.writerow(
                {
                    "folder": folder_name,
                    "name": candidate["name"],
                    "yellow_translation": json.dumps(candidate["placements"].get("yellow_toy_car", [])),
                    "black_translation": json.dumps(candidate["placements"].get("black_toy_car", [])),
                    "E": sim["E"],
                    "yield_stress": sim["yield_stress"],
                    "grid_v_damping_scale": sim["grid_v_damping_scale"],
                    "rpic_damping": sim["rpic_damping"],
                    "approach_velocity": interactions[0]["velocity"][0],
                    "approach_end": interactions[0]["end_time"],
                    "rebound_velocity": rebound_velocity,
                    "rebound_start": rebound_start,
                    "rebound_end": rebound_end,
                }
            )


if __name__ == "__main__":
    main()
