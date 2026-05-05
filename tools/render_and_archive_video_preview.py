#!/usr/bin/env python3
import argparse
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


def next_version_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    max_index = -1
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            max_index = max(max_index, int(child.name))
        except ValueError:
            continue
    return root / f"{max_index + 1:04d}"


def rescale_interaction_times(
    interactions,
    preview_duration,
    start_padding_ratio=0.05,
    end_padding_ratio=0.95,
    preserve_displacement=True,
    forward_only=False,
):
    working = list(interactions)
    if forward_only:
        per_object_seen = {}
        filtered = []
        for item in working:
            object_id = item.get("object_id")
            if object_id is None:
                filtered.append(item)
                continue
            count = per_object_seen.get(object_id, 0)
            if count == 0:
                filtered.append(item)
            per_object_seen[object_id] = count + 1
        working = filtered

    timed = [item for item in working if "start_time" in item and "end_time" in item]
    if not timed:
        return working

    min_start = min(float(item["start_time"]) for item in timed)
    max_end = max(float(item["end_time"]) for item in timed)
    original_duration = max(max_end - min_start, 1e-6)

    preview_start = preview_duration * start_padding_ratio
    preview_end = preview_duration * end_padding_ratio
    target_duration = max(preview_end - preview_start, 1e-6)
    scale = target_duration / original_duration
    velocity_scale = 1.0 / scale if preserve_displacement else 1.0

    rescaled = []
    for item in working:
        copied = dict(item)
        if "start_time" in copied and "end_time" in copied:
            copied["start_time"] = (float(copied["start_time"]) - min_start) * scale + preview_start
            copied["end_time"] = (float(copied["end_time"]) - min_start) * scale + preview_start
        if preserve_displacement and "velocity" in copied:
            copied["velocity"] = [float(v) * velocity_scale for v in copied["velocity"]]
        rescaled.append(copied)
    return rescaled


def main():
    parser = argparse.ArgumentParser(
        description="Render a short multi-frame interaction preview and archive frames into a numbered folder."
    )
    parser.add_argument(
        "--scenario",
        default="config/toy_car_2_first_frame_shared_layout_trial_car_st.json",
        help="Base scenario json to render from.",
    )
    parser.add_argument(
        "--entry-script",
        default="interaction_simulation_shared_layout.py",
        help="Simulation entry script to run.",
    )
    parser.add_argument(
        "--output-path",
        default="tmp_video_inter_new_render",
        help="Temporary render output directory. This folder is overwritten each run.",
    )
    parser.add_argument(
        "--archive-root",
        default="video_inter_new",
        help="Root folder that stores incrementally numbered preview directories.",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=10,
        help="How many frames to render and archive.",
    )
    parser.add_argument(
        "--frame-dt",
        type=float,
        default=0.04,
        help="Frame dt override for the preview scenario.",
    )
    parser.add_argument(
        "--no-preserve-displacement",
        action="store_true",
        help="Only compress time without scaling velocities.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Temporarily drop later per-object motion segments, keeping only the first forward motion for each object.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run the simulation.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    scenario_path = root / args.scenario
    temp_output = root / args.output_path
    archive_root = root / args.archive_root
    temp_scenario = root / "tmp_video_inter_new_scenario.json"

    scenario = load_json(scenario_path)
    time_overrides = dict(scenario.get("time_overrides", {}))
    time_overrides["frame_num"] = args.frame_num
    time_overrides["frame_dt"] = args.frame_dt
    scenario["time_overrides"] = time_overrides
    preview_duration = args.frame_num * args.frame_dt
    scenario["interactions"] = rescale_interaction_times(
        list(scenario.get("interactions", [])),
        preview_duration,
        preserve_displacement=not args.no_preserve_displacement,
        forward_only=args.forward_only,
    )
    save_json(temp_scenario, scenario)

    if temp_output.exists():
        shutil.rmtree(temp_output)

    render_cmd = [
        args.python_bin,
        args.entry_script,
        "--scenario",
        str(temp_scenario),
        "--output_path",
        str(temp_output),
        "--render_img",
        "--white_bg",
    ]
    subprocess.run(render_cmd, check=True)

    archive_dir = next_version_dir(archive_root)
    archive_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(temp_output.glob("*.png"))
    if len(frames) < args.frame_num:
        raise RuntimeError(
            f"Expected at least {args.frame_num} frames, but only found {len(frames)} in {temp_output}"
        )

    for frame_path in frames[: args.frame_num]:
        shutil.copy2(frame_path, archive_dir / frame_path.name)

    print(f"Temporary render: {temp_output}")
    print(f"Archived preview: {archive_dir}")


if __name__ == "__main__":
    main()
