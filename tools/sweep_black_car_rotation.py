#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENARIO = "config/toy_car_2_first_frame_stable.json"
DEFAULT_CONFIG_DIR = "generated_configs/toy_car_2_black_rotation_sweep"
DEFAULT_IMAGE_DIR = "preview_sweeps/toy_car_2_black_rotation"
DEFAULT_TEMP_ROOT = "tmp_outputs/toy_car_2_black_rotation_sweep"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def build_variants(base_scenario):
    objects = base_scenario["objects"]
    black_car = next(obj for obj in objects if obj["id"] == "black_toy_car")
    placement = black_car["placement"]
    base_degrees = list(placement["rotation_degree"])
    base_axes = list(placement["rotation_axis"])

    offsets = [-9, -6, -3, 0, 3, 6, 9]
    variants = []

    for degree_index, axis in enumerate(base_axes):
        base_degree = float(base_degrees[degree_index])
        sign = -1.0 if base_degree < 0 else 1.0
        magnitude = abs(base_degree)

        for offset in offsets:
            new_magnitude = magnitude + offset
            if new_magnitude <= 0:
                continue

            degrees = list(base_degrees)
            degrees[degree_index] = sign * new_magnitude

            variants.append(
                {
                    "axis_index": degree_index,
                    "axis_value": axis,
                    "offset": offset,
                    "degrees": degrees,
                }
            )

    return variants


def render_variant(scenario_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "interaction_simulation.py",
        "--scenario",
        str(scenario_path),
        "--output_path",
        str(output_dir),
        "--render_img",
        "--white_bg",
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep only the black toy car rotation degrees and render first-frame previews."
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--temp-root", default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old configs, images, temp outputs, and manifest before generating new ones.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render each generated scenario into a first-frame preview.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    scenario_path = root / args.scenario
    config_dir = root / args.config_dir
    image_dir = root / args.image_dir
    temp_root = root / args.temp_root
    manifest_path = image_dir / "manifest.csv"

    base_scenario = load_json(scenario_path)
    variants = build_variants(base_scenario)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]

    if args.clean:
        shutil.rmtree(config_dir, ignore_errors=True)
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)

    config_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_name",
        "config_name",
        "axis_index",
        "axis_value",
        "offset",
        "black_rotation_degree",
        "black_rotation_axis",
    ]

    manifest_exists = manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not manifest_exists:
            writer.writeheader()

        for image_index, variant in enumerate(variants, start=args.start_index):
            image_name = f"{image_index:04d}.png"
            config_name = f"{image_index:04d}.json"

            scenario_payload = json.loads(json.dumps(base_scenario))
            black_car = next(
                obj for obj in scenario_payload["objects"] if obj["id"] == "black_toy_car"
            )
            black_car["placement"]["rotation_degree"] = variant["degrees"]

            scenario_out = config_dir / config_name
            save_json(scenario_out, scenario_payload)

            if args.render:
                temp_output_dir = temp_root / f"{image_index:04d}"
                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
                render_variant(scenario_out, temp_output_dir)

                rendered_frame = temp_output_dir / "0000.png"
                if not rendered_frame.exists():
                    raise FileNotFoundError(f"Expected frame not found: {rendered_frame}")
                shutil.copy2(rendered_frame, image_dir / image_name)

            writer.writerow(
                {
                    "image_name": image_name,
                    "config_name": config_name,
                    "axis_index": variant["axis_index"],
                    "axis_value": variant["axis_value"],
                    "offset": variant["offset"],
                    "black_rotation_degree": json.dumps(variant["degrees"]),
                    "black_rotation_axis": json.dumps(
                        base_scenario["objects"][1]["placement"]["rotation_axis"]
                    ),
                }
            )


if __name__ == "__main__":
    main()
