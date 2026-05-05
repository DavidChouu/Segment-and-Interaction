#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENARIO = "config/toy_car_2_first_frame_stable.json"
DEFAULT_CONFIG_DIR = "generated_configs/toy_car_2_camera_sweep"
DEFAULT_IMAGE_DIR = "camera_sweeps/toy_car_2_first_frame"
DEFAULT_TEMP_ROOT = "tmp_outputs/toy_car_2_camera_sweep"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def build_variants(base_scenario):
    camera = base_scenario["camera"]["overrides"]
    base_azimuth = float(camera["init_azimuthm"])
    base_elevation = float(camera["init_elevation"])
    base_radius = float(camera["init_radius"])

    variants = []

    for delta in [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0]:
        variants.append(
            {
                "kind": "azimuth",
                "value": base_azimuth + delta,
                "azimuth": base_azimuth + delta,
                "elevation": base_elevation,
                "radius": base_radius,
            }
        )

    for delta in [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0]:
        variants.append(
            {
                "kind": "elevation",
                "value": base_elevation + delta,
                "azimuth": base_azimuth,
                "elevation": base_elevation + delta,
                "radius": base_radius,
            }
        )

    for delta in [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
        new_radius = base_radius + delta
        if new_radius <= 0.2:
            continue
        variants.append(
            {
                "kind": "radius",
                "value": new_radius,
                "azimuth": base_azimuth,
                "elevation": base_elevation,
                "radius": new_radius,
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
        description="Sweep camera parameters for first-frame preview generation."
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--temp-root", default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--clean", action="store_true")
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
        "kind",
        "value",
        "init_azimuthm",
        "init_elevation",
        "init_radius",
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
            overrides = scenario_payload["camera"]["overrides"]
            overrides["init_azimuthm"] = variant["azimuth"]
            overrides["init_elevation"] = variant["elevation"]
            overrides["init_radius"] = variant["radius"]

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
                    "kind": variant["kind"],
                    "value": variant["value"],
                    "init_azimuthm": variant["azimuth"],
                    "init_elevation": variant["elevation"],
                    "init_radius": variant["radius"],
                }
            )


if __name__ == "__main__":
    main()
