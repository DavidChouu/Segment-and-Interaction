#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create a shared-layout-normalization trial scenario for multi-object MPM rendering."
    )
    parser.add_argument(
        "--base-scenario",
        default="config/toy_car_2_first_frame_stable.json",
        help="Baseline scenario to clone camera/time settings from.",
    )
    parser.add_argument(
        "--yellow-model-path",
        default="./toy_car_2/toy_car_2/object_yellow_toy_car",
        help="Model path for the manually aligned yellow car.",
    )
    parser.add_argument(
        "--black-model-path",
        default="./toy_car_2/toy_car_2/object_black_toy_car",
        help="Model path for the manually aligned black car.",
    )
    parser.add_argument(
        "--output",
        default="config/toy_car_2_first_frame_shared_layout_trial.json",
        help="Output scenario path.",
    )
    parser.add_argument(
        "--shared-layout-scale",
        type=float,
        default=0.40,
        help="Global scale used when all objects are jointly normalized into MPM space.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    base_scenario = load_json(root / args.base_scenario)

    scenario = json.loads(json.dumps(base_scenario))
    scenario["shared_layout_normalization"] = True
    scenario["shared_layout_scale"] = args.shared_layout_scale
    scenario["render_space"] = "mpm"

    objects = scenario["objects"]
    if len(objects) < 2:
        raise ValueError("The base scenario must contain at least two objects.")

    objects[0]["model_path"] = args.yellow_model_path
    objects[1]["model_path"] = args.black_model_path

    # In shared-layout mode, we rely on the manually aligned Gaussian coordinates,
    # so placements should start from identity and only be reintroduced if needed.
    for obj in objects[:2]:
        obj["placement"] = {
            "rotation_degree": [],
            "rotation_axis": [],
            "translation": [0.0, 0.0, 0.0],
        }

    save_json(root / args.output, scenario)
    print(root / args.output)


if __name__ == "__main__":
    main()
