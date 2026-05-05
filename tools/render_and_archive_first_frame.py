#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Render a first-frame preview and archive it into a gallery folder."
    )
    parser.add_argument(
        "--scenario",
        default="config/toy_car_2_first_frame_stable.json",
        help="Scenario json to render.",
    )
    parser.add_argument(
        "--output-path",
        default="tmp_first_frame_render",
        help="Temporary render output directory. This folder is overwritten; archived images go to --gallery.",
    )
    parser.add_argument(
        "--gallery",
        default="first_frame",
        help="Gallery directory that stores incrementally numbered preview images.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run interaction_simulation.py and helper scripts.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    render_cmd = [
        args.python_bin,
        "interaction_simulation.py",
        "--scenario",
        args.scenario,
        "--output_path",
        args.output_path,
        "--render_img",
        "--white_bg",
    ]
    subprocess.run(render_cmd, check=True)

    source_png = root / args.output_path / "0000.png"
    if not source_png.exists():
        raise FileNotFoundError(f"Rendered first frame not found: {source_png}")

    archive_cmd = [
        args.python_bin,
        "tools/save_preview_frame.py",
        str(source_png),
        args.gallery,
    ]
    subprocess.run(archive_cmd, check=True)
    print(f"Temporary render: {source_png}")
    print(f"Archived gallery: {Path(args.gallery).resolve()}")


if __name__ == "__main__":
    main()
