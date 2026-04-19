import json
import math
import re
import subprocess
from copy import deepcopy


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_dict(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def parse_vector_text(text):
    vector_match = re.search(
        r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
        text,
    )
    if vector_match is None:
        return None
    return [float(vector_match.group(i)) for i in range(1, 4)]


def extract_first_number(text, default_value):
    number_match = re.search(r"([-+]?\d*\.?\d+)", text)
    if number_match is None:
        return default_value
    return float(number_match.group(1))


def _find_first_object_id(text, object_ids):
    for object_id in object_ids:
        if object_id.lower() in text.lower():
            return object_id
    return object_ids[0] if len(object_ids) > 0 else None


def _parse_direction_vector(text):
    direction_map = [
        (["left", "向左", "左边"], [-1.0, 0.0, 0.0]),
        (["right", "向右", "右边"], [1.0, 0.0, 0.0]),
        (["forward", "向前", "前方"], [0.0, 1.0, 0.0]),
        (["backward", "向后", "后方"], [0.0, -1.0, 0.0]),
        (["up", "向上", "抬高"], [0.0, 0.0, 1.0]),
        (["down", "向下", "压下"], [0.0, 0.0, -1.0]),
    ]
    lower_text = text.lower()
    for keywords, vector in direction_map:
        if any(keyword in lower_text for keyword in keywords):
            return vector
    return None


def builtin_text_to_actions(text, object_ids, frame_num, frame_dt):
    actions = []
    material_overrides = {}
    camera_overrides = {}
    lower_text = text.lower()

    if "zero gravity" in lower_text or "无重力" in text:
        material_overrides["g"] = [0.0, 0.0, 0.0]

    gravity_vector = None
    gravity_keywords = ["gravity", "重力"]
    if any(keyword in lower_text or keyword in text for keyword in gravity_keywords):
        gravity_vector = parse_vector_text(text)
    if gravity_vector is not None:
        material_overrides["g"] = gravity_vector

    if any(keyword in text for keyword in ["相机", "镜头", "视角"]) or "camera" in lower_text:
        if any(keyword in text for keyword in ["旋转", "绕拍", "环绕"]) or "orbit" in lower_text:
            degrees = extract_first_number(text, 90.0)
            if "顺时针" in text or "clockwise" in lower_text:
                degrees = -abs(degrees)
            elif "逆时针" in text or "counterclockwise" in lower_text:
                degrees = abs(degrees)
            camera_overrides["orbit"] = {"azimuth_deg": degrees}

    if any(keyword in text for keyword in ["拖动", "拖拽"]) or "drag" in lower_text:
        object_id = _find_first_object_id(text, object_ids)
        direction = _parse_direction_vector(text)
        if object_id is not None and direction is not None:
            speed = 0.8
            duration = 0.6
            speed_match = re.search(r"(速度|speed)\s*[:=]?\s*([-+]?\d*\.?\d+)", text, re.I)
            duration_match = re.search(
                r"(时长|duration)\s*[:=]?\s*([-+]?\d*\.?\d+)", text, re.I
            )
            if speed_match is not None:
                speed = float(speed_match.group(2))
            if duration_match is not None:
                duration = float(duration_match.group(2))
            velocity = [speed * value for value in direction]
            actions.append(
                {
                    "type": "object_translation",
                    "object_id": object_id,
                    "velocity": velocity,
                    "start_time": 0.0,
                    "end_time": duration,
                }
            )

    if "碰撞" in text or "collision" in lower_text:
        mentioned_objects = [
            object_id for object_id in object_ids if object_id.lower() in lower_text
        ]
        if len(mentioned_objects) >= 2:
            speed = 1.0
            duration = min(frame_num * frame_dt * 0.5, 1.0)
            speed_match = re.search(r"(速度|speed)\s*[:=]?\s*([-+]?\d*\.?\d+)", text, re.I)
            if speed_match is not None:
                speed = float(speed_match.group(2))
            actions.append(
                {
                    "type": "object_collision",
                    "object_a": mentioned_objects[0],
                    "object_b": mentioned_objects[1],
                    "speed": speed,
                    "start_time": 0.0,
                    "end_time": duration,
                }
            )

    return {
        "interactions": actions,
        "material_overrides": material_overrides,
        "camera_overrides": camera_overrides,
    }


def command_text_to_actions(command, text, object_ids, scenario):
    payload = {
        "text": text,
        "object_ids": object_ids,
        "scenario": scenario,
    }
    response = subprocess.run(
        command,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=True,
    )
    parsed = json.loads(response.stdout)
    parsed.setdefault("interactions", [])
    parsed.setdefault("material_overrides", {})
    parsed.setdefault("camera_overrides", {})
    return parsed


def resolve_text_interaction(
    text,
    object_ids,
    frame_num,
    frame_dt,
    scenario,
    language_interface=None,
):
    if language_interface is not None and language_interface.get("mode") == "command":
        return command_text_to_actions(language_interface["command"], text, object_ids, scenario)
    return builtin_text_to_actions(text, object_ids, frame_num, frame_dt)


def apply_camera_orbit(camera_params, orbit_config, frame_num):
    if orbit_config is None or frame_num <= 1:
        return camera_params
    updated = deepcopy(camera_params)
    updated["default_camera_index"] = -1
    updated["move_camera"] = True
    updated["delta_a"] = orbit_config.get("azimuth_deg", 0.0) / float(frame_num - 1)
    updated["delta_e"] = orbit_config.get("elevation_deg", 0.0) / float(frame_num - 1)
    updated["delta_r"] = orbit_config.get("radius_delta", 0.0) / float(frame_num - 1)
    return updated


def _normalize_vector(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-8:
        return [0.0, 0.0, 0.0]
    return [value / length for value in vector]


def compile_interactions(interactions, object_meta):
    boundary_conditions = []
    for interaction in interactions:
        interaction_type = interaction["type"]
        if interaction_type == "object_translation":
            meta = object_meta[interaction["object_id"]]
            boundary_conditions.append(
                {
                    "type": "enforce_particle_translation",
                    "point": interaction.get("point", meta["center"]),
                    "size": interaction.get("size", meta["size"]),
                    "velocity": interaction["velocity"],
                    "start_time": interaction.get("start_time", 0.0),
                    "end_time": interaction["end_time"],
                }
            )
        elif interaction_type == "object_impulse":
            meta = object_meta[interaction["object_id"]]
            boundary_conditions.append(
                {
                    "type": "particle_impulse",
                    "point": interaction.get("point", meta["center"]),
                    "size": interaction.get("size", meta["size"]),
                    "force": interaction["force"],
                    "num_dt": interaction.get("num_dt", 1),
                    "start_time": interaction.get("start_time", 0.0),
                }
            )
        elif interaction_type == "object_collision":
            meta_a = object_meta[interaction["object_a"]]
            meta_b = object_meta[interaction["object_b"]]
            axis = interaction.get("axis")
            if axis is None:
                axis = [
                    meta_b["center"][i] - meta_a["center"][i]
                    for i in range(3)
                ]
            axis = _normalize_vector(axis)
            speed = interaction.get("speed", 1.0)
            velocity_a = [speed * value for value in axis]
            velocity_b = [-speed * value for value in axis]
            boundary_conditions.extend(
                [
                    {
                        "type": "enforce_particle_translation",
                        "point": meta_a["center"],
                        "size": meta_a["size"],
                        "velocity": velocity_a,
                        "start_time": interaction.get("start_time", 0.0),
                        "end_time": interaction["end_time"],
                    },
                    {
                        "type": "enforce_particle_translation",
                        "point": meta_b["center"],
                        "size": meta_b["size"],
                        "velocity": velocity_b,
                        "start_time": interaction.get("start_time", 0.0),
                        "end_time": interaction["end_time"],
                    },
                ]
            )
            rebound_speed = interaction.get("rebound_speed")
            rebound_end_time = interaction.get("rebound_end_time")
            if rebound_speed is not None and rebound_end_time is not None:
                rebound_start_time = interaction.get(
                    "rebound_start_time", interaction["end_time"]
                )
                rebound_velocity_a = [-rebound_speed * value for value in axis]
                rebound_velocity_b = [rebound_speed * value for value in axis]
                boundary_conditions.extend(
                    [
                        {
                            "type": "enforce_particle_translation",
                            "point": meta_a["center"],
                            "size": meta_a["size"],
                            "velocity": rebound_velocity_a,
                            "start_time": rebound_start_time,
                            "end_time": rebound_end_time,
                        },
                        {
                            "type": "enforce_particle_translation",
                            "point": meta_b["center"],
                            "size": meta_b["size"],
                            "velocity": rebound_velocity_b,
                            "start_time": rebound_start_time,
                            "end_time": rebound_end_time,
                        },
                    ]
                )
        elif interaction_type == "mouse_drag":
            meta = object_meta[interaction["object_id"]]
            drag_path = interaction["path"]
            if len(drag_path) < 2:
                continue
            start_time = interaction.get("start_time", 0.0)
            end_time = interaction["end_time"]
            duration = end_time - start_time
            segment_duration = duration / float(len(drag_path) - 1)
            for index in range(len(drag_path) - 1):
                current_point = drag_path[index]
                next_point = drag_path[index + 1]
                velocity = [
                    (next_point[axis] - current_point[axis]) / max(segment_duration, 1e-6)
                    for axis in range(3)
                ]
                boundary_conditions.append(
                    {
                        "type": "enforce_particle_translation",
                        "point": interaction.get("selection_point", meta["center"]),
                        "size": interaction.get("selection_size", meta["size"]),
                        "velocity": velocity,
                        "start_time": start_time + index * segment_duration,
                        "end_time": start_time + (index + 1) * segment_duration,
                    }
                )
        else:
            raise ValueError(f"Unsupported interaction type: {interaction_type}")
    return boundary_conditions
