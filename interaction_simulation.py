import sys

sys.path.append("gaussian-splatting")

import argparse
import os
import cv2
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration

from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

from particle_filling.filling import *

from utils.decode_param import decode_param_json, set_boundary_conditions
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.interaction_utils import (
    apply_camera_orbit,
    compile_interactions,
    load_json,
    merge_dict,
    resolve_text_interaction,
)

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=4.0)


class PipelineParamsNoparse:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def clone_params(params):
    return deepcopy(params)


def compute_area_mask(position_tensor, boundary):
    if boundary is None:
        return torch.ones(position_tensor.shape[0], dtype=torch.bool, device="cuda")
    mask = torch.ones(position_tensor.shape[0], dtype=torch.bool, device="cuda")
    for axis in range(3):
        mask = torch.logical_and(mask, position_tensor[:, axis] > boundary[2 * axis])
        mask = torch.logical_and(mask, position_tensor[:, axis] < boundary[2 * axis + 1])
    return mask


def parse_object_placement(object_spec):
    placement = object_spec.get("placement", {})
    degrees = torch.tensor(
        placement.get("rotation_degree", []), dtype=torch.float32, device="cuda"
    )
    axes = placement.get("rotation_axis", [])
    translation = torch.tensor(
        placement.get("translation", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device="cuda",
    ).reshape(1, 3)
    rotation_matrices = generate_rotation_matrices(degrees, axes)
    return rotation_matrices, translation


def apply_object_placement(position_tensor, cov_tensor, placement_rotations, placement_translation):
    position_tensor = apply_rotations(position_tensor, placement_rotations)
    position_tensor = position_tensor + placement_translation
    cov_tensor = apply_cov_rotations(cov_tensor, placement_rotations)
    return position_tensor, cov_tensor


def invert_object_placement(position_tensor, cov_tensor, placement_rotations, placement_translation):
    position_tensor = position_tensor - placement_translation
    position_tensor = apply_inverse_rotations(position_tensor, placement_rotations)
    cov_tensor = apply_inverse_cov_rotations(cov_tensor, placement_rotations)
    return position_tensor, cov_tensor


def transform_object_to_world(position_tensor, cov_tensor, object_state, keep_placement=True):
    if not keep_placement:
        position_tensor, cov_tensor = invert_object_placement(
            position_tensor,
            cov_tensor,
            object_state["placement_rotations"],
            object_state["placement_translation"],
        )
    position_tensor = apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(position_tensor),
            object_state["scale_origin"],
            object_state["original_mean_pos"],
        ),
        object_state["rotation_matrices"],
    )
    cov_tensor = cov_tensor / (
        object_state["scale_origin"] * object_state["scale_origin"]
    )
    cov_tensor = apply_inverse_cov_rotations(cov_tensor, object_state["rotation_matrices"])
    return position_tensor, cov_tensor


def map_mpm_point_to_world(point, object_state, keep_placement=True):
    point_tensor = torch.tensor(point, dtype=torch.float32, device="cuda").reshape(1, 3)
    cov_tensor = torch.zeros((1, 6), dtype=torch.float32, device="cuda")
    world_point, _ = transform_object_to_world(
        point_tensor, cov_tensor, object_state, keep_placement=keep_placement
    )
    return np.squeeze(world_point.detach().cpu().numpy(), 0)


def load_dynamic_object(object_spec, pipeline, debug=False):
    config_path = object_spec["config"]
    model_path = object_spec["model_path"]
    (
        material_params,
        _bc_params,
        _time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(config_path)

    gaussians = load_checkpoint(model_path)
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    opacity_mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[opacity_mask, :]
    init_cov = init_cov[opacity_mask, :]
    init_opacity = init_opacity[opacity_mask, :]
    init_shs = init_shs[opacity_mask, :]

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"], device="cuda"),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    unselected_pos = None
    unselected_cov = None
    unselected_opacity = None
    unselected_shs = None
    sim_area = preprocessing_params["sim_area"]
    selection_mask = compute_area_mask(rotated_pos, sim_area)
    if sim_area is not None:
        unselected_pos = init_pos[~selection_mask, :]
        unselected_cov = init_cov[~selection_mask, :]
        unselected_opacity = init_opacity[~selection_mask, :]
        unselected_shs = init_shs[~selection_mask, :]

    rotated_pos = rotated_pos[selection_mask, :]
    init_cov = init_cov[selection_mask, :]
    init_opacity = init_opacity[selection_mask, :]
    init_shs = init_shs[selection_mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(
        rotated_pos, torch.tensor(preprocessing_params["scale"], device="cuda")
    )
    transformed_pos = shift2center111(transformed_pos)
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    placement_rotations, placement_translation = parse_object_placement(object_spec)
    transformed_pos, init_cov = apply_object_placement(
        transformed_pos,
        init_cov,
        placement_rotations,
        placement_translation,
    )

    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]
    if filling_params is not None:
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    if filling_params is not None and filling_params["visualize"] is True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    min_pos = torch.min(mpm_init_pos[:gs_num], dim=0)[0]
    max_pos = torch.max(mpm_init_pos[:gs_num], dim=0)[0]
    center = ((min_pos + max_pos) * 0.5).detach().cpu().tolist()
    size = ((max_pos - min_pos) * 0.5 + 1e-3).detach().cpu().tolist()

    return {
        "id": object_spec["id"],
        "model_path": model_path,
        "gaussians": gaussians,
        "material_params": material_params,
        "camera_params": camera_params,
        "rotation_matrices": rotation_matrices,
        "scale_origin": scale_origin,
        "original_mean_pos": original_mean_pos,
        "placement_rotations": placement_rotations,
        "placement_translation": placement_translation,
        "mpm_init_pos": mpm_init_pos,
        "mpm_init_cov": mpm_init_cov,
        "opacity": opacity,
        "shs": shs,
        "gs_num": gs_num,
        "center": center,
        "size": size,
        "unselected_pos": unselected_pos,
        "unselected_cov": unselected_cov,
        "unselected_opacity": unselected_opacity,
        "unselected_shs": unselected_shs,
        "selection_sim_area": sim_area,
        "debug": debug,
    }


def load_background_scene(background_spec, pipeline, object_specs):
    if background_spec is None:
        return None

    model_path = background_spec["model_path"]
    config_path = background_spec.get("config", object_specs[0]["config"])
    (
        _material_params,
        _bc_params,
        _time_params,
        preprocessing_params,
        _camera_params,
    ) = decode_param_json(config_path)
    gaussians = load_checkpoint(model_path)
    params = load_params_from_gs(gaussians, pipeline)
    pos = params["pos"]
    cov = params["cov3D_precomp"]
    opacity = params["opacity"]
    shs = params["shs"]

    opacity_mask = opacity[:, 0] > preprocessing_params["opacity_threshold"]
    pos = pos[opacity_mask, :]
    cov = cov[opacity_mask, :]
    opacity = opacity[opacity_mask, :]
    shs = shs[opacity_mask, :]

    if background_spec.get("remove_object_regions", False):
        keep_mask = torch.ones(pos.shape[0], dtype=torch.bool, device="cuda")
        for object_spec in object_specs:
            if object_spec["model_path"] != model_path:
                continue
            (
                _obj_material,
                _obj_bc,
                _obj_time,
                obj_preprocessing,
                _obj_camera,
            ) = decode_param_json(object_spec["config"])
            rotation_matrices = generate_rotation_matrices(
                torch.tensor(obj_preprocessing["rotation_degree"], device="cuda"),
                obj_preprocessing["rotation_axis"],
            )
            rotated_pos = apply_rotations(pos, rotation_matrices)
            exclusion_mask = compute_area_mask(rotated_pos, obj_preprocessing["sim_area"])
            keep_mask = torch.logical_and(keep_mask, ~exclusion_mask)

        pos = pos[keep_mask, :]
        cov = cov[keep_mask, :]
        opacity = opacity[keep_mask, :]
        shs = shs[keep_mask, :]

    return {
        "model_path": model_path,
        "gaussians": gaussians,
        "pos": pos,
        "cov": cov,
        "opacity": opacity,
        "shs": shs,
    }


def build_camera_context(camera_section, camera_params, frame_num, object_states):
    merged_camera_params = merge_dict(camera_params, camera_section.get("overrides", {}))
    orbit_config = camera_section.get("orbit")
    merged_camera_params = apply_camera_orbit(merged_camera_params, orbit_config, frame_num)

    if "center_world_space" in camera_section:
        viewpoint_center_worldspace = np.array(camera_section["center_world_space"], dtype=np.float32)
        vertical_axis = np.array(
            camera_section.get("vertical_upward_axis_world", [0.0, 0.0, 1.0]),
            dtype=np.float32,
        )
        vertical, h1, h2 = generate_local_coord(vertical_axis)
        observant_coordinates = np.column_stack((h1, h2, vertical))
        return merged_camera_params, viewpoint_center_worldspace, observant_coordinates

    reference_object = object_states[0]
    center_mpm = merged_camera_params["mpm_space_viewpoint_center"]
    up_mpm = (
        np.array(center_mpm, dtype=np.float32)
        + np.array(merged_camera_params["mpm_space_vertical_upward_axis"], dtype=np.float32)
    )
    viewpoint_center_worldspace = map_mpm_point_to_world(
        center_mpm, reference_object, keep_placement=False
    )
    worldspace_up = map_mpm_point_to_world(
        up_mpm, reference_object, keep_placement=False
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    vertical, h1, h2 = generate_local_coord(world_space_vertical_axis)
    observant_coordinates = np.column_stack((h1, h2, vertical))
    return merged_camera_params, viewpoint_center_worldspace, observant_coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--interaction_text", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    scenario = load_json(args.scenario)
    base_config = scenario["base_config"]
    (
        material_params,
        base_bc_params,
        time_params,
        _preprocessing_params,
        camera_params,
    ) = decode_param_json(base_config)

    material_params = merge_dict(material_params, scenario.get("simulation_overrides", {}))
    time_params = merge_dict(time_params, scenario.get("time_overrides", {}))

    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    pipeline.debug = args.debug
    background_color = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    object_specs = []
    default_model_path = scenario.get("default_model_path")
    for object_spec in scenario["objects"]:
        current_spec = clone_params(object_spec)
        if "model_path" not in current_spec:
            current_spec["model_path"] = default_model_path
        object_specs.append(current_spec)

    object_states = [load_dynamic_object(spec, pipeline, debug=args.debug) for spec in object_specs]
    if len(object_states) == 0:
        raise ValueError("At least one object is required in the interaction scenario.")

    sh_degree_signature = (
        object_states[0]["gaussians"].active_sh_degree,
        object_states[0]["gaussians"].max_sh_degree,
    )
    for object_state in object_states[1:]:
        current_signature = (
            object_state["gaussians"].active_sh_degree,
            object_state["gaussians"].max_sh_degree,
        )
        if current_signature != sh_degree_signature:
            raise ValueError("All objects must use the same SH degree for merged rendering.")

    if "additional_material_params" not in material_params:
        material_params["additional_material_params"] = []
    object_meta = {}
    for object_state, object_spec in zip(object_states, object_specs):
        object_meta[object_state["id"]] = {
            "center": object_state["center"],
            "size": object_state["size"],
        }
        material_override = object_spec.get("material_override")
        if material_override is not None:
            additional_params = {
                "point": object_state["center"],
                "size": object_state["size"],
                "E": material_override.get("E", material_params["E"]),
                "nu": material_override.get("nu", material_params["nu"]),
                "density": material_override.get("density", material_params["density"]),
            }
            material_params["additional_material_params"].append(additional_params)

    language_result = {
        "interactions": [],
        "material_overrides": {},
        "camera_overrides": {},
    }
    interaction_text = args.interaction_text or scenario.get("interaction_text")
    if interaction_text:
        language_result = resolve_text_interaction(
            interaction_text,
            list(object_meta.keys()),
            time_params["frame_num"],
            time_params["frame_dt"],
            scenario,
            scenario.get("language_interface"),
        )

    material_params = merge_dict(material_params, language_result["material_overrides"])
    scenario_camera_section = merge_dict(
        scenario.get("camera", {}),
        language_result["camera_overrides"],
    )
    interactions = list(scenario.get("interactions", [])) + list(language_result["interactions"])
    compiled_bcs = compile_interactions(interactions, object_meta)
    boundary_conditions = list(base_bc_params) + compiled_bcs

    background_scene = load_background_scene(
        scenario.get("background"),
        pipeline,
        object_specs,
    )
    if background_scene is None and len(object_states) == 1 and object_states[0]["unselected_pos"] is not None:
        object_state = object_states[0]
        background_scene = {
            "model_path": object_state["model_path"],
            "gaussians": object_state["gaussians"],
            "pos": object_state["unselected_pos"],
            "cov": object_state["unselected_cov"],
            "opacity": object_state["unselected_opacity"],
            "shs": object_state["unselected_shs"],
        }

    all_positions = torch.cat([state["mpm_init_pos"] for state in object_states], dim=0)
    all_covariances = torch.cat([state["mpm_init_cov"] for state in object_states], dim=0)
    all_volumes = get_particle_volume(
        all_positions,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device="cuda:0")

    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        all_positions,
        all_volumes,
        all_covariances,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    set_boundary_conditions(mpm_solver, boundary_conditions, time_params)
    mpm_solver.finalize_mu_lam()

    camera_source_model_path = scenario.get("camera_model_path")
    if camera_source_model_path is None:
        if background_scene is not None:
            camera_source_model_path = background_scene["model_path"]
        else:
            camera_source_model_path = object_states[0]["model_path"]

    (
        merged_camera_params,
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = build_camera_context(
        scenario_camera_section,
        camera_params,
        time_params["frame_num"],
        object_states,
    )

    frame_dt = time_params["frame_dt"]
    substep_dt = time_params["substep_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    if step_per_frame <= 0:
        raise ValueError("frame_dt must be greater than or equal to substep_dt.")

    render_reference = object_states[0]["gaussians"]
    object_slices = []
    offset = 0
    total_render_gaussians = 0
    for object_state in object_states:
        object_slices.append((offset, offset + object_state["gs_num"], object_state))
        offset += object_state["mpm_init_pos"].shape[0]
        total_render_gaussians += object_state["gs_num"]

    background_count = 0 if background_scene is None else background_scene["pos"].shape[0]
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            camera_source_model_path,
            default_camera_index=merged_camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=merged_camera_params["show_hint"],
            init_azimuthm=merged_camera_params["init_azimuthm"],
            init_elevation=merged_camera_params["init_elevation"],
            init_radius=merged_camera_params["init_radius"],
            move_camera=merged_camera_params["move_camera"],
            current_frame=frame,
            delta_a=merged_camera_params["delta_a"],
            delta_e=merged_camera_params["delta_e"],
            delta_r=merged_camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera,
            render_reference,
            pipeline,
            background_color,
        )

        for _step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device="cuda:0")

        if not args.render_img:
            continue

        solver_pos = mpm_solver.export_particle_x_to_torch().to("cuda:0")
        solver_cov = mpm_solver.export_particle_cov_to_torch().view(-1, 6).to("cuda:0")
        solver_rot = mpm_solver.export_particle_R_to_torch().view(-1, 3, 3).to("cuda:0")

        render_positions = []
        render_covariances = []
        render_rotations = []
        render_opacity = []
        render_shs = []

        for start, end, object_state in object_slices:
            object_pos = solver_pos[start:end][: object_state["gs_num"]]
            object_cov = solver_cov[start:end][: object_state["gs_num"]]
            object_rot = solver_rot[start:end][: object_state["gs_num"]]
            object_pos, object_cov = transform_object_to_world(
                object_pos, object_cov, object_state
            )
            render_positions.append(object_pos)
            render_covariances.append(object_cov)
            render_rotations.append(object_rot)
            render_opacity.append(object_state["opacity"])
            render_shs.append(object_state["shs"])

        if background_scene is not None and background_count > 0:
            render_positions.append(background_scene["pos"])
            render_covariances.append(background_scene["cov"])
            render_rotations.append(
                torch.eye(3, device="cuda").unsqueeze(0).repeat(background_count, 1, 1)
            )
            render_opacity.append(background_scene["opacity"])
            render_shs.append(background_scene["shs"])

        pos = torch.cat(render_positions, dim=0)
        cov3d = torch.cat(render_covariances, dim=0)
        rot = torch.cat(render_rotations, dim=0)
        opacity = torch.cat(render_opacity, dim=0)
        shs = torch.cat(render_shs, dim=0)

        screen_points = torch.zeros_like(pos, device="cuda")
        colors_precomp = convert_SH(shs, current_camera, render_reference, pos, rot)
        rendering, _radii = rasterize(
            means3D=pos,
            means2D=screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3d,
        )
        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        if height is None or width is None:
            height = cv2_img.shape[0] // 2 * 2
            width = cv2_img.shape[1] // 2 * 2
        cv2.imwrite(
            os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
            255 * cv2_img,
        )

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )


if __name__ == "__main__":
    main()
