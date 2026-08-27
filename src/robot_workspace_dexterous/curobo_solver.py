from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .sampling import DexterousWorkspace, regular_grid


def _normalized_urdf_for_curobo(urdf_path: str, output_path: str | None) -> str:
    """Resolve ROS package mesh URIs for cuRobo's filesystem-only parser."""
    source = Path(urdf_path).expanduser().resolve()
    root = ET.parse(source).getroot()
    changed = False
    for mesh in root.findall(".//mesh"):
        value = mesh.get("filename")
        if not value or not value.startswith("package://"):
            continue
        package_relative = Path(value[len("package://") :])
        # ROS package URIs contain a package name followed by the path within
        # that package. In this repository the package share directory itself
        # is available, not an installed ROS package index.
        suffix = Path(*package_relative.parts[1:])
        resolved = None
        for parent in source.parents:
            candidate = (parent / suffix).resolve()
            if candidate.is_file() and candidate.stat().st_size > 0:
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"cannot resolve URDF mesh URI {value!r} from {source}"
            )
        mesh.set("filename", resolved.as_posix())
        changed = True
    if not changed:
        return str(source)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path else source.with_name(source.stem + "_curobo.urdf")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return str(destination)


def _resolve_model_path(config_path: Path, value: str) -> Path:
    source = Path(value)
    if source.is_absolute() and source.is_file():
        return source
    for root in (config_path.parent, *config_path.parents):
        candidate = (root / source).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve {value!r} referenced by {config_path}")


def build_collision_robots_from_config(
    config_path: str, base_link: str, ee_links: tuple[str, ...]
) -> dict[str, object]:
    from curobo._src.types.robot import RobotCfg

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not data:
        raise ValueError(f"cuRobo configuration is empty: {path}")
    data = deepcopy(data)
    kin = data["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(_resolve_model_path(path, kin["urdf_path"]))
    asset = kin.get("asset_root_path")
    kin["asset_root_path"] = str(Path(kin["urdf_path"]).parent) if not asset else str(
        _resolve_model_path(path, asset)
    )
    if str(kin.get("base_link", base_link)) != base_link:
        raise ValueError("base_link differs from the selected cuRobo configuration")
    for key in (
        "ee_link", "link_names", "usd_path", "usd_robot_root", "isaac_usd_path",
        "usd_flip_joints", "usd_flip_joint_limits",
    ):
        kin.pop(key, None)
    result = {}
    for link in ee_links:
        per_link = deepcopy(data)
        per_link["robot_cfg"]["kinematics"]["tool_frames"] = [link]
        result[link] = RobotCfg.create(per_link, load_collision_spheres=True)
    return result


def build_collision_robots_from_urdf(
    urdf_path: str, base_link: str, ee_links: tuple[str, ...],
    sphere_density: float = 1.0, collision_matrix_samples: int = 1000,
    prune_collision_matrix: bool = True, generated_config_path: str | None = None,
) -> dict[str, object]:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.robot import RobotCfg
    from curobo.robot_builder import RobotBuilder

    normalized_output = (
        str(Path(generated_config_path).with_name("normalized_robot.urdf"))
        if generated_config_path else None
    )
    resolved_urdf = _normalized_urdf_for_curobo(urdf_path, normalized_output)
    builder = RobotBuilder(
        resolved_urdf, str(Path(resolved_urdf).resolve().parent), list(ee_links)
    )
    builder.fit_collision_spheres(
        sphere_density=sphere_density, use_collision_mesh=True, compute_metrics=True
    )
    builder.compute_collision_matrix(
        prune_collisions=prune_collision_matrix, num_samples=collision_matrix_samples
    )
    loader = builder.build()
    if loader.base_link != base_link:
        raise ValueError(f"base_link must be URDF root {loader.base_link!r}")
    if generated_config_path:
        builder.save(loader, generated_config_path)
    result = {}
    for link in ee_links:
        cfg = deepcopy(loader)
        cfg.tool_frames = [link]
        kin = KinematicsCfg.from_config(cfg)
        result[link] = RobotCfg(kinematics=kin, device_cfg=kin.device_cfg)
    return result


def compute_dexterous_workspace(
    urdf_path: str, base_link: str, ee_link: str,
    x_range: tuple[float, float], y_range: tuple[float, float],
    heights: np.ndarray, resolution: float, orientations_wxyz: np.ndarray,
    num_seeds: int = 8, batch_size: int = 256,
    position_tolerance: float = 0.005, orientation_tolerance: float = 0.08,
    self_collision: bool = True,
    progress: Callable[[int, int, float], None] | None = None,
    robot: object | None = None,
) -> DexterousWorkspace:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo workspace computation requires CUDA-enabled PyTorch")
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo._src.types.robot import RobotCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import GoalToolPose, Pose

    if robot is None:
        if self_collision:
            robot = build_collision_robots_from_urdf(
                urdf_path, base_link, (ee_link,)
            )[ee_link]
        else:
            kin = KinematicsCfg.from_basic_urdf(urdf_path, base_link, [ee_link])
            robot = RobotCfg(kinematics=kin, device_cfg=kin.device_cfg)
    solver_cfg = InverseKinematicsCfg.create(
        robot=robot, num_seeds=num_seeds, seed_solver_num_seeds=num_seeds,
        max_batch_size=batch_size, position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        self_collision_check=self_collision, load_collision_spheres=self_collision,
    )
    solver = InverseKinematics(solver_cfg)
    jacobian_model = Kinematics(
        robot.kinematics, compute_jacobian=True, compute_spheres=False
    )
    device, dtype = solver_cfg.device_cfg.device, solver_cfg.device_cfg.dtype
    positions = regular_grid(x_range, y_range, np.asarray(heights), resolution)
    orientations = np.asarray(orientations_wxyz, dtype=np.float32)
    orientation_count = len(orientations)
    counts = np.zeros(len(positions), dtype=np.int32)
    w_sum = np.zeros(len(positions), dtype=np.float64)
    w2_sum = np.zeros(len(positions), dtype=np.float64)
    condition_max = np.full(len(positions), np.nan, dtype=np.float64)
    sigma_minimum = np.full(len(positions), np.nan, dtype=np.float64)
    total = len(positions) * orientation_count
    started = time.monotonic()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = np.arange(start, stop, dtype=np.int64)
        point_index = flat // orientation_count
        orientation_index = flat % orientation_count
        pose = Pose(
            position=torch.as_tensor(positions[point_index], device=device, dtype=dtype),
            quaternion=torch.as_tensor(
                orientations[orientation_index], device=device, dtype=dtype
            ),
        )
        goals = GoalToolPose.from_poses({ee_link: pose}, num_goalset=1)
        solved = solver.solve_pose(goal_tool_poses=goals)
        success = solved.success.reshape(-1)[: len(flat)].detach().cpu().numpy().astype(bool)
        np.add.at(counts, point_index[success], 1)
        if np.any(success):
            state = jacobian_model.compute_kinematics(solved.js_solution)
            jac = state.tool_jacobians.reshape(
                len(flat), -1, 6, state.tool_jacobians.shape[-1]
            )[:, 0]
            singular = torch.linalg.svdvals(jac)[success].detach().cpu().numpy()
            successful_points = point_index[success]
            w = np.prod(singular, axis=1).astype(np.float64)
            sigma_min = singular[:, -1].astype(np.float64)
            condition = np.divide(
                singular[:, 0], sigma_min,
                out=np.full(len(sigma_min), np.inf), where=sigma_min > 1e-9,
            )
            np.add.at(w_sum, successful_points, w)
            np.add.at(w2_sum, successful_points, w * w)
            for index, kappa, sigma in zip(successful_points, condition, sigma_min):
                condition_max[index] = np.fmax(condition_max[index], kappa)
                sigma_minimum[index] = np.fmin(sigma_minimum[index], sigma)
        if progress:
            progress(stop, total, time.monotonic() - started)
    denominator = np.maximum(counts, 1)
    return DexterousWorkspace(
        positions, counts.astype(np.float32) / orientation_count, counts,
        orientation_count, (w_sum / denominator).astype(np.float32),
        (w2_sum / denominator).astype(np.float32), condition_max.astype(np.float32),
        sigma_minimum.astype(np.float32),
    )
