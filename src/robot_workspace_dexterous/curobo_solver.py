from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import numpy as np
import yaml

from .sampling import DexterousWorkspace, regular_grid


def compute_dexterous_workspace(
    robot_config: str | Path,
    base_link: str,
    ee_link: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    heights: np.ndarray,
    resolution: float,
    orientations_wxyz: np.ndarray,
    num_seeds: int = 8,
    batch_size: int = 256,
    position_tolerance: float = 0.005,
    orientation_tolerance: float = 0.08,
    self_collision: bool = True,
    progress: Callable[[int, int, float], None] | None = None,
) -> DexterousWorkspace:
    """Evaluate collision-free orientation reachability at every grid point.

    Dexterity is the fraction of sampled SO(3) orientations for which at least
    one cuRobo IK seed converges. This is a discrete orientation-coverage index,
    not a Jacobian manipulability measure.
    """
    if num_seeds < 1 or batch_size < 1:
        raise ValueError("num_seeds and batch_size must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo workspace computation requires CUDA-enabled PyTorch")

    try:
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo._src.types.robot import RobotCfg
        from curobo.kinematics import Kinematics
        from curobo.types import GoalToolPose, Pose
    except ImportError as exc:
        raise RuntimeError(
            "This package requires the cuRobo V2 inverse_kinematics API"
        ) from exc

    config_path = Path(robot_config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        robot_data = yaml.safe_load(stream)
    kinematics = robot_data["robot_cfg"]["kinematics"]
    urdf = Path(kinematics["urdf_path"])
    if not urdf.is_absolute():
        urdf = (config_path.parent / urdf).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    kinematics["urdf_path"] = str(urdf)
    kinematics["asset_root_path"] = str(config_path.parent)
    kinematics["base_link"] = base_link
    kinematics["tool_frames"] = [ee_link]

    robot = RobotCfg.create(robot_data, load_collision_spheres=self_collision)
    solver_cfg = InverseKinematicsCfg.create(
        robot=robot,
        num_seeds=num_seeds,
        seed_solver_num_seeds=num_seeds,
        max_batch_size=batch_size,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        self_collision_check=self_collision,
        load_collision_spheres=self_collision,
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
    manipulability_sum = np.zeros(len(positions), dtype=np.float64)
    manipulability_squared_sum = np.zeros(len(positions), dtype=np.float64)
    condition_number_max = np.full(len(positions), np.nan, dtype=np.float64)
    minimum_singular_value = np.full(len(positions), np.nan, dtype=np.float64)

    # Chunk flattened (position, orientation) queries. A position may span
    # chunks; np.add.at safely accumulates every successful orientation.
    total = len(positions) * orientation_count
    started = time.monotonic()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = np.arange(start, stop, dtype=np.int64)
        position_index = flat // orientation_count
        orientation_index = flat % orientation_count
        position_tensor = torch.as_tensor(
            positions[position_index], device=device, dtype=dtype
        )
        quaternion_tensor = torch.as_tensor(
            orientations[orientation_index], device=device, dtype=dtype
        )
        pose = Pose(position=position_tensor, quaternion=quaternion_tensor)
        goals = GoalToolPose.from_poses({ee_link: pose}, num_goalset=1)
        result = solver.solve_pose(goal_tool_poses=goals)
        success = result.success.reshape(-1)[: len(flat)].detach().cpu().numpy().astype(bool)
        np.add.at(counts, position_index[success], 1)
        if np.any(success):
            joint_state = result.js_solution
            state = jacobian_model.compute_kinematics(joint_state)
            jacobian = state.tool_jacobians.reshape(len(flat), -1, 6, state.tool_jacobians.shape[-1])[:, 0]
            singular_values = torch.linalg.svdvals(jacobian)[success]
            singular_np = singular_values.detach().cpu().numpy().astype(np.float64)
            successful_position_index = position_index[success]
            # Yoshikawa w = sqrt(det(J J^T)) = product of singular values.
            w = np.prod(singular_np, axis=1)
            sigma_min = singular_np[:, -1]
            sigma_max = singular_np[:, 0]
            condition = np.divide(
                sigma_max, sigma_min,
                out=np.full_like(sigma_max, np.inf), where=sigma_min > 1e-9,
            )
            np.add.at(manipulability_sum, successful_position_index, w)
            np.add.at(manipulability_squared_sum, successful_position_index, w * w)
            for point_index, point_condition, point_sigma_min in zip(
                successful_position_index, condition, sigma_min
            ):
                condition_number_max[point_index] = np.fmax(
                    condition_number_max[point_index], point_condition
                )
                minimum_singular_value[point_index] = np.fmin(
                    minimum_singular_value[point_index], point_sigma_min
                )
        if progress is not None:
            progress(stop, total, time.monotonic() - started)

    denominator = np.maximum(counts, 1)
    return DexterousWorkspace(
        positions=positions,
        dexterity=counts.astype(np.float32) / float(orientation_count),
        reachable_orientations=counts,
        orientation_count=orientation_count,
        manipulability_mean=(manipulability_sum / denominator).astype(np.float32),
        manipulability_squared_mean=(manipulability_squared_sum / denominator).astype(np.float32),
        condition_number_max=condition_number_max.astype(np.float32),
        minimum_singular_value=minimum_singular_value.astype(np.float32),
    )
