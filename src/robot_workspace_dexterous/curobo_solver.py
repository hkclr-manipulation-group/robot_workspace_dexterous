from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
import warnings
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .sampling import DexterousWorkspace, regular_grid
from .urdf_compat import align_joint_axes


def _prepare_joint_limits(root: ET.Element, defaults: dict[str, float] | None = None) -> list[str]:
    """Validate movable joints, filling only missing/zero dynamic limits explicitly."""
    defaults = defaults or {}
    if set(defaults) - {"velocity", "effort"}:
        raise ValueError("joint_limit_defaults accepts only velocity and effort")
    for name, value in defaults.items():
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"joint_limit_defaults.{name} must be finite and positive")
    replacements = []
    for joint in root.findall("joint"):
        kind = joint.get("type")
        if kind not in {"revolute", "continuous", "prismatic"}:
            continue
        name = joint.get("name", "<unnamed>")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"joint {name}: missing limit element")
        if kind != "continuous":
            lower = float(limit.get("lower", "nan"))
            upper = float(limit.get("upper", "nan"))
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"joint {name}: finite position lower < upper is required")
        for field in ("velocity", "effort"):
            raw = limit.get(field)
            value = float(raw) if raw is not None else 0.0
            if value == 0 and field in defaults:
                value = float(defaults[field])
                limit.set(field, str(value))
                replacements.append(f"{name}.{field}={value:g}")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"joint {name}: {field}={raw!r} must be finite and positive; "
                    "set the actual URDF limit or explicitly configure "
                    f"robot.joint_limit_defaults.{field} for workspace-only evaluation"
                )
    return replacements


def _normalized_urdf_for_curobo(
    urdf_path: str, output_path: str | None,
    joint_limit_defaults: dict[str, float] | None = None,
) -> str:
    """Create a mesh-free kinematic URDF; collision geometry comes from YAML."""
    source = Path(urdf_path).expanduser().resolve()
    tree = ET.parse(source)
    replacements = _prepare_joint_limits(tree.getroot(), joint_limit_defaults)
    converted_axes = align_joint_axes(tree.getroot())
    changed = bool(replacements or converted_axes)
    if replacements:
        warnings.warn("Workspace-only joint limit defaults applied: " + ", ".join(replacements), stacklevel=2)
    for element in list(tree.getroot()):
        if element.tag not in {"link", "joint"}:
            tree.getroot().remove(element)
            changed = True
    for link in tree.getroot().findall("link"):
        for element in list(link):
            if element.tag in {"visual", "collision"}:
                link.remove(element)
                changed = True
    if not changed and output_path is None:
        return str(source)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path else source.with_name(source.stem + "_curobo.urdf")
    )
    if destination == source:
        raise ValueError("normalized URDF output must differ from the source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return str(destination)


def _load_collision_spheres(path: str) -> dict[str, list[dict[str, object]]]:
    """Load and validate a collision sphere file generated for cuRobo."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    spheres = data.get("collision_spheres") if isinstance(data, dict) else None
    if not isinstance(spheres, dict) or not spheres:
        raise ValueError(f"collision sphere file is empty or invalid: {source}")
    for link, entries in spheres.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"collision sphere list is empty for link {link!r}")
        for entry in entries:
            center = entry.get("center") if isinstance(entry, dict) else None
            radius = entry.get("radius") if isinstance(entry, dict) else None
            try:
                valid_values = (
                    isinstance(center, list)
                    and len(center) == 3
                    and all(np.isfinite(float(value)) for value in center)
                    and np.isfinite(float(radius))
                    and float(radius) > 0
                )
            except (TypeError, ValueError):
                valid_values = False
            if not valid_values:
                raise ValueError(f"invalid collision sphere for link {link!r}: {entry!r}")
    return spheres


def validate_robot_inputs(
    urdf_path: str, collision_spheres_path: str, base_link: str,
    ee_links: tuple[str, ...], self_collision_ignore: dict[str, list[str]],
    joint_limit_defaults: dict[str, float] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Validate portable robot inputs without importing CUDA or cuRobo."""
    root = ET.parse(urdf_path).getroot()
    _prepare_joint_limits(root, joint_limit_defaults)
    align_joint_axes(root)
    urdf_links = {link.get("name") for link in root.findall("link")}
    child_links = {
        child.get("link")
        for child in root.findall("joint/child")
        if child.get("link") is not None
    }
    root_links = urdf_links.difference(child_links)
    if base_link not in root_links:
        raise ValueError(
            f"base_link must be a URDF root; got {base_link!r}, expected one of "
            + ", ".join(sorted(root_links))
        )
    spheres = _load_collision_spheres(collision_spheres_path)
    required_links = set(spheres) | set(ee_links) | {base_link}
    unknown = required_links.difference(urdf_links)
    if unknown:
        raise ValueError(f"links not found in URDF: {', '.join(sorted(unknown))}")
    for link, ignored in self_collision_ignore.items():
        unknown_ignore = ({link} | set(ignored)).difference(urdf_links)
        if unknown_ignore:
            raise ValueError(
                "self-collision ignore contains links not found in URDF: "
                + ", ".join(sorted(unknown_ignore))
            )

    return spheres


def build_collision_robots(
    urdf_path: str,
    collision_spheres_path: str,
    base_link: str,
    ee_links: tuple[str, ...],
    self_collision_ignore: dict[str, list[str]],
    normalized_urdf_path: str | None = None,
    joint_limit_defaults: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build cuRobo models from precomputed spheres without collision fitting."""
    from curobo._src.types.robot import RobotCfg

    resolved_urdf = _normalized_urdf_for_curobo(urdf_path, normalized_urdf_path, joint_limit_defaults)
    spheres = validate_robot_inputs(
        resolved_urdf, collision_spheres_path, base_link, ee_links, self_collision_ignore
    )

    common = {
        "urdf_path": resolved_urdf,
        "asset_root_path": str(Path(resolved_urdf).parent),
        "base_link": base_link,
        "collision_link_names": list(spheres),
        "collision_spheres": spheres,
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": self_collision_ignore,
        "self_collision_buffer": {},
    }
    result: dict[str, object] = {}
    for link in ee_links:
        data = {"robot_cfg": {"kinematics": {**deepcopy(common), "tool_frames": [link]}}}
        result[link] = RobotCfg.create(data, load_collision_spheres=True)
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
            raise ValueError("a prebuilt collision robot is required for self-collision checking")
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
            np.fmax.at(condition_max, successful_points, condition)
            np.fmin.at(sigma_minimum, successful_points, sigma_min)
        if progress:
            progress(stop, total, time.monotonic() - started)
    denominator = np.maximum(counts, 1)
    return DexterousWorkspace(
        positions, counts.astype(np.float32) / orientation_count, counts,
        orientation_count, (w_sum / denominator).astype(np.float32),
        (w2_sum / denominator).astype(np.float32), condition_max.astype(np.float32),
        sigma_minimum.astype(np.float32),
    )
