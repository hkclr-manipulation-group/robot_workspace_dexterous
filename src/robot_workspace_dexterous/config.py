from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from .sampling import generate_uniform_quaternions


@dataclass(frozen=True)
class Config:
    urdf_path: Path | None
    curobo_config_path: Path | None
    base_link: str
    ee_links: tuple[str, ...]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    heights: np.ndarray
    resolution: float
    orientations: np.ndarray
    ik_seeds: int
    batch_size: int
    position_tolerance: float
    orientation_tolerance: float
    self_collision: bool
    sphere_density: float
    collision_matrix_samples: int
    prune_collision_matrix: bool
    minimum_dexterity: float


def _local_path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    result = Path(value)
    if not result.is_absolute():
        result = config_path.parent / result
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    if result.stat().st_size == 0:
        raise ValueError(f"robot model/configuration file is empty: {result}")
    return result


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    robot, grid = raw["robot"], raw["grid"]
    if (robot.get("urdf") is None) == (robot.get("curobo_config") is None):
        raise ValueError("robot must define exactly one of urdf or curobo_config")
    urdf_path = _local_path(config_path, robot.get("urdf"))
    curobo_path = _local_path(config_path, robot.get("curobo_config"))
    x_range = tuple(float(v) for v in grid["x_range"])
    y_range = tuple(float(v) for v in grid["y_range"])
    if len(x_range) != 2 or x_range[0] >= x_range[1]:
        raise ValueError("grid.x_range must be [min, max]")
    if len(y_range) != 2 or y_range[0] >= y_range[1]:
        raise ValueError("grid.y_range must be [min, max]")
    orientation = raw.get("orientations", {})
    if "samples_wxyz" in orientation:
        orientations = np.asarray(orientation["samples_wxyz"], dtype=np.float32)
    else:
        orientations = generate_uniform_quaternions(int(orientation.get("count", 32)))
    if orientations.ndim != 2 or orientations.shape[1] != 4 or not len(orientations):
        raise ValueError("orientations.samples_wxyz must have shape (N, 4)")
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    if not np.isfinite(orientations).all() or np.any(norms <= 1e-8):
        raise ValueError("orientation quaternions must be finite and non-zero")
    orientations = orientations / norms
    resolution = float(grid["resolution"])
    z_min = float(grid.get("z_min", grid.get("height", 0.3)))
    z_max = float(grid.get("z_max", z_min))
    z_step = float(grid.get("z_step", resolution))
    minimum = float(raw.get("output", {}).get("minimum_dexterity", 1.0))
    if resolution <= 0 or z_step <= 0 or z_max < z_min or not 0 <= minimum <= 1:
        raise ValueError("grid steps must be positive and minimum_dexterity in [0, 1]")
    solver = raw.get("solver", {})
    density = float(solver.get("sphere_density", 1.0))
    collision_samples = int(solver.get("collision_matrix_samples", 1000))
    ee_links = tuple(str(link) for link in robot["ee_links"])
    if density <= 0 or collision_samples < 1 or not ee_links:
        raise ValueError("collision settings must be positive and ee_links non-empty")
    return Config(
        urdf_path, curobo_path, str(robot["base_link"]), ee_links,
        x_range, y_range,
        np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=np.float32),
        resolution, orientations,
        int(solver.get("ik_seeds", 8)), int(solver.get("batch_size", 256)),
        float(solver.get("position_tolerance", 0.005)),
        float(solver.get("orientation_tolerance", 0.08)),
        bool(solver.get("self_collision", True)), density, collision_samples,
        bool(solver.get("prune_collision_matrix", True)), minimum,
    )
