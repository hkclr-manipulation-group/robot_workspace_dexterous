from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .sampling import generate_uniform_quaternions


@dataclass(frozen=True)
class Config:
    robot_config: Path
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
    minimum_dexterity: float


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    robot, grid, orientation = raw["robot"], raw["grid"], raw.get("orientations", {})
    robot_config = Path(robot["curobo_config"])
    if not robot_config.is_absolute():
        robot_config = (config_path.parent / robot_config).resolve()
    if not robot_config.is_file():
        raise FileNotFoundError(robot_config)
    x_range = tuple(float(v) for v in grid["x_range"])
    y_range = tuple(float(v) for v in grid["y_range"])
    if len(x_range) != 2 or x_range[0] >= x_range[1]:
        raise ValueError("grid.x_range must be [min, max]")
    if len(y_range) != 2 or y_range[0] >= y_range[1]:
        raise ValueError("grid.y_range must be [min, max]")
    if "samples_wxyz" in orientation:
        orientations = np.asarray(orientation["samples_wxyz"], dtype=np.float32)
    else:
        orientations = generate_uniform_quaternions(int(orientation.get("count", 32)))
    if orientations.ndim != 2 or orientations.shape[1] != 4 or len(orientations) == 0:
        raise ValueError("orientations.samples_wxyz must have shape (N, 4)")
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    if not np.isfinite(orientations).all() or np.any(norms <= 1e-8):
        raise ValueError("orientation quaternions must be finite and non-zero")
    orientations = orientations / norms
    z_min = float(grid.get("z_min", grid.get("height", 0.3)))
    z_max = float(grid.get("z_max", z_min))
    z_step = float(grid.get("z_step", grid["resolution"]))
    resolution = float(grid["resolution"])
    minimum = float(raw.get("output", {}).get("minimum_dexterity", 1.0))
    if resolution <= 0 or z_step <= 0 or z_max < z_min or not 0 <= minimum <= 1:
        raise ValueError("grid steps must be positive and minimum_dexterity in [0, 1]")
    solver = raw.get("solver", {})
    return Config(
        robot_config=robot_config,
        base_link=str(robot["base_link"]),
        ee_links=tuple(str(link) for link in robot["ee_links"]),
        x_range=x_range, y_range=y_range,
        heights=np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=np.float32),
        resolution=resolution, orientations=orientations,
        ik_seeds=int(solver.get("ik_seeds", 8)),
        batch_size=int(solver.get("batch_size", 256)),
        position_tolerance=float(solver.get("position_tolerance", 0.005)),
        orientation_tolerance=float(solver.get("orientation_tolerance", 0.08)),
        self_collision=bool(solver.get("self_collision", True)),
        minimum_dexterity=minimum,
    )
