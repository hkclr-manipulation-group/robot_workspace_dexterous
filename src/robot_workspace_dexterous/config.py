from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from .sampling import generate_uniform_quaternions


@dataclass(frozen=True)
class Config:
    urdf_path: Path
    collision_spheres_path: Path
    base_link: str
    ee_links: tuple[str, ...]
    self_collision_ignore: dict[str, list[str]]
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
    plot_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    plot_sections: tuple[float, float, float]
    base_position: tuple[float, float, float]


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
    urdf_path = _local_path(config_path, robot.get("urdf"))
    collision_spheres_path = _local_path(config_path, robot.get("collision_spheres"))
    if urdf_path is None or collision_spheres_path is None:
        raise ValueError("robot.urdf and robot.collision_spheres are required")
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
    plot = raw.get("plot", {})
    plot_x_range = tuple(float(v) for v in plot.get("x_range", x_range))
    plot_y_range = tuple(float(v) for v in plot.get("y_range", y_range))
    plot_z_range = tuple(float(v) for v in plot.get("z_range", (z_min, z_max)))
    plot_sections = tuple(float(v) for v in plot.get("sections_xyz", (0.0, 0.0, 0.0)))
    base_position = tuple(float(v) for v in plot.get("base_position", (0.0, 0.0, 0.0)))
    for name, limits in (
        ("plot.x_range", plot_x_range),
        ("plot.y_range", plot_y_range),
        ("plot.z_range", plot_z_range),
    ):
        if len(limits) != 2 or limits[0] >= limits[1]:
            raise ValueError(f"{name} must be [min, max]")
    if len(plot_sections) != 3 or len(base_position) != 3:
        raise ValueError("plot.sections_xyz and plot.base_position must contain three values")
    ee_links = tuple(str(link) for link in robot["ee_links"])
    ignore_raw: dict[str, list[str]] = {}
    ignore_file = _local_path(config_path, robot.get("self_collision_ignore_file"))
    if ignore_file is not None:
        with ignore_file.open("r", encoding="utf-8") as stream:
            ignore_data = yaml.safe_load(stream)
        external = (
            ignore_data.get("self_collision_ignore")
            if isinstance(ignore_data, dict) else None
        )
        if not isinstance(external, dict):
            raise ValueError(
                f"self-collision ignore file is invalid: {ignore_file}"
            )
        ignore_raw.update(external)
    inline_ignore = robot.get("self_collision_ignore", {})
    if not isinstance(inline_ignore, dict):
        raise ValueError("robot.self_collision_ignore must be a mapping")
    for link, ignored in inline_ignore.items():
        ignore_raw.setdefault(link, [])
        ignore_raw[link] = list(dict.fromkeys([*ignore_raw[link], *ignored]))
    if not isinstance(ignore_raw, dict):
        raise ValueError("robot.self_collision_ignore must be a mapping")
    self_collision_ignore = {
        str(link): [str(other) for other in ignored]
        for link, ignored in ignore_raw.items()
    }
    if not ee_links:
        raise ValueError("robot.ee_links must be non-empty")
    return Config(
        urdf_path, collision_spheres_path, str(robot["base_link"]), ee_links,
        self_collision_ignore,
        x_range, y_range,
        np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=np.float32),
        resolution, orientations,
        int(solver.get("ik_seeds", 8)), int(solver.get("batch_size", 256)),
        float(solver.get("position_tolerance", 0.005)),
        float(solver.get("orientation_tolerance", 0.08)),
        bool(solver.get("self_collision", True)), minimum,
        (plot_x_range, plot_y_range, plot_z_range),
        plot_sections, base_position,
    )
