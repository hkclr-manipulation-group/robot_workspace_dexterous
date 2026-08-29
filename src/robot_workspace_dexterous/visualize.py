from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .sampling import DexterousWorkspace


AxisRanges = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _center_planes(
    sections_xyz: tuple[float, float, float],
) -> tuple[tuple[str, int, tuple[int, int], float, str, str], ...]:
    section_x, section_y, section_z = sections_xyz
    return (
        ("XY", 2, (0, 1), section_z, "X (m)", "Y (m)"),
        ("XZ", 1, (0, 2), section_y, "X (m)", "Z (m)"),
        ("YZ", 0, (1, 2), section_x, "Y (m)", "Z (m)"),
    )


def save_dexterity_center_views(
    workspace: DexterousWorkspace,
    path: str | Path,
    title: str,
    sections_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis_ranges: AxisRanges | None = None,
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Save positive-dexterity XY, XZ, and YZ center sections."""
    points = workspace.positions
    scores = workspace.dexterity
    planes = _center_planes(sections_xyz)
    if axis_ranges is None:
        axis_ranges = tuple(
            (float(points[:, dimension].min()), float(points[:, dimension].max()))
            for dimension in range(3)
        )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    artist = None

    for axis, (name, normal, dimensions, requested, x_label, y_label) in zip(axes, planes):
        available = np.unique(points[:, normal])
        selected = float(available[np.argmin(np.abs(available - requested))])
        in_plane = np.isclose(points[:, normal], selected, atol=1e-6)
        plane_points = points[in_plane]
        plane_scores = scores[in_plane]
        visible = plane_scores > 0
        artist = axis.scatter(
            plane_points[visible, dimensions[0]],
            plane_points[visible, dimensions[1]],
            c=plane_scores[visible],
            s=22,
            cmap="turbo",
            vmin=0,
            vmax=1,
        )
        axis.set_xlim(*axis_ranges[dimensions[0]])
        axis.set_ylim(*axis_ranges[dimensions[1]])
        axis.scatter(
            [base_position[dimensions[0]]], [base_position[dimensions[1]]],
            marker="+", color="black", s=100, label="base",
        )
        axis.set(
            xlabel=x_label,
            ylabel=y_label,
            title=f"{name} section at {'XYZ'[normal]}={selected:.3f} m",
            aspect="equal",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    figure.suptitle(f"{title} — dexterity center sections", fontsize=15)
    figure.colorbar(artist, ax=axes, label="orientation coverage / dexterity", shrink=0.88)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_metric_center_views(
    workspace: DexterousWorkspace,
    values: np.ndarray,
    path: str | Path,
    title: str,
    label: str,
    sections_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis_ranges: AxisRanges | None = None,
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmap: str = "viridis",
    vmin: float = 0.0,
    cap_positive_infinity: bool = False,
) -> Path:
    """Save reachable XY, XZ, and YZ center sections for one metric."""
    points = workspace.positions
    values = np.asarray(values).copy()
    reachable = workspace.reachable_orientations > 0
    finite_reachable = reachable & np.isfinite(values)
    planes = _center_planes(sections_xyz)
    if axis_ranges is None:
        axis_ranges = tuple(
            (float(points[:, dimension].min()), float(points[:, dimension].max()))
            for dimension in range(3)
        )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    vmax = float(np.percentile(values[finite_reachable], 95)) if np.any(finite_reachable) else vmin + 1.0
    vmax = max(vmax, vmin + 1e-12)
    if cap_positive_infinity:
        values[reachable & np.isposinf(values)] = vmax
    valid = reachable & np.isfinite(values)
    artist = None

    for axis, (name, normal, dimensions, requested, x_label, y_label) in zip(axes, planes):
        available = np.unique(points[:, normal])
        selected = float(available[np.argmin(np.abs(available - requested))])
        visible = np.isclose(points[:, normal], selected, atol=1e-6) & valid
        artist = axis.scatter(
            points[visible, dimensions[0]],
            points[visible, dimensions[1]],
            c=values[visible],
            s=22,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xlim(*axis_ranges[dimensions[0]])
        axis.set_ylim(*axis_ranges[dimensions[1]])
        axis.scatter(
            [base_position[dimensions[0]]], [base_position[dimensions[1]]],
            marker="+", color="black", s=100, label="base",
        )
        axis.set(
            xlabel=x_label,
            ylabel=y_label,
            title=f"{name} section at {'XYZ'[normal]}={selected:.3f} m",
            aspect="equal",
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    figure.suptitle(f"{title} — center sections", fontsize=15)
    figure.colorbar(artist, ax=axes, label=label, shrink=0.88)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_top_view(
    workspace: DexterousWorkspace, path: str | Path, title: str, height: float | None = None
) -> Path:
    points = workspace.positions
    scores = workspace.dexterity
    full_limits = (
        (float(points[:, 0].min()), float(points[:, 0].max())),
        (float(points[:, 1].min()), float(points[:, 1].max())),
    )
    if height is not None:
        available = np.unique(points[:, 2])
        selected = available[np.argmin(np.abs(available - height))]
        keep = np.isclose(points[:, 2], selected, atol=1e-6)
        points, scores = points[keep], scores[keep]
        title = f"{title} (z={selected:.3f} m)"
    reachable = scores > 0
    points, scores = points[reachable], scores[reachable]
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 7), constrained_layout=True)
    artist = axes.scatter(points[:, 0], points[:, 1], c=scores, s=18, cmap="turbo", vmin=0, vmax=1)
    axes.scatter([0], [0], marker="+", color="black", s=100, label="base")
    axes.set(xlabel="X (m)", ylabel="Y (m)", title=title, aspect="equal")
    axes.set_xlim(*full_limits[0])
    axes.set_ylim(*full_limits[1])
    axes.grid(alpha=0.25)
    axes.legend()
    figure.colorbar(artist, ax=axes, label="orientation coverage / dexterity")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def save_metric_top_view(
    workspace: DexterousWorkspace, values: np.ndarray, path: str | Path,
    title: str, label: str, height: float | None = None,
) -> Path:
    points = workspace.positions
    values = np.asarray(values)
    reachable = workspace.reachable_orientations > 0
    full_limits = (
        (float(points[:, 0].min()), float(points[:, 0].max())),
        (float(points[:, 1].min()), float(points[:, 1].max())),
    )
    if height is not None:
        available = np.unique(points[:, 2])
        selected = available[np.argmin(np.abs(available - height))]
        keep = np.isclose(points[:, 2], selected, atol=1e-6)
        points, values, reachable = points[keep], values[keep], reachable[keep]
        title = f"{title} (z={selected:.3f} m)"
    finite = np.isfinite(values) & reachable
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 7), constrained_layout=True)
    vmax = float(np.percentile(values[finite], 95)) if np.any(finite) else 1.0
    artist = axes.scatter(
        points[finite, 0], points[finite, 1], c=values[finite], s=18,
        cmap="viridis", vmin=0, vmax=max(vmax, 1e-12),
    )
    axes.scatter([0], [0], marker="+", color="black", s=100)
    axes.set(xlabel="X (m)", ylabel="Y (m)", title=title, aspect="equal")
    axes.set_xlim(*full_limits[0])
    axes.set_ylim(*full_limits[1])
    axes.grid(alpha=0.25)
    figure.colorbar(artist, ax=axes, label=label)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
