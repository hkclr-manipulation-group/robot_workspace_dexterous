from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .sampling import DexterousWorkspace


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
