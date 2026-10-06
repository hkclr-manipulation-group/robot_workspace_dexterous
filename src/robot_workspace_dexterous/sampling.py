from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DexterousWorkspace:
    """Dexterity values on a regular XYZ task-space grid."""

    positions: np.ndarray
    dexterity: np.ndarray
    reachable_orientations: np.ndarray
    orientation_count: int
    manipulability_mean: np.ndarray | None = None
    manipulability_squared_mean: np.ndarray | None = None
    condition_number_max: np.ndarray | None = None
    minimum_singular_value: np.ndarray | None = None
    orientation_success: np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions)
        scores = np.asarray(self.dexterity)
        counts = np.asarray(self.reachable_orientations)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if scores.shape != (len(positions),) or counts.shape != (len(positions),):
            raise ValueError("dexterity and reachable_orientations must have shape (N,)")
        if self.orientation_count < 1:
            raise ValueError("orientation_count must be positive")
        if self.orientation_success is not None and np.asarray(self.orientation_success).shape != (len(positions), self.orientation_count):
            raise ValueError('orientation_success must have shape (N, orientation_count)')
        for name in (
            "manipulability_mean", "manipulability_squared_mean",
            "condition_number_max", "minimum_singular_value",
        ):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != (len(positions),):
                raise ValueError(f"{name} must have shape (N,)")

    def threshold(self, minimum: float) -> "DexterousWorkspace":
        if not 0.0 <= minimum <= 1.0:
            raise ValueError("minimum dexterity must be in [0, 1]")
        keep = self.dexterity >= minimum
        return DexterousWorkspace(
            self.positions[keep], self.dexterity[keep],
            self.reachable_orientations[keep], self.orientation_count,
            *(
                None if value is None else value[keep]
                for value in (
                    self.manipulability_mean,
                    self.manipulability_squared_mean,
                    self.condition_number_max,
                    self.minimum_singular_value,
                )
            ),
            None if self.orientation_success is None else self.orientation_success[keep],
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "positions": self.positions, "dexterity": self.dexterity,
            "reachable_orientations": self.reachable_orientations,
            "orientation_count": np.asarray(self.orientation_count, dtype=np.int32),
        }
        for name in (
            "manipulability_mean", "manipulability_squared_mean",
            "condition_number_max", "minimum_singular_value",
        ):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        if self.orientation_success is not None:
            arrays['orientation_success_bits'] = np.packbits(self.orientation_success, axis=1, bitorder='little')
        return arrays

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.to_arrays())

    def volume_summary(self, minimum_dexterity: float, voxel_volume: float) -> dict[str, float | int]:
        """Return equal-grid RWS/DWS volumes and their ratio."""
        if not 0.0 <= minimum_dexterity <= 1.0 or voxel_volume <= 0:
            raise ValueError("invalid dexterity threshold or voxel volume")
        rws_cells = int(np.count_nonzero(self.reachable_orientations > 0))
        dws_cells = int(np.count_nonzero(
            (self.reachable_orientations > 0) & (self.dexterity >= minimum_dexterity)
        ))
        return {
            "rws_cells": rws_cells,
            "dws_cells": dws_cells,
            "rws_volume_m3": rws_cells * voxel_volume,
            "dws_volume_m3": dws_cells * voxel_volume,
            "dws_rws_ratio": dws_cells / rws_cells if rws_cells else 0.0,
            "dws_minimum_dexterity": minimum_dexterity,
        }


def shared_orientation_workspace(workspaces) -> DexterousWorkspace:
    """Intersect the same tested orientation IDs; separate arm solutions are used."""
    items = list(workspaces)
    if not items or any(item.orientation_success is None for item in items):
        raise ValueError('Shared orientation results require per-orientation success masks')
    first = items[0]
    if any(item.orientation_count != first.orientation_count or
           not np.array_equal(item.positions, first.positions) for item in items):
        raise ValueError('Shared workspaces require identical grids and orientation ordering')
    mask = np.logical_and.reduce([item.orientation_success for item in items])
    counts = mask.sum(axis=1).astype(np.int32)
    return DexterousWorkspace(first.positions, counts.astype(np.float32)/first.orientation_count,
                             counts, first.orientation_count, orientation_success=mask)


def generate_uniform_quaternions(count: int) -> np.ndarray:
    """Return deterministic, approximately uniform SO(3) samples in wxyz order.

    The low-discrepancy construction is the Shoemake map from three sequences
    on the unit cube. Antipodal quaternions represent the same rotation.
    """
    if count < 1:
        raise ValueError("orientation count must be positive")
    index = np.arange(count, dtype=np.float64) + 0.5
    u1 = index / count
    u2 = np.mod(index * 0.6180339887498949, 1.0)
    u3 = np.mod(index * 0.7548776662466927, 1.0)
    xyzw = np.column_stack(
        (
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        )
    )
    return xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def regular_grid(
    x_range: tuple[float, float], y_range: tuple[float, float],
    heights: np.ndarray, resolution: float,
) -> np.ndarray:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    xs = np.arange(x_range[0], x_range[1] + 0.5 * resolution, resolution)
    ys = np.arange(y_range[0], y_range[1] + 0.5 * resolution, resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.concatenate([
        np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
        for z in heights
    ]).astype(np.float32)
