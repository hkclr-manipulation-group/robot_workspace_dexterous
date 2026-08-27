import numpy as np
import pytest

from robot_workspace_dexterous.sampling import (
    DexterousWorkspace,
    generate_uniform_quaternions,
    regular_grid,
)


def test_quaternions_are_deterministic_unit_wxyz() -> None:
    first = generate_uniform_quaternions(32)
    second = generate_uniform_quaternions(32)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert first.shape == (32, 4)


def test_regular_grid_has_all_layers() -> None:
    points = regular_grid((0, 0.1), (0, 0.1), np.array([0.2, 0.3]), 0.1)
    assert points.shape == (8, 3)
    np.testing.assert_array_equal(np.unique(points[:, 2]), [0.2, 0.3])


def test_threshold_preserves_counts() -> None:
    workspace = DexterousWorkspace(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        np.array([0.25, 0.75]), np.array([1, 3]), 4,
    )
    filtered = workspace.threshold(0.5)
    np.testing.assert_array_equal(filtered.positions, [[1, 0, 0]])
    np.testing.assert_array_equal(filtered.reachable_orientations, [3])
    with pytest.raises(ValueError):
        workspace.threshold(1.1)


def test_dws_rws_volume_ratio() -> None:
    workspace = DexterousWorkspace(
        np.zeros((4, 3)), np.array([0.0, 0.25, 0.5, 1.0]),
        np.array([0, 1, 2, 4]), 4,
    )
    summary = workspace.volume_summary(0.5, 0.001)
    assert summary["rws_cells"] == 3
    assert summary["dws_cells"] == 2
    assert summary["rws_volume_m3"] == pytest.approx(0.003)
    assert summary["dws_rws_ratio"] == pytest.approx(2 / 3)
