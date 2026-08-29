from pathlib import Path

import yaml

from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.curobo_solver import _load_collision_spheres
from robot_workspace_dexterous.curobo_solver import _normalized_urdf_for_curobo


def test_load_config_uses_precomputed_collision_spheres(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>', encoding="utf-8")
    spheres = tmp_path / "spheres.yaml"
    spheres.write_text(
        "collision_spheres:\n  base:\n    - center: [0, 0, 0]\n      radius: 0.1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "urdf": "robot.urdf",
                    "collision_spheres": "spheres.yaml",
                    "base_link": "base",
                    "ee_links": ["base"],
                    "self_collision_ignore": {},
                },
                "grid": {
                    "x_range": [0, 1],
                    "y_range": [0, 1],
                    "z_min": 0,
                    "z_max": 0,
                    "resolution": 0.1,
                    "z_step": 0.1,
                },
                "orientations": {"count": 4},
                "solver": {"self_collision": True},
                "output": {"minimum_dexterity": 1.0},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.collision_spheres_path == spheres
    assert config.base_link == "base"
    assert not hasattr(config, "sphere_density")
    assert set(_load_collision_spheres(str(spheres))) == {"base"}


def test_normalizes_legacy_dual_v2_2_base_mesh_without_changing_source(tmp_path: Path) -> None:
    robot = tmp_path / "dual_v2_2"
    urdf = robot / "share" / "no_gripper" / "dual_arm.urdf"
    mesh = robot / "share" / "meshes" / "base" / "DZ.STL"
    urdf.parent.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid mesh\nendsolid mesh\n")
    original = (
        '<robot name="test"><link name="base"><visual><geometry>'
        '<mesh filename="../base/DP.stl"/>'
        '</geometry></visual></link></robot>'
    )
    urdf.write_text(original, encoding="utf-8")
    output = tmp_path / "normalized.urdf"

    result = _normalized_urdf_for_curobo(str(urdf), str(output))

    assert Path(result) == output.resolve()
    assert "DZ.STL" in output.read_text(encoding="utf-8")
    assert urdf.read_text(encoding="utf-8") == original
