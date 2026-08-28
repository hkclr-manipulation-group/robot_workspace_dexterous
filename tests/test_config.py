from pathlib import Path

import yaml

from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.curobo_solver import _load_collision_spheres


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
