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
    assert len(config.heights) == 1
    assert config.plot_ranges[2][0] < 0 < config.plot_ranges[2][1]
    assert not hasattr(config, "sphere_density")
    assert set(_load_collision_spheres(str(spheres))) == {"base"}


def test_normalizes_without_mesh_files_and_preserves_kinematics(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    robot = tmp_path / "dual_v2_2"
    urdf = robot / "share" / "no_gripper" / "dual_arm.urdf"
    urdf.parent.mkdir(parents=True)
    original = (
        '<robot name="test"><link name="base"><visual><geometry>'
        '<mesh filename="package://missing/mesh.stl"/>'
        '</geometry></visual><collision><geometry><mesh filename="absent.stl"/>'
        '</geometry></collision><inertial><mass value="2"/></inertial></link>'
        '<link name="tip"/><joint name="j" type="revolute">'
        '<parent link="base"/><child link="tip"/><origin xyz="0 0 0.1"/>'
        '<axis xyz="0 1 0"/><limit lower="-1" upper="2" effort="3" velocity="4"/>'
        '</joint></robot>'
    )
    urdf.write_text(original, encoding="utf-8")
    output = tmp_path / "normalized.urdf"

    result = _normalized_urdf_for_curobo(str(urdf), str(output))

    assert Path(result) == output.resolve()
    root = ET.parse(output).getroot()
    assert root.findall(".//mesh") == []
    assert root.findall("link/visual") == []
    assert root.findall("link/collision") == []
    assert ET.tostring(root.find("joint")) == ET.tostring(ET.fromstring(original).find("joint"))
    assert root.find("link/inertial/mass").get("value") == "2"
    assert urdf.read_text(encoding="utf-8") == original


def test_load_config_merges_external_and_inline_collision_ignores(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>', encoding="utf-8")
    spheres = tmp_path / "spheres.yaml"
    spheres.write_text(
        "collision_spheres:\n  base:\n    - center: [0, 0, 0]\n      radius: 0.1\n",
        encoding="utf-8",
    )
    ignores = tmp_path / "ignores.yaml"
    ignores.write_text("self_collision_ignore:\n  base: [first]\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "robot": {
            "urdf": "robot.urdf", "collision_spheres": "spheres.yaml",
            "base_link": "base", "ee_links": ["base"],
            "self_collision_ignore_file": "ignores.yaml",
            "self_collision_ignore": {"base": ["second"]},
        },
        "grid": {"x_range": [0, 1], "y_range": [0, 1], "z_min": 0,
                 "z_max": 0, "resolution": 0.1, "z_step": 0.1},
    }), encoding="utf-8")

    config = load_config(config_path)

    assert config.self_collision_ignore == {"base": ["first", "second"]}
