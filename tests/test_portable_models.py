from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.curobo_solver import validate_robot_inputs
from robot_workspace_dexterous.visualize import load_zero_pose_collision_spheres


def test_all_presets_load_after_relocation_without_sibling_repositories(tmp_path):
    project = Path(__file__).resolve().parents[1]
    relocated = tmp_path / "standalone"
    shutil.copytree(project / "models", relocated / "models")
    shutil.copytree(project / "configs", relocated / "configs")
    configs = sorted((relocated / "configs").glob("*.yaml"))
    assert len(configs) >= 11
    for path in configs:
        config = load_config(path)
        for asset in (config.urdf_path, config.collision_spheres_path):
            assert asset.is_relative_to(relocated)
        assert not ET.parse(config.urdf_path).findall(".//mesh")
        spheres = validate_robot_inputs(str(config.urdf_path), str(config.collision_spheres_path),
                                        config.base_link, config.ee_links, config.self_collision_ignore)
        world_spheres = load_zero_pose_collision_spheres(config.urdf_path, config.collision_spheres_path)
        assert len(world_spheres) == sum(map(len, spheres.values()))
