from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.curobo_solver import (
    _normalized_urdf_for_curobo, _prepare_joint_limits,
)


@pytest.mark.parametrize("name", ["D20260901B", "D20260902B60", "D20260903B10"])
def test_design_normalization_supplies_strict_dynamic_bounds_without_changing_source(tmp_path, name):
    project = Path(__file__).resolve().parents[1]
    config = load_config(project / "configs" / f"{name}.yaml")
    original = config.urdf_path.read_bytes()
    output = tmp_path / "normalized.urdf"
    with pytest.warns(UserWarning, match="Workspace-only"):
        _normalized_urdf_for_curobo(str(config.urdf_path), str(output), config.joint_limit_defaults)
    assert config.urdf_path.read_bytes() == original
    joints = ET.parse(output).findall("joint")
    source_joints = ET.fromstring(original).findall("joint")
    for joint, source in zip(joints, source_joints):
        if joint.get("type") == "fixed":
            continue
        for field in ("velocity", "effort"):
            value = float(joint.find("limit").get(field))
            assert -value < value
        for element in ("origin", "axis", "parent", "child"):
            assert ET.tostring(joint.find(element)) == ET.tostring(source.find(element))
        for field in ("lower", "upper"):
            assert joint.find("limit").get(field) == source.find("limit").get(field)
    # The normalized result also passes without fallback defaults.
    assert _prepare_joint_limits(ET.parse(output).getroot()) == []


@pytest.mark.parametrize("field", ["velocity", "effort"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_dynamic_limits_fail_before_gpu_initialization(field, value):
    root = ET.fromstring('<robot><joint name="J1" type="revolute"><limit lower="-1" upper="1" velocity="2" effort="3"/></joint></robot>')
    root.find("joint/limit").set(field, value)
    with pytest.raises(ValueError, match=f"joint J1: {field}"):
        _prepare_joint_limits(root)


def test_defaults_preserve_existing_positive_limits_and_ignore_fixed_joints():
    root = ET.fromstring('<robot><joint name="fixed" type="fixed"/><joint name="J1" type="revolute"><limit lower="-1" upper="1" velocity="2" effort="3"/></joint></robot>')
    before = ET.tostring(root)
    assert _prepare_joint_limits(root, {"velocity": 1, "effort": 1}) == []
    assert ET.tostring(root) == before
