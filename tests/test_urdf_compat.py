from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from robot_workspace_dexterous.urdf_compat import align_joint_axes
from robot_workspace_dexterous.diagnostics import forward_kinematics


def test_tilted_design_axis_preserves_every_original_link_pose():
    source = Path(__file__).resolve().parents[1] / "models/D20260903B10/robot.urdf"
    original = ET.parse(source).getroot()
    converted = deepcopy(original)
    assert align_joint_axes(converted) == ["J5"]
    rng = np.random.default_rng(27)
    positions = {j.get("name"): rng.uniform(-3.14, 3.14, 128)
                 for j in original.findall("joint") if j.get("type") != "fixed"}
    before = forward_kinematics(original, positions)
    after = forward_kinematics(converted, positions)
    for name in before:
        np.testing.assert_allclose(after[name], before[name], atol=2e-12)
    for joint in converted.findall("joint"):
        if joint.get("type") != "fixed":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            assert np.count_nonzero(axis) == 1 and np.max(np.abs(axis)) == 1
    # A second normalization must not insert another helper chain.
    before_second_pass = ET.tostring(converted)
    assert align_joint_axes(converted) == []
    assert ET.tostring(converted) == before_second_pass


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_arbitrary_axis_equivalence_with_rotated_parent_origin(kind):
    original = ET.fromstring(f'''<robot><link name="base"/><link name="tip"/>
      <joint name="test" type="{kind}"><parent link="base"/><child link="tip"/>
      <origin xyz="0.2 -0.1 0.4" rpy="0.5 1.5707963267948966 -0.7"/>
      <axis xyz="-0.4 0.8 -0.3"/></joint></robot>''')
    converted = deepcopy(original)
    align_joint_axes(converted)
    positions = {"test": np.linspace(-1, 1, 21)}
    np.testing.assert_allclose(forward_kinematics(converted, positions)["tip"],
                               forward_kinematics(original, positions)["tip"], atol=2e-12)
