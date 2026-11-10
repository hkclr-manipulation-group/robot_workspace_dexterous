import math
import xml.etree.ElementTree as ET
import pytest

from robot_workspace_dexterous.comparison_limits import write_half_turn_arm


def test_comparison_changes_only_independent_arm_axes(tmp_path):
    source, result = tmp_path / "source.urdf", tmp_path / "comparison.urdf"
    root = ET.Element("robot")
    for name, kind in [(f"J{i}", "revolute") for i in range(6)] + [
            ("jaw", "prismatic"), ("coupled", "revolute"), ("locked", "revolute")]:
        joint = ET.SubElement(root, "joint", name=name, type=kind)
        ET.SubElement(joint, "limit", lower="0", upper="0" if name == "locked" else "1")
        if name == "coupled":
            ET.SubElement(joint, "mimic", joint="J1")
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    assert write_half_turn_arm(source, result) == [f"J{i}" for i in range(6)]
    assert source.read_bytes() == original
    for joint in ET.parse(result).findall("joint"):
        bounds = [float(joint.find("limit").attrib[k]) for k in ("lower", "upper")]
        if joint.get("name").startswith("J"):
            assert bounds == [-math.pi, math.pi]
        else:
            assert bounds == [0, 0 if joint.get("name") == "locked" else 1]
    with pytest.raises(ValueError):
        write_half_turn_arm(source, source)
