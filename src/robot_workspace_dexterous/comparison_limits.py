"""Offline common-range comparison for six-axis arms."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def write_half_turn_arm(source, destination):
    """Write a kinematic comparison copy and return the affected arm joints."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Comparison output must differ from the source URDF")
    tree = ET.parse(source)
    joints = []
    for joint in tree.findall("joint"):
        if joint.get("type") not in {"continuous", "revolute"} or joint.find("mimic") is not None:
            continue
        limit = joint.find("limit")
        if joint.get("type") == "revolute" and limit is not None:
            if float(limit.attrib["lower"]) == float(limit.attrib["upper"]):
                continue
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        joint.set("type", "revolute")
        limit.set("lower", str(-math.pi))
        limit.set("upper", str(math.pi))
        joints.append(joint.attrib["name"])
    if len(joints) != 6:
        raise ValueError("The common ±180° comparison requires six independent rotary joints")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return joints
