"""Represent arbitrary URDF joint axes using cuRobo's cardinal-axis joints."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
        raise ValueError("URDF origin must contain three finite xyz and rpy values")
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y]); sr, sp, sy = np.sin([r, p, y])
    result = np.eye(4)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    result[:3, 3] = xyz
    return result


def _set_origin(element: ET.Element, matrix: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    r = matrix[:3, :3]
    horizontal = np.hypot(r[0, 0], r[1, 0])
    pitch = np.arctan2(-r[2, 0], horizontal)
    if horizontal > 1e-12:
        roll, yaw = np.arctan2(r[2, 1], r[2, 2]), np.arctan2(r[1, 0], r[0, 0])
    else:
        roll, yaw = np.arctan2(-r[1, 2], r[1, 1]), 0.0
    origin.set("xyz", " ".join(f"{x:.17g}" for x in matrix[:3, 3]))
    origin.set("rpy", " ".join(f"{x:.17g}" for x in (roll, pitch, yaw)))


def align_joint_axes(root: ET.Element) -> list[str]:
    """Insert a rotated joint frame and an inverse fixed transform.

    O Rot(axis,q) = O R Rot(z,q) R^-1. Original link frames, collision sphere
    coordinates, joint names/positions and tool poses therefore stay unchanged.
    """
    used = {element.get("name") for element in root}
    converted = []
    for joint in list(root.findall("joint")):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        axis_element = joint.find("axis")
        axis = np.fromstring(axis_element.get("xyz", "1 0 0"), sep=" ") if axis_element is not None else np.array([1., 0., 0.])
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError(f"joint {joint.get('name')}: invalid joint axis")
        axis = axis / np.linalg.norm(axis)
        if axis_element is None:
            axis_element = ET.SubElement(joint, "axis")
        index = int(np.argmax(np.abs(axis)))
        cardinal = np.zeros(3); cardinal[index] = np.sign(axis[index])
        if np.linalg.norm(axis - cardinal) < 1e-12:
            axis_element.set("xyz", " ".join(f"{x:g}" for x in cardinal))
            continue
        # Pick a well-conditioned perpendicular vector; R's third column is axis.
        seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
        x = np.cross(seed, axis); x /= np.linalg.norm(x)
        y = np.cross(axis, x)
        rotation = np.eye(4); rotation[:3, :3] = np.column_stack((x, y, axis))
        helper = f"__axis_{joint.get('name')}"
        while helper in used or helper + "_restore" in used:
            helper += "_"
        used.update((helper, helper + "_restore"))
        child = joint.find("child")
        original_child = child.get("link")
        _set_origin(joint, origin_matrix(joint.find("origin")) @ rotation)
        axis_element.set("xyz", "0 0 1")
        child.set("link", helper)
        ET.SubElement(root, "link", {"name": helper})
        fixed = ET.SubElement(root, "joint", {"name": helper + "_restore", "type": "fixed"})
        ET.SubElement(fixed, "parent", {"link": helper})
        ET.SubElement(fixed, "child", {"link": original_child})
        _set_origin(fixed, rotation.T)
        converted.append(joint.get("name"))
    return converted
