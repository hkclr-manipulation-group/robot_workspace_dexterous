"""Explicit STL collision policy for contacts at rigid assemblies and joints."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .mesh_model import MeshModel


def adjacent_collision_pairs(urdf_path: str | Path) -> set[tuple[str, str]]:
    """Return link pairs that form one rigid body or one moving joint.

    CAD exports normally contain overlapping housings at a joint. Those
    interfaces are part of the robot and must not invalidate every IK result.
    Pairs separated by two or more moving joints remain collision checked.
    """
    root = ET.parse(urdf_path).getroot()
    parent = {link.get("name"): link.get("name") for link in root.findall("link")}

    def representative(link: str) -> str:
        while parent[link] != link:
            parent[link] = parent[parent[link]]
            link = parent[link]
        return link

    moving: list[tuple[str, str]] = []
    for joint in root.findall("joint"):
        first = joint.find("parent").get("link")
        second = joint.find("child").get("link")
        fixed = joint.get("type") == "fixed"
        limit = joint.find("limit")
        if (joint.get("type") in {"revolute", "prismatic"} and limit is not None
                and joint.find("mimic") is None):
            low, high = limit.get("lower"), limit.get("upper")
            if low is not None and high is not None:
                fixed = float(low) == float(high)
        if fixed:
            parent[representative(second)] = representative(first)
        else:
            moving.append((first, second))

    result: set[tuple[str, str]] = set()
    for first, second in moving:
        result.add(tuple(sorted((first, second))))
        result.add(tuple(sorted((representative(first), representative(second)))))
    groups: dict[str, list[str]] = {}
    for link in parent:
        groups.setdefault(representative(link), []).append(link)
    for links in groups.values():
        for i, first in enumerate(links):
            for second in links[i + 1:]:
                result.add(tuple(sorted((first, second))))
    return result


def merge_contact_ignores(urdf_path: str | Path,
                          explicit: dict[str, list[str]] | None = None
                          ) -> dict[str, list[str]]:
    """Merge reviewed ignores with the structural adjacent-joint policy."""
    pairs = adjacent_collision_pairs(urdf_path)
    for first, others in (explicit or {}).items():
        pairs.update(tuple(sorted((first, second))) for second in others)
    merged: dict[str, list[str]] = {}
    for first, second in sorted(pairs):
        merged.setdefault(first, []).append(second)
    return merged


def allow_joint_contacts(model: MeshModel, urdf_path: str | Path) -> MeshModel:
    """Exclude whole link pairs within one rigid body or across one moving joint.

    This is an opt-in modeling assumption, not an inference from collision
    frequency. Fixed and non-mimic zero-range joints form rigid bodies. Geometry-free
    links participate in the graph, so fixed tool adapters do not hide adjacency.
    Pairs across two or more moving joints remain subject to collision checking.
    """
    root = ET.parse(urdf_path).getroot()
    parent = {link.get("name"): link.get("name") for link in root.findall("link")}

    def representative(link):
        while parent[link] != link:
            parent[link] = parent[parent[link]]
            link = parent[link]
        return link

    moving = []
    for joint in root.findall("joint"):
        first, second = joint.find("parent").get("link"), joint.find("child").get("link")
        fixed = joint.get("type") == "fixed"
        limit = joint.find("limit")
        if (joint.get("type") in {"revolute", "prismatic"} and limit is not None
                and joint.find("mimic") is None):
            low, high = limit.get("lower"), limit.get("upper")
            if low is not None and high is not None:
                low, high = float(low), float(high)
                fixed = math.isfinite(low) and low == high
        if fixed:
            parent[representative(second)] = representative(first)
        else:
            moving.append((first, second))
    adjacent = {frozenset((representative(a), representative(b))) for a, b in moving}
    retained, excluded = [], set()
    for i, j in model.pairs:
        first, second = model.parts[int(i)].link, model.parts[int(j)].link
        a, b = representative(first), representative(second)
        if a == b or frozenset((a, b)) in adjacent:
            excluded.add(tuple(sorted((first, second))))
        else:
            retained.append((i, j))
    metadata = {**model.metadata,
                "joint_contact_policy": "allow_rigid_and_joint_neighbors",
                "joint_contact_excluded_pairs": [list(pair) for pair in sorted(excluded)],
                "joint_contact_limitation": "Whole listed link pairs are excluded, including collisions away from their joint interfaces.",
                "collision_pairs": len(retained)}
    return MeshModel(model.parts, np.asarray(retained, dtype=np.int32).reshape(-1, 2), metadata)
