"""CPU forward-kinematics and sphere-collision checks before a GPU workspace run."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml

from .urdf_compat import origin_matrix


def forward_kinematics(root: ET.Element, positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(next(iter(positions.values()))) if positions else 1
    joints = root.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    bases = {link.get("name") for link in root.findall("link")} - children
    poses = {base: np.broadcast_to(np.eye(4), (count, 4, 4)).copy() for base in bases}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in list(pending):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in poses:
                continue
            transform = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            if joint.get("type") != "fixed":
                q = positions[joint.get("name")]
                mimic = joint.find("mimic")
                if mimic is not None:
                    q = positions[mimic.get("joint")] * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
                axis_tag = joint.find("axis")
                axis = np.fromstring(axis_tag.get("xyz"), sep=" ") if axis_tag is not None else np.array([1., 0., 0.])
                axis /= np.linalg.norm(axis)
                if joint.get("type") == "prismatic":
                    transform[:, :3, 3] = q[:, None] * axis
                else:
                    x, y, z = axis
                    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    transform[:, :3, :3] = np.eye(3) + np.sin(q)[:, None, None]*k + (1-np.cos(q))[:, None, None]*(k@k)
            poses[child] = poses[parent] @ origin_matrix(joint.find("origin")) @ transform
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF joint graph is disconnected or cyclic")
    return poses


def diagnose_collisions(urdf: str | Path, sphere_path: str | Path,
                        ignores: dict[str, list[str]], samples: int = 2048) -> dict:
    root = ET.parse(urdf).getroot()
    rng = np.random.default_rng(42)
    positions = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            limit = joint.find("limit")
            low, high = (-np.pi, np.pi) if joint.get("type") == "continuous" else (float(limit.get("lower")), float(limit.get("upper")))
            positions[joint.get("name")] = rng.uniform(low, high, samples)
    poses = forward_kinematics(root, positions)
    data = yaml.safe_load(Path(sphere_path).read_text(encoding="utf-8"))["collision_spheres"]
    excluded = {tuple(sorted((a, b))) for a, others in ignores.items() for b in others}
    world = {}
    for link, spheres in data.items():
        centers = np.array([s["center"] for s in spheres])
        world[link] = (np.einsum("nij,sj->nsi", poses[link][:, :3, :3], centers) + poses[link][:, None, :3, 3],
                       np.array([s["radius"] for s in spheres]))
    valid = np.ones(samples, dtype=bool)
    pairs = []
    names = list(world)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if tuple(sorted((first, second))) in excluded:
                continue
            a, ra = world[first]; b, rb = world[second]
            gaps = np.empty(samples)
            for start in range(0, samples, 64):
                distance = np.linalg.norm(a[start:start+64, :, None] - b[start:start+64, None, :], axis=-1)
                gaps[start:start+64] = (distance - ra[None, :, None] - rb[None, None, :]).min(axis=(1, 2))
            colliding = gaps < 0
            valid &= ~colliding
            if colliding.any():
                pairs.append({"links": [first, second], "collision_fraction": float(colliding.mean()),
                              "best_gap_m": float(gaps.max()), "worst_gap_m": float(gaps.min())})
    return {"samples": samples, "collision_free_samples": int(valid.sum()),
            "pairs": sorted(pairs, key=lambda pair: -pair["collision_fraction"])}
