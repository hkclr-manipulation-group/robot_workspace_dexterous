"""Load portable STL geometry in the exact URDF link frames, without CUDA."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .checksums import matches_text_sha256
from .urdf_compat import origin_matrix


@dataclass
class MeshPart:
    link: str
    vertices: np.ndarray
    faces: np.ndarray
    representatives: np.ndarray
    watertight: bool


@dataclass
class MeshModel:
    parts: list[MeshPart]
    pairs: np.ndarray
    metadata: dict


def load_mesh_model(manifest_path: str | Path, urdf_path: str | Path,
                    ignores: dict[str, list[str]]) -> MeshModel:
    """Load STL triangles, preserving scale/origin and testing closed-mesh containment eligibility."""
    import trimesh

    manifest_path, urdf_path = Path(manifest_path).resolve(), Path(urdf_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = yaml.safe_load(raw)
    if manifest.get("version") != 1 or not manifest.get("links"):
        raise ValueError("Invalid STL collision manifest")
    if not matches_text_sha256(urdf_path.read_bytes(), manifest.get("urdf_sha256", "")):
        raise ValueError("STL manifest kinematics checksum differs from the configured URDF")
    links = {link.get("name") for link in ET.parse(urdf_path).getroot().findall("link")}
    required = set(manifest["links"]) | set(ignores) | {v for values in ignores.values() for v in values}
    if required - links:
        raise ValueError(f"STL/ignore links absent from URDF: {sorted(required - links)}")
    parts, assets, cache, digests = [], {}, {}, {}
    for link, entries in manifest["links"].items():
        if not entries:
            raise ValueError(f"No STL geometry for collision link {link}")
        for entry in entries:
            path = (manifest_path.parent / entry["file"]).resolve()
            if path.suffix.lower() != ".stl":
                raise ValueError(f"Collision geometry must be STL: {path}")
            key = (str(path), tuple(entry["scale"]), tuple(entry["xyz"]), tuple(entry["rpy"]))
            if path not in digests:
                digests[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            digest = digests[path]
            if digest != entry["sha256"]:
                raise ValueError(f"STL checksum mismatch: {path}")
            assets[entry["file"]] = digest
            if key not in cache:
                mesh = trimesh.load(path, force="mesh", process=False)
                triangles = np.asarray(mesh.triangles, dtype=np.float32)
                if not len(triangles) or not np.isfinite(triangles).all():
                    raise ValueError(f"Empty or nonfinite STL: {path}")
                scale = np.asarray(entry["scale"], dtype=np.float32)
                if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale == 0):
                    raise ValueError(f"Invalid STL scale: {path}")
                origin = ET.Element("origin", xyz=" ".join(map(str, entry["xyz"])),
                                    rpy=" ".join(map(str, entry["rpy"])))
                transform = origin_matrix(origin).astype(np.float32)
                triangles = (triangles * scale) @ transform[:3, :3].T + transform[:3, 3]
                # Exact welding only: topology checks must not seal CAD gaps.
                vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
                faces = inverse.reshape(-1, 3).astype(np.int32)
                area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                               triangles[:, 2]-triangles[:, 0]), axis=1)
                faces = faces[area > 0]
                if not len(faces):
                    raise ValueError(f"STL has no nondegenerate triangles: {path}")
                topology = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                components = trimesh.graph.connected_components(
                    topology.face_adjacency, nodes=np.arange(len(faces)), engine="scipy")
                representatives = np.asarray([vertices[faces[component[0], 0]]
                                              for component in components], dtype=np.float32)
                cache[key] = (vertices.astype(np.float32), faces, representatives,
                              bool(topology.is_watertight))
            parts.append(MeshPart(link, *cache[key]))
    excluded = {tuple(sorted((a, b))) for a, others in ignores.items() for b in others}
    pairs = []
    for i, a in enumerate(parts):
        for j in range(i + 1, len(parts)):
            b = parts[j]
            if a.link == b.link or tuple(sorted((a.link, b.link))) in excluded:
                continue
            # Traverse the smaller mesh and query the larger mesh's static BVH.
            pairs.append((i, j) if len(a.faces) <= len(b.faces) else (j, i))
    metadata = {
        "mode": "stl", "file": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "assets": assets,
        "total_triangles": sum(len(part.faces) for part in parts),
        "mesh_parts": len(parts), "collision_pairs": len(pairs),
        "open_mesh_links": sorted({part.link for part in parts if not part.watertight}),
        "collision_method": "BVH triangle intersection + containment for closed meshes",
        "open_mesh_policy": "surface intersection only; open STL has no defined solid interior",
        "contact_tolerance_m": 1e-6,
        "ik_collision_method": "all returned pose-IK seeds checked against STL; no sphere cost",
        "joint_contact_policy": "strict",
        "joint_contact_excluded_pairs": [],
    }
    return MeshModel(parts, np.asarray(pairs, dtype=np.int32).reshape(-1, 2), metadata)
