"""Target-point occupancy in fixed robot geometry, independent of pair ignores."""
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from .urdf_compat import origin_matrix


def fixed_poses(urdf_path):
    root = ET.parse(urdf_path).getroot()
    children = {j.find('child').get('link') for j in root.findall('joint')}
    poses = {l.get('name'): np.eye(4) for l in root.findall('link')
             if l.get('name') not in children}
    pending = [j for j in root.findall('joint') if j.get('type') == 'fixed']
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in poses]
        if not ready:
            break
        for joint in ready:
            poses[joint.find('child').get('link')] = (
                poses[joint.find('parent').get('link')] @ origin_matrix(joint.find('origin')))
            pending.remove(joint)
    return poses


def fixed_occupancy(model, urdf_path, points):
    """Return blocked target centres and open fixed links needing geometry review.

    Closed meshes use parity containment (retaining cavities). Open meshes
    only block their surfaces; their interior cannot be certified. Moving
    links must be checked at each IK solution, never at the zero pose.
    """
    poses = fixed_poses(urdf_path)
    blocked = np.zeros(len(points), dtype=bool)
    unresolved = set()
    tolerance = float(model.metadata.get('contact_tolerance_m', 1e-6))
    for part in model.parts:
        if part.link not in poses:
            continue
        pose = poses[part.link]
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        lo, hi = part.vertices.min(0), part.vertices.max(0)
        candidates = np.flatnonzero(~blocked & np.all(
            (local >= lo - tolerance) & (local <= hi + tolerance), axis=1))
        if not part.watertight:
            unresolved.add(part.link)
        mesh = trimesh.Trimesh(part.vertices, part.faces, process=False)
        for start in range(0, len(candidates), 128):
            indices = candidates[start:start + 128]
            xyz = local[indices]
            _, distance, _ = trimesh.proximity.closest_point(mesh, xyz)
            hit = distance <= tolerance
            if part.watertight:
                hit[~hit] = mesh.contains(xyz[~hit])
            blocked[indices] |= hit
    return blocked, sorted(unresolved)
