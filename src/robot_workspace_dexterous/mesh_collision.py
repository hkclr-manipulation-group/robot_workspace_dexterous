"""Batched STL self-collision checking with persistent local-frame mesh BVHs."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import time

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from .mesh_model import MeshModel
from .urdf_compat import origin_matrix
from . import mesh_kernels as kernels


class MeshCollisionChecker:
    def __init__(self, model: MeshModel, urdf_path: str | Path, device: str = "cuda:0"):
        wp.init()
        self.device = wp.get_device(device)
        self.model = model
        self.root = ET.parse(urdf_path).getroot()
        joints = self.root.findall("joint")
        children = {joint.find("child").get("link") for joint in joints}
        bases = [link.get("name") for link in self.root.findall("link")
                 if link.get("name") not in children]
        if len(bases) != 1:
            raise ValueError("STL collision FK requires a single URDF root")
        self.links, self.joints, parents, origins, kinds, axes = [bases[0]], [None], [-1], [np.eye(4)], [0], [[1, 0, 0]]
        pending = list(joints)
        while pending:
            progressed = False
            for joint in list(pending):
                parent = joint.find("parent").get("link")
                if parent not in self.links:
                    continue
                child = joint.find("child").get("link")
                kind = joint.get("type")
                if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                    raise ValueError(f"Unsupported joint type: {kind}")
                axis_node = joint.find("axis")
                axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                    raise ValueError(f"Invalid joint axis: {joint.get('name')}")
                parents.append(self.links.index(parent))
                self.links.append(child)
                self.joints.append(joint)
                origins.append(origin_matrix(joint.find("origin")))
                axes.append(axis / np.linalg.norm(axis))
                kinds.append(0 if kind == "fixed" else 2 if kind == "prismatic" else 1)
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("URDF joint graph is disconnected or cyclic")
        origins = np.asarray(origins)
        transforms = np.concatenate([origins[:, :3, 3], Rotation.from_matrix(origins[:, :3, :3]).as_quat()], axis=1)
        self.parents = self._array(parents, wp.int32)
        self.origins = self._array(transforms, wp.transform)
        self.kinds = self._array(kinds, wp.int32)
        self.axes = self._array(axes, wp.vec3)
        self.meshes = [wp.Mesh(points=self._array(part.vertices, wp.vec3),
                               indices=self._array(part.faces.reshape(-1), wp.int32)) for part in model.parts]
        self.mesh_ids = self._array(np.asarray([mesh.id for mesh in self.meshes], dtype=np.uint64), wp.uint64)
        self.part_links = self._array([self.links.index(part.link) for part in model.parts], wp.int32)
        self.triangle_counts = self._array([len(part.faces) for part in model.parts], wp.int32)
        self.closed = self._array([part.watertight for part in model.parts], wp.int32)
        self.local_lo = self._array([part.vertices.min(axis=0) for part in model.parts], wp.vec3)
        self.local_hi = self._array([part.vertices.max(axis=0) for part in model.parts], wp.vec3)
        self.pairs = self._array(model.pairs, wp.int32)
        self.representatives = self._array(np.concatenate([p.representatives for p in model.parts]), wp.vec3)
        self.representative_offsets = self._array(np.cumsum([0] + [len(p.representatives) for p in model.parts]), wp.int32)
        self._joint_names = None
        self._batch = 0
        # BVHs are built on Warp's stream; publish them before use on a Torch stream.
        if self.device.is_cuda:
            wp.synchronize_device(self.device)

    def _array(self, data, dtype):
        return wp.array(data, dtype=dtype, device=self.device)

    def _prepare(self, batch: int, joint_names: list[str]) -> None:
        if self._joint_names != tuple(joint_names):
            mapping = {name: i for i, name in enumerate(joint_names)}
            by_name = {joint.get("name"): joint for joint in self.joints if joint is not None}

            def driver(joint, seen):
                name = joint.get("name")
                if name in seen:
                    raise ValueError("Cyclic mimic joints")
                mimic = joint.find("mimic")
                if mimic is not None:
                    index, mult, offset = driver(by_name[mimic.get("joint")], seen | {name})
                    m, o = float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))
                    return index, mult*m, offset*m+o
                if name not in mapping:
                    raise ValueError(f"IK result lacks joint required by STL FK: {name}")
                return mapping[name], 1., 0.

            drivers = [(-1, 1., 0.) if joint is None or joint.get("type") == "fixed"
                       else driver(joint, set()) for joint in self.joints]
            self.joint_indices = self._array([v[0] for v in drivers], wp.int32)
            self.multipliers = self._array([v[1] for v in drivers], wp.float32)
            self.joint_offsets = self._array([v[2] for v in drivers], wp.float32)
            self._joint_names = tuple(joint_names)
        if batch != self._batch:
            self.transforms = wp.empty((batch, len(self.links)), dtype=wp.transform, device=self.device)
            self.lows = wp.empty((batch, len(self.meshes)), dtype=wp.vec3, device=self.device)
            self.highs = wp.empty_like(self.lows)
            self.colliding = wp.zeros(batch, dtype=wp.int32, device=self.device)
            self.active = wp.ones(batch, dtype=wp.int32, device=self.device)
            self._batch = batch

    def _launch(self, q, active, stream=None) -> None:
        wp.launch(kernels.forward_kinematics, dim=self._batch, inputs=[
            q, active, self.parents, self.origins, self.joint_indices, self.kinds, self.axes,
            self.multipliers, self.joint_offsets, self.transforms, self.colliding,
        ], device=self.device, stream=stream)
        if not len(self.model.pairs):
            return
        wp.launch(kernels.world_bounds, dim=(self._batch, len(self.meshes)), inputs=[
            self.transforms, self.part_links, self.local_lo, self.local_hi,
            self.colliding, self.lows, self.highs,
        ], device=self.device, stream=stream)
        wp.launch(kernels.intersect_mesh_pairs,
                  dim=(self._batch, len(self.model.pairs), kernels.TRIANGLE_LANES), inputs=[
            self.transforms, self.part_links, self.mesh_ids, self.triangle_counts,
            self.pairs, self.lows, self.highs, self.colliding,
        ], device=self.device, stream=stream)
        wp.launch(kernels.contained_mesh_pairs, dim=(self._batch, len(self.model.pairs)), inputs=[
            self.transforms, self.part_links, self.mesh_ids, self.closed, self.representatives,
            self.representative_offsets, self.pairs, self.lows, self.highs, self.colliding,
        ], device=self.device, stream=stream)

    def check_numpy(self, q: np.ndarray, joint_names: list[str]) -> np.ndarray:
        """Return collision flags [B] for joint positions [B,D], for diagnostics/tests."""
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != len(joint_names) or len(q) == 0:
            raise ValueError("Joint positions must have shape [B,D], B > 0")
        self._prepare(len(q), joint_names)
        self._launch(self._array(q, wp.float32), self.active)
        return self.colliding.numpy().astype(bool)

    def check_torch(self, q, joint_names: list[str], valid):
        """Return collision flags shaped like q[...,0], on the input PyTorch stream."""
        import torch
        if q.dtype != torch.float32 or not q.is_contiguous() or q.shape[-1] != len(joint_names):
            raise ValueError("STL checker requires contiguous float32 joint positions")
        if str(q.device) != str(self.device):
            raise ValueError("IK and STL checker must use the same device")
        flat = q.reshape(-1, q.shape[-1])
        stream = wp.stream_from_torch(torch.cuda.current_stream(q.device)) if q.is_cuda else None
        active = valid.reshape(-1).to(dtype=torch.int32).contiguous()
        if stream is not None:
            with wp.ScopedStream(stream):
                self._prepare(len(flat), joint_names)
                self._launch(wp.from_torch(flat, requires_grad=False),
                             wp.from_torch(active, requires_grad=False), stream)
        else:
            self._prepare(len(flat), joint_names)
            self._launch(wp.from_torch(flat, requires_grad=False),
                         wp.from_torch(active, requires_grad=False))
        return wp.to_torch(self.colliding).view(q.shape[:-1]) != 0

    def pair_frequencies(self, q: np.ndarray, joint_names: list[str]) -> list[dict]:
        """Diagnose each pair separately, reusing BVHs; intentionally slower than boolean checking."""
        original_pairs, original_array = self.model.pairs, self.pairs
        results = []
        try:
            for pair in original_pairs:
                self.model.pairs = pair.reshape(1, 2)
                self.pairs = self._array(self.model.pairs, wp.int32)
                count = sum(int(self.check_numpy(q[start:start+32], joint_names).sum())
                            for start in range(0, len(q), 32))
                if count:
                    results.append({"links": [self.model.parts[int(i)].link for i in pair],
                                    "colliding_samples": count, "fraction": count / len(q)})
        finally:
            self.model.pairs, self.pairs = original_pairs, original_array
        return sorted(results, key=lambda item: item["colliding_samples"], reverse=True)


def diagnose_mesh_collisions(model: MeshModel, urdf_path: str | Path, samples: int = 2048) -> dict:
    """Check random configurations against STL on GPU; report collision throughput, not IK speed."""
    if samples < 1:
        raise ValueError("diagnostic-samples must be positive")
    started = time.perf_counter()
    checker = MeshCollisionChecker(model, urdf_path)
    joints = [joint for joint in checker.root.findall("joint") if joint.get("type") != "fixed"]
    names = [joint.get("name") for joint in joints]
    rng = np.random.default_rng(42)
    columns = []
    for joint in joints:
        limit = joint.find("limit")
        low, high = ((-np.pi, np.pi) if joint.get("type") == "continuous" else
                     (float(limit.get("lower")), float(limit.get("upper"))))
        columns.append(rng.uniform(low, high, samples))
    q = np.asarray(columns, dtype=np.float32).T if columns else np.empty((samples, 0), dtype=np.float32)
    checker.check_numpy(q[:min(32, samples)], names)
    initialization = time.perf_counter() - started
    started = time.perf_counter()
    flags = np.concatenate([checker.check_numpy(q[start:start+32], names)
                            for start in range(0, samples, 32)])
    elapsed = time.perf_counter() - started
    # A small deterministic subset identifies structural contacts without making
    # the default 2048-sample throughput run quadratic in the number of pairs.
    pair_samples = min(32, samples)
    frequencies = checker.pair_frequencies(q[:pair_samples], names)
    return {"samples": samples, "collision_free_samples": int((~flags).sum()),
            "pair_diagnostic_samples": pair_samples, "pair_frequencies": frequencies,
            "collision_model": model.metadata, "initialization_seconds": initialization,
            "collision_check_seconds": elapsed, "configurations_per_second": samples/elapsed,
            "timing_scope": "FK + STL collision + host copies; excludes IK"}
