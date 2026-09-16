"""Measure GPU broad-phase savings on real robot poses without running IK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.diagnostics import forward_kinematics
from curobo._src.geom.collision.wp_self_collision import (
    PAIR_LANES, check_link_pairs, finish_collision, launch_grouped_collision, reduce_link_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spark2_v2.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--kernel-cache", default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch-size and repeats must be positive")
    if args.kernel_cache:
        wp.config.kernel_cache_dir = args.kernel_cache
    wp.init()
    device = "cuda:0"
    config = load_config(args.config)
    root = ET.parse(config.urdf_path).getroot()
    rng = np.random.default_rng(42)
    positions = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        low, high = ((-np.pi, np.pi) if joint.get("type") == "continuous" else
                     (float(limit.get("lower")), float(limit.get("upper"))))
        positions[joint.get("name")] = rng.uniform(low, high, args.batch_size).astype(np.float32)
    transforms = forward_kinematics(root, positions)
    with config.collision_spheres_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)["collision_spheres"]
    offsets = [0]
    world = []
    for link, entries in data.items():
        centers = np.asarray([item["center"] for item in entries], dtype=np.float32)
        spheres = np.empty((args.batch_size, len(entries), 4), dtype=np.float32)
        spheres[..., :3] = (np.einsum("bij,nj->bni", transforms[link][:, :3, :3], centers)
                            + transforms[link][:, None, :3, 3])
        spheres[..., 3] = np.asarray([item["radius"] for item in entries], dtype=np.float32)
        world.append(spheres)
        offsets.append(offsets[-1] + len(entries))
    names = list(data)
    ignored = {tuple(sorted((a, b))) for a, others in config.self_collision_ignore.items()
               for b in others}
    pairs = np.asarray([(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
                        if tuple(sorted((names[i], names[j]))) not in ignored], dtype=np.int32)
    batch, count, _ = np.concatenate(world, axis=1).shape
    spheres = wp.array(np.concatenate(world, axis=1), dtype=wp.vec4, device=device)
    padding = wp.zeros(count, dtype=float, device=device)
    indices = wp.array(np.arange(count, dtype=np.int32), device=device)
    group_offsets = wp.array(np.asarray(offsets, dtype=np.int32), device=device)
    link_pairs = wp.array(pairs, device=device)
    bounds = wp.empty((batch, len(names)), dtype=wp.vec4, device=device)
    tasks = len(pairs) * PAIR_LANES
    values = wp.empty((batch, tasks), dtype=float, device=device)
    winners = wp.empty((batch, tasks), dtype=wp.vec2i, device=device)
    link_values = wp.empty((batch, len(pairs)), dtype=float, device=device)
    link_winners = wp.empty((batch, len(pairs)), dtype=wp.vec2i, device=device)
    weight = wp.array(np.array([1.0], dtype=np.float32), device=device)
    distance = wp.zeros(batch, dtype=float, device=device)
    gradient = wp.zeros((batch, count), dtype=wp.vec4, device=device)
    gradient_indices = wp.zeros(batch, dtype=wp.vec2i, device=device)
    launch_args = [spheres, padding, indices, group_offsets, link_pairs, bounds, values,
                   winners, link_values, link_winners, weight, distance, gradient,
                   gradient_indices, True, device]

    def grouped() -> None:
        launch_grouped_collision(*launch_args)

    def exhaustive() -> None:
        # Same exact narrow phase, with deliberately overlapping outer bounds.
        # This is a culling-on/off comparison, not the legacy CUDA backend.
        wp.launch(check_link_pairs, dim=(batch, tasks), inputs=[
            spheres, padding, indices, group_offsets, link_pairs, bounds,
            values, winners, PAIR_LANES,
        ], device=device)
        wp.launch(reduce_link_pairs, dim=link_values.shape, inputs=[
            values, winners, link_values, link_winners, PAIR_LANES,
        ], device=device)
        wp.launch(finish_collision, dim=batch, inputs=[
            spheres, link_values, link_winners, weight, distance, gradient, gradient_indices,
            len(pairs), True,
        ], device=device)

    def measure(fn) -> float:
        fn()
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            fn()
        started = time.perf_counter()
        for _ in range(args.repeats):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        return (time.perf_counter() - started) * 1000 / args.repeats

    grouped_ms = measure(grouped)
    expected_distance, expected_gradient = distance.numpy(), gradient.numpy()
    outer = bounds.numpy()
    checks = np.asarray([(offsets[i + 1] - offsets[i]) * (offsets[j + 1] - offsets[j])
                         for i, j in pairs], dtype=np.int64)
    delta = outer[:, pairs[:, 0], :3] - outer[:, pairs[:, 1], :3]
    radius = outer[:, pairs[:, 0], 3] + outer[:, pairs[:, 1], 3]
    separated = np.sum(delta * delta, axis=-1) > radius * radius
    outer[..., :3] = 0
    outer[..., 3] = 1e10
    bounds.assign(outer)
    exhaustive_ms = measure(exhaustive)
    np.testing.assert_allclose(distance.numpy(), expected_distance, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(gradient.numpy(), expected_gradient, atol=2e-6, rtol=2e-5)
    print(json.dumps({
        "model": Path(args.config).stem, "batch_size": batch, "spheres": count,
        "link_pairs": len(pairs), "sphere_pairs_per_pose": int(checks.sum()),
        "sphere_pair_checks_culled_fraction": float((separated * checks).sum() / (batch * checks.sum())),
        "grouped_collision_ms": grouped_ms, "unculled_same_kernel_ms": exhaustive_ms,
        "culling_speedup": exhaustive_ms / grouped_ms,
        "cost_and_gradient_match": True, "scope": "collision only, excludes FK and IK",
    }, indent=2))


if __name__ == "__main__":
    main()
