"""CPU workspace IK, collision filtering and Jacobian metrics."""
from __future__ import annotations

import time
import numpy as np

from .sampling import DexterousWorkspace, regular_grid


def mesh_collision_callback(model, kinematics):
    """Reuse CPU FCL BVHs and the exact effective part-pair exclusions."""
    import trimesh
    from collision_shpere_generation.mesh_collision import MeshCollisionChecker
    meshes = {f'{part.link}#{i}':trimesh.Trimesh(part.vertices, part.faces, process=False)
              for i, part in enumerate(model.parts)}
    checker = MeshCollisionChecker(meshes, containment=True)
    allowed = {tuple(sorted((int(i),int(j)))) for i,j in model.pairs}
    names = list(meshes)
    ignored = {(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names))
               if (i,j) not in allowed}
    model.metadata['cpu_collision_engine'] = 'FCL BVH + closed-mesh containment'
    clearance = float(model.metadata.get('contact_tolerance_m', 1e-6))
    def check(q):
        poses = kinematics.forward(q)
        return checker.check({name:poses[part.link] for name,part in zip(names,model.parts)},
                             ignores=ignored, clearance=clearance)
    return check


def sphere_collision_callback(path, kinematics, ignores):
    from collision_shpere_generation.fast_collision import FastSphereCollisionChecker
    from .curobo_solver import _load_collision_spheres
    checker = FastSphereCollisionChecker(_load_collision_spheres(str(path)))
    excluded = {(first, second) for first, others in ignores.items() for second in others}
    return lambda q: checker.check(kinematics.forward(q), ignores=excluded)


def compute_cpu_workspace(config, ee_link, *, mesh_model=None, contact_ignores=None,
                          progress=None, snapshot=None, snapshot_seconds=30.):
    from collision_shpere_generation.kinematics import RobotKinematics
    kinematics = RobotKinematics(config.urdf_path, config.base_link)
    collision = None
    if config.self_collision:
        collision = (mesh_collision_callback(mesh_model, kinematics) if mesh_model is not None else
                     sphere_collision_callback(config.collision_spheres_path, kinematics,
                                               contact_ignores or config.self_collision_ignore))
    positions = regular_grid(config.x_range, config.y_range, config.heights, config.resolution)
    blocked = np.zeros(len(positions), bool)
    if config.self_collision and mesh_model is not None:
        from .occupancy import fixed_occupancy
        blocked, unresolved = fixed_occupancy(mesh_model, config.urdf_path, positions, config.base_link)
        mesh_model.metadata['fixed_target_occupancy'] = {
            'blocked_cells':int(blocked.sum()), 'unresolved_open_links':unresolved,
            'policy':'closed material and mesh surfaces; no convex-hull filling'}
    n, orientation_count = len(positions), len(config.orientations)
    counts = np.zeros(n, np.int32)
    w_sum, w2_sum = np.zeros(n), np.zeros(n)
    conditions, minima = np.full(n, np.nan), np.full(n, np.nan)
    reached = np.zeros((n, orientation_count), bool)
    total = n*orientation_count
    started, last_snapshot, warm = time.monotonic(), None, None

    def result():
        denominator = np.maximum(counts, 1)
        return DexterousWorkspace(positions, counts.astype(np.float32)/orientation_count,
            counts.copy(), orientation_count, (w_sum/denominator).astype(np.float32),
            (w2_sum/denominator).astype(np.float32), conditions.astype(np.float32),
            minima.astype(np.float32), reached.copy())

    for flat in range(total):
        cell, orientation = divmod(flat, orientation_count)
        if not blocked[cell]:
            solved = kinematics.solve(positions[cell], ee_link, config.orientations[orientation],
                seeds=config.ik_seeds, max_iterations=config.ik_iterations,
                position_tolerance=config.position_tolerance,
                orientation_tolerance=config.orientation_tolerance,
                collision=collision, warm_start=warm, seed=config.random_seed+flat)
            if solved.success:
                warm = solved.positions
                singular = np.linalg.svd(kinematics.jacobian(warm, ee_link), compute_uv=False)
                singular = np.pad(singular, (0, max(0, 6-len(singular))))
                w, minimum = float(np.prod(singular)), float(singular[-1])
                condition = float(singular[0]/minimum) if minimum > 1e-9 else np.inf
                counts[cell] += 1; reached[cell, orientation] = True
                w_sum[cell] += w; w2_sum[cell] += w*w
                conditions[cell] = np.fmax(conditions[cell], condition)
                minima[cell] = np.fmin(minima[cell], minimum)
        done, now = flat+1, time.monotonic()
        # CPU callbacks need not wait for a large GPU-oriented batch size.
        if progress and (done == total or done % min(config.batch_size, 32) == 0):
            progress(done, total, now-started)
        if snapshot and (last_snapshot is None or done == total or now-last_snapshot >= snapshot_seconds):
            snapshot(result(), done, total, now-started)
            last_snapshot = time.monotonic()
    return result()


def diagnose_cpu_collisions(config, mesh_model, contact_ignores, samples):
    from collision_shpere_generation.kinematics import RobotKinematics
    if samples < 1:
        raise ValueError('diagnostic-samples must be positive')
    started = time.perf_counter()
    kinematics = RobotKinematics(config.urdf_path,config.base_link)
    check = (mesh_collision_callback(mesh_model,kinematics) if mesh_model is not None else
             sphere_collision_callback(config.collision_spheres_path,kinematics,contact_ignores))
    initialization = time.perf_counter()-started
    q = np.random.default_rng(config.random_seed).uniform(kinematics.lower,kinematics.upper,
                                                         (samples,len(kinematics.joint_names)))
    started = time.perf_counter()
    flags = np.asarray([check(row) for row in q],bool)
    elapsed = time.perf_counter()-started
    return {'backend':'cpu','samples':samples,'collision_free_samples':int((~flags).sum()),
            'initialization_seconds':initialization,'collision_check_seconds':elapsed,
            'configurations_per_second':samples/elapsed,
            'timing_scope':'CPU FK + collision; excludes IK'}
