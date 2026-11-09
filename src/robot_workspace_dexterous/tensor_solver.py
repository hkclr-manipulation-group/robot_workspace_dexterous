"""Batched sphere workspace computation without cuRobo binary extensions."""
from __future__ import annotations
import time
import numpy as np
from .sampling import DexterousWorkspace, regular_grid
from .curobo_solver import _load_collision_spheres


def compute_tensor_workspace(config, ee_link, *, contact_ignores=None,
                             progress=None, snapshot=None, snapshot_seconds=30.):
    from collision_shpere_generation.tensor_kinematics import TensorKinematics
    if config.collision_backend != 'spheres':
        raise ValueError('The torch backend requires sphere collision geometry')
    spheres = _load_collision_spheres(str(config.collision_spheres_path)) if config.self_collision else {}
    ignores = contact_ignores if contact_ignores is not None else config.self_collision_ignore
    robot = TensorKinematics(config.urdf_path, config.base_link, spheres=spheres,
        ignores={(a,b) for a,others in ignores.items() for b in others})
    positions = regular_grid(config.x_range,config.y_range,config.heights,config.resolution)
    blocked = robot.blocked_targets(positions) if config.self_collision else np.zeros(len(positions),bool)
    blocked |= ~robot.reachable_bound(positions,ee_link,config.position_tolerance)
    count, orientations = len(positions), len(config.orientations)
    reached = np.zeros((count,orientations),bool)
    ws, w2s = np.zeros(count), np.zeros(count)
    conditions, minima = np.full(count,np.nan), np.full(count,np.nan)
    started, last_snapshot = time.monotonic(), 0.

    def result():
        counts = reached.sum(1).astype(np.int32)
        denominator = np.maximum(counts,1)
        return DexterousWorkspace(positions,counts.astype(np.float32)/orientations,counts,orientations,
            (ws/denominator).astype(np.float32),(w2s/denominator).astype(np.float32),
            conditions.astype(np.float32),minima.astype(np.float32),reached.copy())

    batch = min(config.batch_size,max(64,2048//config.ik_seeds))
    total = count*orientations
    for start in range(0,total,batch):
        stop = min(total,start+batch)
        ids = np.arange(start,stop)
        cells, angles = ids//orientations,ids%orientations
        eligible = ~blocked[cells]
        if eligible.any():
            ci,oi = cells[eligible],angles[eligible]
            ok,_,singular = robot.solve_batch(positions[ci],ee_link,config.orientations[oi],
                seeds=config.ik_seeds,iterations=config.ik_iterations,
                position_tolerance=config.position_tolerance,orientation_tolerance=config.orientation_tolerance,
                seed=config.random_seed+start)
            ci,oi,singular = ci[ok],oi[ok],singular[ok]
            reached[ci,oi] = True
            if len(ci):
                w = singular.prod(1)
                np.add.at(ws,ci,w); np.add.at(w2s,ci,w*w)
                condition = np.divide(singular[:,0],singular[:,-1],out=np.full(len(ci),np.inf),where=singular[:,-1]>1e-9)
                np.fmax.at(conditions,ci,condition); np.fmin.at(minima,ci,singular[:,-1])
        now = time.monotonic()
        if progress:
            progress(stop,total,now-started)
        if snapshot and (last_snapshot == 0 or stop == total or now-last_snapshot >= snapshot_seconds):
            snapshot(result(),stop,total,now-started)
            last_snapshot = time.monotonic()
    return result()
