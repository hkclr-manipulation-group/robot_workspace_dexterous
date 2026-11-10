"""Compare native bare flanges at common arm limits with full-mesh contact filtering."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import qmc

from collision_shpere_generation.kinematics import RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
from robot_workspace_dexterous.comparison_limits import write_half_turn_arm
from robot_workspace_dexterous.sampling import generate_uniform_quaternions

ROOT = Path(__file__).resolve().parents[3]
MODELS = ["spark2_v1", "spark2_v2", "spark2_v2_2", *[f"P2AB-{i}" for i in range(1, 5)]]
POINTS = [[x / 10, 0., z / 10] for z in (-3, -1, 1, 3, 5, 7) for x in range(-6, 7, 2)]


def evaluate(name):
    if name.startswith("spark"):
        source = ROOT / "collision_shpere_generation/models" / name / "robot.urdf"
        flange = "arm_end_effector"
    else:
        source, = (ROOT / "cuarm_mesh" / name / "urdf").glob("*.urdf")
        flange = "L" + "6" * int(name[-1])
    mesh = MeshCollisionChecker.from_urdf(source, asset_roots=[source.parent.parent])
    ignore = [(j.find("parent").get("link"), j.find("child").get("link"))
              for j in ET.parse(source).findall("joint")]
    with TemporaryDirectory() as tmp:
        effective = Path(tmp) / "arm.urdf"
        write_half_turn_arm(source, effective)
        arm = RobotKinematics(effective)
    angles = (2 * qmc.Sobol(6, scramble=True, seed=42).random_base2(11) - 1) * np.pi
    cells, free_cells = set(), set()
    free = 0
    for q in angles:
        poses = arm.forward(q)
        cell = tuple(np.floor(poses[flange][:3, 3] / .1).astype(int))
        cells.add(cell)
        if not mesh.check(poses, ignores=ignore):
            free += 1
            free_cells.add(cell)

    orientations = generate_uniform_quaternions(16)
    coverage = []
    for index, point in enumerate(POINTS):
        accepted = []
        for i, orientation in enumerate(orientations):
            result = arm.solve(point, flange, orientation, seeds=8, seed=index*16+i,
                               collision=lambda q: mesh.check(arm.forward(q), ignores=ignore))
            if result.success:
                accepted.append(i)
        coverage.append(accepted)
    return dict(model=name, flange=flange, joint_names=arm.joint_names,
                collision_free_samples=free, occupied_cells=len(cells),
                collision_free_cells=len(free_cells), accepted_orientation_ids=coverage)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate, MODELS))
    print(json.dumps(dict(limits_deg=[-180, 180], sobol_samples=2048, sobol_seed=42,
                          cell_m=.1, targets_m=POINTS, orientations=16, ik_seeds=8,
                          position_tolerance_m=.005, orientation_tolerance_rad=.08,
                          models=rows), indent=2))
