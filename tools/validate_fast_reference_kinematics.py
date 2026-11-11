"""Compare optional JIT FK/Jacobians against the independent Python reference."""
import json
from pathlib import Path
import numpy as np
from collision_shpere_generation.kinematics import RobotKinematics
from fast_reference_kinematics import FastReferenceKinematics

ROOT = Path(__file__).resolve().parents[1] / 'workspace_results/nine_model_comparison'
np.seterr(all='ignore')
rng = np.random.default_rng(3)
report = {}
for p in sorted((ROOT / 'models').glob('*/model.json')):
    meta = json.loads(p.read_text())
    args = (p.parent / 'robot.urdf', meta['base'])
    reference = RobotKinematics(*args, locked_joints=meta['locks'])
    accelerated = FastReferenceKinematics(*args, locked_joints=meta['locks'])
    worst_fk = worst_jac = 0.
    for q in rng.uniform(reference.lower, reference.upper, size=(64, len(reference.lower))):
        expected, actual = reference.forward(q), accelerated.forward(q)
        for link in expected:
            np.testing.assert_allclose(expected[link], actual[link], rtol=0, atol=1e-12)
            worst_fk = max(worst_fk, float(np.max(np.abs(expected[link] - actual[link]))))
        a = reference.jacobian(q, meta['ee'])
        b = accelerated.jacobian(q, meta['ee'])
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        worst_jac = max(worst_jac, float(np.max(np.abs(a - b))))
    report[p.parent.name] = dict(status='PASS',configurations=64,max_fk_difference=worst_fk,max_jacobian_difference=worst_jac)
(ROOT / 'jit_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: 576 configurations, every link FK and tool Jacobian')
