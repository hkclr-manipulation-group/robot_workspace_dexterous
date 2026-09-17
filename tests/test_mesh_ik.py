import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from robot_workspace_dexterous.curobo_solver import compute_dexterous_workspace, select_mesh_candidates


def test_all_seeds_checked_and_failed_pose_not_accepted():
    positions = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    pose_success = torch.tensor([[True, True, False], [True, True, True], [False, False, False]])
    collision = torch.tensor([[True, False, False], [True, True, True], [False, False, False]])
    class Checker:
        def check_torch(self, q, names, valid):
            assert q is positions
            assert torch.equal(valid, pose_success)
            return collision
    result = SimpleNamespace(js_solution=SimpleNamespace(position=positions, joint_names=["a", "b"]),
                             success=pose_success)
    valid, chosen = select_mesh_candidates(result, Checker())
    assert valid.tolist() == [True, False, False]
    torch.testing.assert_close(chosen[0], positions[0, 1])


@pytest.mark.parametrize("legacy_curobo", [False, True])
def test_workspace_uses_pose_only_ik_then_mesh_checked_seed(monkeypatch, legacy_curobo):
    calls = {}
    class Checker:
        links = ["base", "tool", "other_arm"]
        def check_torch(self, q, names, valid):
            calls["checked_seeds"] = q.shape[1]
            return torch.tensor([[True, False]])
    class RobotCfg:
        @staticmethod
        def create(data, load_collision_spheres):
            if legacy_curobo:
                raise TypeError("unexpected keyword argument 'kinematic_link_names'")
            calls["robot"] = data
            assert load_collision_spheres is False
            return SimpleNamespace(kinematics=SimpleNamespace())
    class IKCfg:
        @staticmethod
        def create(**kwargs):
            calls["config"] = kwargs
            return SimpleNamespace(device_cfg=SimpleNamespace(device="cpu", dtype=torch.float32))
    class IK:
        def __init__(self, config):
            pass
        def solve_pose(self, goal_tool_poses, return_seeds):
            assert return_seeds == 2
            q = torch.zeros(1, 2, 6)
            q[:, 1] = 1
            return SimpleNamespace(success=torch.ones(1, 2, dtype=torch.bool),
                                    js_solution=SimpleNamespace(position=q, joint_names=list("abcdef")))
    class Kinematics:
        def __init__(self, *args, **kwargs):
            pass
        def compute_kinematics(self, js):
            torch.testing.assert_close(js.position, torch.ones(1, 6))
            return SimpleNamespace(tool_jacobians=torch.eye(6).reshape(1, 1, 6, 6))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "curobo.inverse_kinematics", SimpleNamespace(
        InverseKinematics=IK, InverseKinematicsCfg=IKCfg))
    monkeypatch.setitem(sys.modules, "curobo._src.types.robot", SimpleNamespace(RobotCfg=RobotCfg))
    monkeypatch.setitem(sys.modules, "curobo.kinematics", SimpleNamespace(
        Kinematics=Kinematics, KinematicsCfg=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(
        Pose=lambda **kw: SimpleNamespace(**kw),
        GoalToolPose=SimpleNamespace(from_poses=lambda poses, num_goalset: poses),
        JointState=SimpleNamespace(from_position=lambda position, joint_names:
                                   SimpleNamespace(position=position, joint_names=joint_names))))
    if legacy_curobo:
        with pytest.raises(RuntimeError) as error:
            compute_dexterous_workspace("unused.urdf", "base", "tool", (0, 0), (0, 0),
                np.array([0]), .1, np.array([[1, 0, 0, 0]]), num_seeds=2, mesh_checker=Checker())
        assert sys.executable in str(error.value)
        assert "Loaded cuRobo robot module:" in str(error.value)
        assert "python -m pip install -e ../curobo --no-build-isolation" in str(error.value)
        return
    result = compute_dexterous_workspace("unused.urdf", "base", "tool", (0, 0), (0, 0),
        np.array([0]), .1, np.array([[1, 0, 0, 0]]), num_seeds=2, mesh_checker=Checker())
    assert result.reachable_orientations.tolist() == [1]
    assert calls["checked_seeds"] == 2
    assert calls["config"]["self_collision_check"] is False
    assert calls["config"]["load_collision_spheres"] is False
    assert calls["robot"]["robot_cfg"]["kinematics"]["kinematic_link_names"] == Checker.links
