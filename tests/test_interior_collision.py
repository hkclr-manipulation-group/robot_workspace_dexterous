from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.curobo_solver import _load_collision_spheres, build_collision_robots
from robot_workspace_dexterous.diagnostics import minimum_sphere_gaps

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('original_newline', [b'\n', b'\r\n'], ids=['report-lf', 'report-crlf'])
@pytest.mark.parametrize('current_newline', [b'\n', b'\r\n'], ids=['file-lf', 'file-crlf'])
def test_checksum_accepts_newlines_but_rejects_changed_spheres(tmp_path, original_newline, current_newline):
    import hashlib
    import json
    data=b'collision_spheres:\n  arm:\n    - center: [0, 0, 0]\n      radius: 0.01\n'
    path=tmp_path/'collision_spheres_interior.yaml'
    path.write_bytes(data.replace(b'\n',current_newline))
    report={'mode':'interior','complete':True,'sphere_sha256':hashlib.sha256(data.replace(b'\n',original_newline)).hexdigest()}
    path.with_suffix('.json').write_text(json.dumps(report))
    assert _load_collision_spheres(str(path))['arm'][0]['radius']==.01
    path.write_bytes(path.read_bytes().replace(b'0.01',b'0.02'))
    with pytest.raises(ValueError,match='checksum'):_load_collision_spheres(str(path))
    path.write_bytes(data.replace(b'\n',current_newline))
    report['complete']=False
    path.with_suffix('.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='incomplete'):_load_collision_spheres(str(path))


def test_blocked_gaps_match_dense_reference():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(11, 71, 3)), rng.normal(size=(11, 67, 3))
    ra, rb = rng.uniform(.01, .2, 71), rng.uniform(.01, .2, 67)
    expected = (np.linalg.norm(a[:, :, None]-b[:, None, :], axis=-1)
                -ra[None, :, None]-rb[None, None, :]).min(axis=(1, 2))
    np.testing.assert_allclose(minimum_sphere_gaps(a, ra, b, rb), expected)


def test_changed_interior_file_rejected(tmp_path):
    source = PROJECT/'models/spark2_v2/collision_spheres_interior.yaml'
    destination = tmp_path/source.name
    shutil.copyfile(source, destination)
    shutil.copyfile(source.with_suffix('.json'), destination.with_suffix('.json'))
    with destination.open('a') as stream:
        stream.write('\n# changed after generation\n')
    with pytest.raises(ValueError, match='checksum'):
        _load_collision_spheres(str(destination))


def test_curobo_receives_complete_interior_set_without_expansion(monkeypatch, tmp_path):
    config = load_config(PROJECT/'configs/dual_v2_1_no_gripper.yaml')
    calls = []

    class RobotCfg:
        @staticmethod
        def create(data, load_collision_spheres):
            assert load_collision_spheres
            calls.append(data['robot_cfg']['kinematics'])
            return data

    monkeypatch.setitem(sys.modules, 'curobo._src.types.robot', SimpleNamespace(RobotCfg=RobotCfg))
    robots = build_collision_robots(str(config.urdf_path), str(config.collision_spheres_path),
                                    config.base_link, config.ee_links, config.self_collision_ignore,
                                    normalized_urdf_path=str(tmp_path/'normalized.urdf'))
    expected = _load_collision_spheres(str(config.collision_spheres_path))
    assert set(robots) == set(config.ee_links)
    assert len(calls) == 2
    for data in calls:
        assert data['collision_spheres'] == expected
        assert set(data['collision_link_names']) == set(expected)
        assert data['collision_sphere_buffer'] == 0
        assert data['self_collision_buffer'] == {}
        assert data['self_collision_ignore'] == config.self_collision_ignore
