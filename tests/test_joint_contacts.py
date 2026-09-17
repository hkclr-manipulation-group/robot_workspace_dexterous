from itertools import combinations
from types import SimpleNamespace

import numpy as np

from robot_workspace_dexterous.joint_contacts import allow_joint_contacts
from robot_workspace_dexterous.mesh_model import MeshModel


def make_model(tmp_path, joints, geometry_links):
    links = sorted({name for a, b, kind, limits in joints for name in (a, b)})
    xml = '<robot name="test">' + ''.join(f'<link name="{name}"/>' for name in links)
    for i, (a, b, kind, limits) in enumerate(joints):
        xml += (f'<joint name="j{i}" type="{kind}"><parent link="{a}"/>'
                f'<child link="{b}"/>{limits}</joint>')
    path = tmp_path / 'robot.urdf'
    path.write_text(xml + '</robot>', encoding='utf-8')
    model = MeshModel([SimpleNamespace(link=name) for name in geometry_links],
                      np.array(list(combinations(range(len(geometry_links)), 2)), dtype=np.int32),
                      {'joint_contact_policy': 'strict', 'joint_contact_excluded_pairs': []})
    return model, path


def test_fixed_adapters_preserve_neighbor_relation_without_ignoring_remote_links(tmp_path):
    joints = [('base', 'mount', 'fixed', ''), ('mount', 'upper', 'revolute', ''),
              ('upper', 'adapter', 'fixed', ''), ('adapter', 'forearm', 'revolute', ''),
              ('base', 'other_arm', 'revolute', '')]
    model, path = make_model(tmp_path, joints, ['base', 'mount', 'upper', 'forearm', 'other_arm'])
    filtered = allow_joint_contacts(model, path)
    remaining = {frozenset((filtered.parts[i].link, filtered.parts[j].link)) for i, j in filtered.pairs}
    assert frozenset(('base', 'forearm')) in remaining
    assert frozenset(('upper', 'other_arm')) in remaining
    assert frozenset(('forearm', 'other_arm')) in remaining
    assert frozenset(('mount', 'forearm')) in remaining
    assert len(remaining) == 4
    assert len(model.pairs) == 10  # The strict source model is unchanged.
    assert model.metadata['joint_contact_policy'] == 'strict'
    assert len(filtered.metadata['joint_contact_excluded_pairs']) == 6


def test_zero_range_joint_is_rigid_but_continuous_joint_is_not(tmp_path):
    limits = '<limit lower="0.2" upper="0.2"/>'
    joints = [('base', 'locked', 'revolute', limits), ('locked', 'wrist', 'continuous', limits),
              ('wrist', 'tool', 'revolute', '')]
    model, path = make_model(tmp_path, joints, ['base', 'locked', 'wrist', 'tool'])
    filtered = allow_joint_contacts(model, path)
    remaining = {frozenset((filtered.parts[i].link, filtered.parts[j].link)) for i, j in filtered.pairs}
    assert remaining == {frozenset(('base', 'tool')), frozenset(('locked', 'tool'))}


def test_existing_exclusion_stays_excluded_and_empty_pairs_keep_shape(tmp_path):
    model, path = make_model(tmp_path, [('base', 'tool', 'fixed', '')], ['base', 'tool'])
    result = allow_joint_contacts(model, path)
    assert result.pairs.shape == (0, 2)
    assert result.metadata['joint_contact_excluded_pairs'] == [['base', 'tool']]
    model.pairs = np.empty((0, 2), dtype=np.int32)
    result = allow_joint_contacts(model, path)
    assert result.pairs.shape == (0, 2)
    assert result.metadata['joint_contact_excluded_pairs'] == []


def test_mimic_motion_is_not_treated_as_rigid_from_zero_limits(tmp_path):
    mimic = '<limit lower="0" upper="0"/><mimic joint="j0" multiplier="1"/>'
    joints = [('base', 'upper', 'revolute', ''), ('upper', 'tip', 'revolute', mimic)]
    model, path = make_model(tmp_path, joints, ['base', 'upper', 'tip'])
    filtered = allow_joint_contacts(model, path)
    assert filtered.pairs.tolist() == [[0, 2]]
