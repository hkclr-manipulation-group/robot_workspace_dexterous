import hashlib
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from robot_workspace_dexterous.mesh_model import load_mesh_model


def model_files(tmp_path: Path, first=None, second=None, scale=1.0):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/><link name="tool"/>'
                    '<joint name="slide" type="prismatic"><parent link="base"/>'
                    '<child link="tool"/><axis xyz="1 0 0"/><origin xyz="2 0 0"/>'
                    '<limit lower="-5" upper="5" velocity="1" effort="1"/>'
                    '</joint></robot>', encoding="utf-8")
    geometries = [trimesh.creation.box() if first is None else first,
                  trimesh.creation.box() if second is None else second]
    links = {}
    for link, geometry in zip(["base", "tool"], geometries):
        path = tmp_path / (link + ".stl")
        geometry.export(path)
        links[link] = [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "scale": [scale]*3, "xyz": [0, 0, 0], "rpy": [0, 0, 0]}]
    manifest = tmp_path / "collision_meshes.yaml"
    manifest.write_text(yaml.safe_dump({"version": 1, "links": links,
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}), encoding="utf-8")
    return urdf, manifest


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    wp = pytest.importorskip("warp")
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA device required")
    return request.param


def check(tmp_path, q, first=None, second=None, ignores=None, device="cpu"):
    from robot_workspace_dexterous.mesh_collision import MeshCollisionChecker
    urdf, manifest = model_files(tmp_path, first, second)
    model = load_mesh_model(manifest, urdf, ignores or {})
    checker = MeshCollisionChecker(model, urdf, device=device)
    return checker.check_numpy(np.asarray(q, dtype=np.float32).reshape(-1, 1), ["slide"])


def test_separation_touching_penetration_and_reset(tmp_path, device):
    result = check(tmp_path, [0, -1, -1.5, 0], device=device)
    assert result.tolist() == [False, True, True, False]


def test_containment_without_surface_intersection(tmp_path, device):
    result = check(tmp_path, [-2, 0], first=trimesh.creation.box(extents=[2, 2, 2]),
                   second=trimesh.creation.box(extents=[.1, .1, .1]), device=device)
    assert result.tolist() == [True, False]


def test_reverse_containment_and_disconnected_components(tmp_path, device):
    far = trimesh.creation.box(extents=[.1, .1, .1])
    far.apply_translation([0, 10, 0])
    small = trimesh.creation.box(extents=[.1, .1, .1])
    result = check(tmp_path, [-2, 0], first=trimesh.util.concatenate([far, small]),
                   second=trimesh.creation.box(extents=[2, 2, 2]), device=device)
    assert result.tolist() == [True, False]


def test_coplanar_and_crossing_triangles(tmp_path, device):
    a = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    b = trimesh.Trimesh(vertices=[[.2, .2, -1], [.2, .2, 1], [.8, .2, 0]], faces=[[0, 1, 2]], process=False)
    assert check(tmp_path, [-2, 0], first=a, second=b, device=device).tolist() == [True, False]
    assert check(tmp_path, [-2, 0], first=a, second=a, device=device).tolist() == [True, False]


def test_ignore_pair_is_bidirectional(tmp_path, device):
    assert not check(tmp_path, [-2], ignores={"tool": ["base"]}, device=device).any()


def test_joint_contact_policy_is_explicit_and_handles_no_remaining_pairs(tmp_path, device):
    from robot_workspace_dexterous.joint_contacts import allow_joint_contacts
    from robot_workspace_dexterous.mesh_collision import MeshCollisionChecker
    urdf, manifest = model_files(tmp_path)
    strict = load_mesh_model(manifest, urdf, {})
    q = np.array([[-2.]], dtype=np.float32)
    assert MeshCollisionChecker(strict, urdf, device=device).check_numpy(q, ['slide']).tolist() == [True]
    permitted = allow_joint_contacts(strict, urdf)
    assert MeshCollisionChecker(permitted, urdf, device=device).check_numpy(q, ['slide']).tolist() == [False]
    assert strict.metadata['joint_contact_policy'] == 'strict'
    assert permitted.metadata['joint_contact_excluded_pairs'] == [['base', 'tool']]


def test_stl_scale_and_checksum(tmp_path):
    urdf, manifest = model_files(tmp_path, scale=.001)
    model = load_mesh_model(manifest, urdf, {})
    np.testing.assert_allclose(model.parts[0].vertices.max(axis=0), [.0005]*3)
    assert model.metadata["mode"] == "stl"
    assert model.metadata["total_triangles"] == 24
    with (tmp_path / "base.stl").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_mesh_model(manifest, urdf, {})


def test_open_mesh_is_reported_as_surface_only(tmp_path):
    triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    urdf, manifest = model_files(tmp_path, first=triangle)
    model = load_mesh_model(manifest, urdf, {})
    assert model.metadata["open_mesh_links"] == ["base"]


def test_nonfinite_joint_values_are_rejected(tmp_path, device):
    assert check(tmp_path, [np.nan, np.inf], device=device).all()


def test_hollow_closed_mesh_preserves_cavity(tmp_path, device):
    outer = trimesh.creation.box(extents=[4, 4, 4])
    inner = trimesh.creation.box(extents=[2, 2, 2])
    inner.invert()
    shell = trimesh.util.concatenate([outer, inner])
    assert check(tmp_path, [-2, -.5], first=shell,
                 second=trimesh.creation.box(extents=[.1]*3), device=device).tolist() == [False, True]


def test_reused_checker_and_pair_diagnostics(tmp_path, device):
    from robot_workspace_dexterous.mesh_collision import MeshCollisionChecker
    urdf, manifest = model_files(tmp_path)
    checker = MeshCollisionChecker(load_mesh_model(manifest, urdf, {}), urdf, device)
    for value, expected in [(-2, True), (0, False), (-1, True), (0, False)]:
        assert checker.check_numpy(np.array([[value]], dtype=np.float32), ["slide"])[0] == expected
    frequencies = checker.pair_frequencies(np.array([[-2], [0]], dtype=np.float32), ["slide"])
    assert frequencies == [{"links": ["base", "tool"], "colliding_samples": 1, "fraction": .5}]
    assert not checker.check_numpy(np.zeros((1, 1), dtype=np.float32), ["slide"])[0]


def test_cuda_graph_replay_resets_collision_flags(tmp_path):
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")
    from robot_workspace_dexterous.mesh_collision import MeshCollisionChecker
    urdf, manifest = model_files(tmp_path)
    checker = MeshCollisionChecker(load_mesh_model(manifest, urdf, {}), urdf, "cuda:0")
    checker.check_numpy(np.zeros((1, 1), dtype=np.float32), ["slide"])
    q = wp.zeros((1, 1), dtype=wp.float32, device=checker.device)
    with wp.ScopedCapture(device=checker.device) as capture:
        checker._launch(q, checker.active)
    for value, expected in [(-2, True), (0, False), (-1, True), (0, False)]:
        q.assign(np.array([[value]], dtype=np.float32))
        wp.capture_launch(capture.graph)
        assert bool(checker.colliding.numpy()[0]) == expected


def test_cached_geometry_still_checks_every_manifest_digest(tmp_path):
    urdf, manifest = model_files(tmp_path)
    raw = yaml.safe_load(manifest.read_text())
    raw["links"]["tool"] = [dict(raw["links"]["base"][0], sha256="bad")]
    manifest.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="checksum"):
        load_mesh_model(manifest, urdf, {})


def test_full_tree_fk_with_tilted_axis_origin_and_mimic(tmp_path, device):
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from robot_workspace_dexterous.diagnostics import forward_kinematics
    from robot_workspace_dexterous.mesh_collision import MeshCollisionChecker
    urdf, manifest = model_files(tmp_path)
    root = ET.parse(urdf).getroot()
    joint = root.find("joint")
    joint.set("type", "revolute")
    joint.find("axis").set("xyz", "1 2 3")
    joint.find("origin").set("rpy", ".2 -.5 .3")
    ET.SubElement(root, "link", name="other_branch")
    other = ET.SubElement(root, "joint", name="mimic", type="prismatic")
    ET.SubElement(other, "parent", link="base")
    ET.SubElement(other, "child", link="other_branch")
    ET.SubElement(other, "axis", xyz="0 1 1")
    ET.SubElement(other, "mimic", joint="slide", multiplier="-.5", offset=".2")
    urdf.write_bytes(ET.tostring(root))
    raw = yaml.safe_load(manifest.read_text())
    raw["urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    manifest.write_text(yaml.safe_dump(raw))
    checker = MeshCollisionChecker(load_mesh_model(manifest, urdf, {}), urdf, device)
    q = np.array([[-.7], [.2], [1.1]], dtype=np.float32)
    checker.check_numpy(q, ["slide"])
    expected = forward_kinematics(root, {"slide": q[:, 0], "mimic": q[:, 0]})
    actual = checker.transforms.numpy()
    for i, link in enumerate(checker.links):
        np.testing.assert_allclose(actual[:, i, :3], expected[link][:, :3, 3], atol=3e-6)
        np.testing.assert_allclose(Rotation.from_quat(actual[:, i, 3:]).as_matrix(),
                                   expected[link][:, :3, :3], atol=3e-6)


def test_stl_config_and_cpu_validation_require_no_sphere_file(tmp_path, capsys):
    from robot_workspace_dexterous.config import load_config
    from robot_workspace_dexterous.cli import main
    urdf, manifest = model_files(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"robot": {"urdf": urdf.name,
        "collision_meshes": manifest.name, "base_link": "base", "ee_links": ["tool"]},
        "grid": {"x_range": [0, 1], "y_range": [0, 1], "height": 0, "resolution": .1}}))
    config = load_config(path)
    assert config.collision_spheres_path is None
    assert config.collision_backend == "stl"
    main(["--config", str(path), "--validate-only"])
    output = capsys.readouterr().out
    assert "collision mode=stl" in output
    assert '"joint_contact_policy": "strict"' in output
    main(["--config", str(path), "--validate-only", "--allow-joint-contacts"])
    output = capsys.readouterr().out
    assert '"joint_contact_policy": "allow_rigid_and_joint_neighbors"' in output
    assert '"collision_pairs": 0' in output
