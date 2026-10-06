from dataclasses import replace
import numpy as np

from robot_workspace_dexterous.config import Config
from robot_workspace_dexterous.cpu_solver import compute_cpu_workspace
from robot_workspace_dexterous.sampling import DexterousWorkspace, shared_orientation_workspace


def cartesian_robot(path):
    links = ['base','x','y','z','roll','pitch','tool']
    text = '<robot name="cartesian">' + ''.join(f'<link name="{link}"/>' for link in links)
    for i,axis in enumerate(['1 0 0','0 1 0','0 0 1']*2):
        kind = 'prismatic' if i < 3 else 'revolute'
        text += (f'<joint name="j{i}" type="{kind}"><parent link="{links[i]}"/>'
                 f'<child link="{links[i+1]}"/><axis xyz="{axis}"/>'
                 '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>')
    path.write_text(text+'</robot>',encoding='utf-8')


def test_cpu_ik_collision_and_metrics(tmp_path):
    import yaml
    urdf = tmp_path/'robot.urdf'
    cartesian_robot(urdf)
    spheres = tmp_path/'spheres.yaml'
    spheres.write_text(yaml.safe_dump({'collision_spheres': {
        'base':[{'center':[0,0,0],'radius':.1}],
        'tool':[{'center':[0,0,0],'radius':.1}]}}),encoding='utf-8')
    config = Config(urdf,spheres,'base',('tool',),{},(0,.3),(0,0),np.array([0]),.3,
        np.array([[1,0,0,0]]),2,32,.001,.01,True,1.,
        ((-.1,.4),(-.1,.1),(-.1,.1)),(0,0,0),(0,0,0),backend='cpu')
    result = compute_cpu_workspace(config,'tool')
    assert result.reachable_orientations.tolist() == [0,1]
    np.testing.assert_allclose(result.manipulability_mean,[0,1],atol=.001)
    np.testing.assert_allclose(result.condition_number_max[1],1,atol=.001)
    assert result.orientation_success.tolist() == [[False],[True]]
    no_collision = compute_cpu_workspace(replace(config,self_collision=False),'tool')
    assert no_collision.reachable_orientations.tolist() == [1,1]


def test_shared_requires_the_same_orientation_and_packs_mask(tmp_path):
    position=np.zeros((1,3))
    a=DexterousWorkspace(position,np.array([.5]),np.array([1]),2,
                        orientation_success=np.array([[True,False]]))
    b=DexterousWorkspace(position,np.array([.5]),np.array([1]),2,
                        orientation_success=np.array([[False,True]]))
    shared=shared_orientation_workspace([a,b])
    assert shared.reachable_orientations.tolist() == [0]
    a.save(str(tmp_path/'mask.npz'))
    with np.load(tmp_path/'mask.npz') as saved:
        mask=np.unpackbits(saved['orientation_success_bits'],axis=1,bitorder='little')[:,:2]
        np.testing.assert_array_equal(mask,a.orientation_success)


def test_cpu_stl_checker_preserves_closed_parts_and_pair_policy(tmp_path):
    import trimesh
    from collision_shpere_generation.kinematics import RobotKinematics
    from robot_workspace_dexterous.mesh_model import MeshPart, MeshModel
    from robot_workspace_dexterous.cpu_solver import mesh_collision_callback
    urdf=tmp_path/'robot.urdf'
    cartesian_robot(urdf)
    geometries=[trimesh.creation.box(extents=[1,1,1]),
                trimesh.creation.box(extents=[1,1,1]),
                trimesh.creation.box(extents=[.1,.1,.1])]
    geometries[1].faces=geometries[1].faces[:-1]
    geometries[1].apply_translation([3,0,0])
    parts=[MeshPart(link,mesh.vertices,mesh.faces,np.empty((0,3)),mesh.is_watertight)
           for link,mesh in zip(['base','base','tool'],geometries)]
    kinematics=RobotKinematics(urdf,'base')
    model=MeshModel(parts,np.array([[0,2],[1,2]]),{'contact_tolerance_m':1e-6})
    check=mesh_collision_callback(model,kinematics)
    assert check(np.zeros(6))  # Fully enclosed, no surface intersection.
    assert not check(np.array([.8,0,0,0,0,0]))
    model.pairs=np.empty((0,2),dtype=int)
    ignored=mesh_collision_callback(model,kinematics)
    assert not ignored(np.zeros(6))
