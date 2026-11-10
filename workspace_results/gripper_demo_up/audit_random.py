exec(open('/home/hkclr-manipulation-spark1/rt_ws/robot_workspace_dexterous/workspace_results/gripper_demo/inspect_mounts.py').read().split('for i in range')[0])
out=Path(__file__).parent
from collections import Counter
records=[]
for i in range(1,5):
 c=load_config(out/'configs'/f'P2AB-{i}_open.yaml');k=RobotKinematics(c.urdf_path,c.base_link);m=load_mesh_model(c.collision_meshes_path,c.urdf_path,c.self_collision_ignore);checker=MeshCollisionChecker({p.link:trimesh.Trimesh(p.vertices,p.faces,process=False) for p in m.parts});ign=[(a,b) for a,bs in c.self_collision_ignore.items() for b in bs]
 counts=Counter();free=0
 for q in np.random.default_rng(42).uniform(k.lower,k.upper,(200,len(k.joint_names))):
  hits=checker.collisions(k.forward(q),ignores=ign)
  free+=not hits
  counts.update(tuple(sorted((h.first,h.second))) for h in hits)
 record=dict(model=f'P2AB-{i}',samples=200,collision_free=free,pairs=[dict(pair=p,count=n) for p,n in counts.most_common(8)])
 records.append(record);print(record,flush=True)
(out/'random_collision_audit.json').write_text(json.dumps(records,indent=2))
