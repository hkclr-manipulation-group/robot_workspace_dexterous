exec(open('/home/hkclr-manipulation-spark1/rt_ws/robot_workspace_dexterous/workspace_results/gripper_demo/inspect_mounts.py').read().split('for i in range')[0])
import xml.etree.ElementTree as ET
from collections import Counter
folder=out/'rotation_audit';folder.mkdir(exist_ok=True)
records=[]
for state in ['open','holding']:
 for angle in [0,90]:
  c=load_config(out/'configs'/f'P2AB-3_{state}.yaml')
  tree=ET.parse(c.urdf_path);tree.find(".//joint[@name='mount_gripper']/origin").set('rpy',f'0 0 {np.deg2rad(angle)}')
  urdf=folder/f'P2AB-3_{state}_z{angle}.urdf';tree.write(urdf)
  k=RobotKinematics(urdf,c.base_link);m=load_mesh_model(c.collision_meshes_path,c.urdf_path,c.self_collision_ignore)
  checker=MeshCollisionChecker({p.link:trimesh.Trimesh(p.vertices,p.faces,process=False) for p in m.parts})
  ign=[(a,b) for a,bs in c.self_collision_ignore.items() for b in bs]
  counts=Counter();free=0
  for q in np.random.default_rng(42).uniform(k.lower,k.upper,(1000,len(k.joint_names))):
   hits=checker.collisions(k.forward(q),ignores=ign);free+=not hits
   counts.update(tuple(sorted((h.first,h.second))) for h in hits)
  wrist=0
  for q6 in np.linspace(-3.14,3.14,361):
   q=np.zeros(len(k.joint_names));q[-1]=q6
   hits=checker.collisions(k.forward(q),ignores=ign)
   wrist+=any(set([h.first,h.second])==set(['L555','gripper_base_link']) for h in hits)
  record=dict(model='P2AB-3',state=state,z_angle_deg=angle,samples=1000,collision_free=free,wrist_sweep_collision=wrist,wrist_sweep_samples=361,pairs=[dict(pair=p,count=n) for p,n in counts.most_common(5)])
  records.append(record);print(record,flush=True)
(folder/'results.json').write_text(json.dumps(records,indent=2))
