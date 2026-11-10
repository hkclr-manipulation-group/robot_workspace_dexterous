from pathlib import Path
import numpy as np,trimesh,json
from collision_shpere_generation.kinematics import RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.mesh_model import load_mesh_model
out=Path(__file__).parent
for i in range(1,5):
 c=load_config(out/'configs'/f'P2AB-{i}_open.yaml');k=RobotKinematics(c.urdf_path,c.base_link);m=load_mesh_model(c.collision_meshes_path,c.urdf_path,c.self_collision_ignore);checker=MeshCollisionChecker({p.link:trimesh.Trimesh(p.vertices,p.faces,process=False) for p in m.parts});ign=[(a,b) for a,bs in c.self_collision_ignore.items() for b in bs]
 r=k.solve([.4,-.22,.28],'grasp_tcp',[0,1,0,0],seeds=32)
 print(i,'IK without collision',r.success)
 if r.success:print([(h.first,h.second,h.kind) for h in checker.collisions(k.forward(r.positions),ignores=ign)])
