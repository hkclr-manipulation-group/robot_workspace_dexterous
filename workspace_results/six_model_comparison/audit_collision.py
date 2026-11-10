from pathlib import Path
import json
from collections import Counter
import numpy as np,trimesh
from collision_shpere_generation.kinematics import RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.mesh_model import load_mesh_model
out=Path(__file__).parent;records=[]
for state in ['bare','open','holding']:
 c=load_config(out/'configs'/f'spark2_v2_2_{state}.yaml');k=RobotKinematics(c.urdf_path,c.base_link);m=load_mesh_model(c.collision_meshes_path,c.urdf_path,c.self_collision_ignore)
 checker=MeshCollisionChecker({p.link:trimesh.Trimesh(p.vertices,p.faces,process=False) for p in m.parts});ignore=[(a,b) for a,bs in c.self_collision_ignore.items() for b in bs];counts=Counter();free=0
 for q in np.random.default_rng(42).uniform(k.lower,k.upper,(200,6)):
  hits=checker.collisions(k.forward(q),ignores=ignore);free+=not hits;counts.update(tuple(sorted((h.first,h.second))) for h in hits)
 records.append(dict(state=state,samples=200,collision_free=free,pairs=[dict(pair=p,count=n) for p,n in counts.most_common(8)],explicit_ignored_pairs=ignore));print(records[-1],flush=True)
(out/'spark_collision_audit.json').write_text(json.dumps(records,indent=2))
