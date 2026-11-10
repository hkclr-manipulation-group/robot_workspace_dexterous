from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.mesh_model import load_mesh_model
from collision_shpere_generation.kinematics import RobotKinematics
out=Path(__file__).parent;names=['spark2_v2_2','spark2_v1']+[f'P2AB-{i}' for i in range(1,5)]
fig=plt.figure(figsize=(18,9),layout='constrained')
for idx,name in enumerate(names):
 root=out if name.startswith('spark2_') else out.parent/'gripper_demo_up'
 cfg=load_config(root/'configs'/f'{name}_open.yaml');model=load_mesh_model(cfg.collision_meshes_path,cfg.urdf_path,cfg.self_collision_ignore);k=RobotKinematics(cfg.urdf_path,cfg.base_link);poses=k.forward(np.zeros(6))
 ax=fig.add_subplot(2,3,idx+1,projection='3d')
 for part in model.parts:
  vertices=part.vertices;faces=part.faces
  if len(faces)>2000:
   _,inverse=np.unique(np.round(vertices/.001).astype(int),axis=0,return_inverse=True);counts=np.bincount(inverse);vertices=np.column_stack([np.bincount(inverse,weights=vertices[:,j])/counts for j in range(3)]);faces=inverse[faces];faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])]
  T=poses[part.link];v=vertices@T[:3,:3].T+T[:3,3]
  ax.add_collection3d(Poly3DCollection(v[faces],facecolors='#394b59' if part.link.startswith('gripper') else '#5d91aa',shade=True,linewidth=0))
 ax.set(xlim=(-.5,.5),ylim=(-.5,.5),zlim=(0,1.2),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=name);ax.set_box_aspect((1,1,1.2));ax.view_init(18,-60)
fig.suptitle('Six models at zero joint positions | normalized arm chain length 1 m + same Spark2 gripper\nSame camera and axis scale | display meshes clustered at 1 mm; collision uses full meshes',fontsize=16)
fig.savefig(out/'models_zero_pose.png',dpi=160)
