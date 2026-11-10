from pathlib import Path
import json
import numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collision_shpere_generation.kinematics import RobotKinematics
from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.mesh_model import load_mesh_model
out=Path(__file__).parent
cache={}
for i in range(1,5):
 for state in ['open','holding']:
  c=load_config(out/'configs'/f'P2AB-{i}_{state}.yaml');k=RobotKinematics(c.urdf_path,c.base_link);m=load_mesh_model(c.collision_meshes_path,c.urdf_path,c.self_collision_ignore)
  for part in m.parts:
   if len(part.faces)>2000:
    bins=np.round(part.vertices/.001).astype(np.int64)
    _,inverse=np.unique(bins,axis=0,return_inverse=True)
    totals=np.bincount(inverse);v=np.column_stack([np.bincount(inverse,weights=part.vertices[:,d])/totals for d in range(3)])
    f=inverse[part.faces];good=(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2]);part.vertices=v;part.faces=f[good]
  cache[i,state]=(k,m)
def geometry(i,frame):
 k,m=cache[i,frame['state']];poses=k.forward(frame['q'])
 if 'jaw_fraction' in frame:
  op=cache[i,'open'][0].forward(frame['q']);cl=cache[i,'holding'][0].forward(frame['q']);f=frame['jaw_fraction']
  for name in ['gripper_L1','gripper_L2']:poses[name][:3,3]=(1-f)*op[name][:3,3]+f*cl[name][:3,3]
 pieces=[]
 for part in m.parts:
  t=poses[part.link];v=part.vertices@t[:3,:3].T+t[:3,3]
  # Display fewer triangles only for rendering. Collision uses every triangle.
  faces=part.faces
  color='#ec9735' if part.link=='grasp_tcp' else '#213c50' if part.link.startswith('gripper') else '#78b6ce'
  pieces.append((v[faces],color))
 if frame['state']=='open':
  cube=trimesh.creation.box(extents=[.03]*3);cube.apply_translation(frame['block']);pieces.append((cube.triangles,'#ec9735'))
 return pieces,poses['grasp_tcp'][:3,3]
def setup(ax,row):
 for center in [row['pick'],row['place']]:
  mesh=trimesh.creation.box(extents=[.18,.18,.08]);mesh.apply_translation([*center[:2],.102]);ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor='#ccd2d5',alpha=.7,linewidths=.1,edgecolor='#777'))
 if row['obstacle']:
  mesh=trimesh.creation.box(extents=[.18,.08,.28]);mesh.apply_translation([.4,0,.14]);ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor='#c55359',alpha=.6))
 ax.scatter(*row['pick'],c='green',s=35,marker='x');ax.scatter(*row['place'],c='purple',s=35,marker='x')
 ax.set(xlim=(-.6,.8),ylim=(-.6,.8),zlim=(0,1.15),xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]');ax.set_box_aspect((1.4,1.4,1.15));ax.view_init(22,-48)
for task in ['front','side','obstacle']:
 fig=plt.figure(figsize=(11,9),layout='constrained');rows=[];artists=[];axes=[];tracks=[]
 for i in range(1,5):
  row=json.loads((out/f'P2AB-{i}_{task}_trajectory.json').read_text());rows.append(row)
  ax=fig.add_subplot(2,2,i,projection='3d');axes.append(ax);setup(ax,row);artists.append([]);tracks.append([])
 def update(t):
  for idx,(row,ax) in enumerate(zip(rows,axes)):
   for artist in artists[idx]:artist.remove()
   artists[idx]=[]
   frames=row['frames']
   if frames:frame=frames[min(int(t/39*(len(frames)-1)),len(frames)-1)]
   else:frame=dict(q=cache[idx+1,'open'][0].zero_positions(),state='open',stage='No accepted approach IK',block=row['pick'])
   pieces,tip=geometry(idx+1,frame)
   for triangles,color in pieces:
    art=Poly3DCollection(triangles,facecolors=color,alpha=1.,shade=True);ax.add_collection3d(art);artists[idx].append(art)
   status='PASS' if row['success'] else 'STOP'
   ax.set_title(f'P2AB-{idx+1}: {status}\n{frame["stage"]}',fontsize=11)
  return []
 fig.suptitle(f'{task.upper()} pick/place | equal-length arms + same Spark2 gripper\nKinematic mesh-checked demo, not contact dynamics | playback normalized per model; render mesh clustered at 1 mm',fontsize=12)
 update(22);fig.savefig(out/f'{task}_comparison.png',dpi=140)
 animation=FuncAnimation(fig,update,frames=40,interval=150,blit=False)
 animation.save(out/f'{task}_comparison.gif',writer=PillowWriter(fps=7),dpi=90)
 plt.close(fig);print(task,'rendered',flush=True)
