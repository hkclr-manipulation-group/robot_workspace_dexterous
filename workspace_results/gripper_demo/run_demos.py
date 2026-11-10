from pathlib import Path
import json,time
import numpy as np,trimesh
from collision_shpere_generation.kinematics import RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.mesh_model import load_mesh_model
out=Path(__file__).parent
TASKS=[('front',[.40,-.22,.16],[.40,.22,.16],.40,False),('side',[.25,.40,.16],[-.25,.40,.16],.40,False),('obstacle',[.40,-.25,.16],[.40,.25,.16],.48,True)]
def transform(x):
 t=np.eye(4);t[:3,3]=x;return t
summary=[]
for i in range(1,5):
 name=f'P2AB-{i}';configs={s:load_config(out/'configs'/f'{name}_{s}.yaml') for s in ['open','holding']}
 kin={s:RobotKinematics(c.urdf_path,c.base_link) for s,c in configs.items()}
 models={s:load_mesh_model(c.collision_meshes_path,c.urdf_path,c.self_collision_ignore) for s,c in configs.items()}
 for task,pick,place,height,obstacle in TASKS:
  started=time.monotonic();pick=np.array(pick);place=np.array(place)
  env={'pick_pad':trimesh.creation.box(extents=[.18,.18,.08]),'place_pad':trimesh.creation.box(extents=[.18,.18,.08])}
  ep={'pick_pad':transform([pick[0],pick[1],.102]),'place_pad':transform([place[0],place[1],.102])}
  # Pad top .142, block bottom .145: 3 mm geometric clearance, no contact physics.
  if obstacle:env['barrier']=trimesh.creation.box(extents=[.18,.08,.28]);ep['barrier']=transform([.4,0,.14])
  checkers={};excluded={}
  for state in ['open','holding']:
   meshes={p.link:trimesh.Trimesh(p.vertices,p.faces,process=False) for p in models[state].parts}
   meshes.update(env)
   if state=='open':meshes['workpiece']=trimesh.creation.box(extents=[.03]*3)
   checkers[state]=MeshCollisionChecker(meshes,containment=True)
   excluded[state]={(a,b) for a,bs in configs[state].self_collision_ignore.items() for b in bs}
   envnames=list(env)
   excluded[state].update((a,b) for j,a in enumerate(envnames) for b in envnames[j+1:])
  frames=[];checks=0;reason=None;q=None;block_at=pick.copy();worst_pos=0.;worst_rot=0.
  def poses(q,state):
   p=kin[state].forward(q);p.update(ep)
   if state=='open':p['workpiece']=transform(block_at)
   return p
  def collision(q,state):
   global checks
   checks+=1
   return checkers[state].check(poses(q,state),ignores=excluded[state])
  stages=[('approach','open',pick+[0,0,.12]),('descend','open',pick),('lift','holding',np.array([*pick[:2],height])),('transfer','holding',np.array([*place[:2],height])),('place','holding',place),('retreat','open',place+[0,0,.12])]
  previous=None
  previous_state='open'
  for stage,state,target in stages:
   if stage=='retreat':block_at=place.copy()
   if q is not None and state != previous_state:
    open_poses=kin['open'].forward(q);closed_poses=kin['holding'].forward(q)
    for fraction in np.linspace(0,1,15) if state=='holding' else np.linspace(1,0,15):
     transition=poses(q,'holding')
     for finger in ['gripper_L1','gripper_L2']:
      transition[finger][:3,3]=(1-fraction)*open_poses[finger][:3,3]+fraction*closed_poses[finger][:3,3]
     checks+=1
     if checkers['holding'].check(transition,ignores=excluded['holding']):
      reason=f'{stage}: jaw motion collision';break
     frames.append(dict(q=q.tolist(),state='holding',stage='close jaws' if state=='holding' else 'open jaws',jaw_fraction=float(fraction),block=block_at.tolist()))
    if reason:break
   previous_state=state
   n=1 if previous is None else max(1,int(np.ceil(np.linalg.norm(target-previous)/.035)))
   for k in range(1,n+1):
    point=target if previous is None else previous+(target-previous)*k/n
    solved=kin[state].solve(point,'grasp_tcp',[0,1,0,0],seeds=12,max_iterations=120,position_tolerance=.003,orientation_tolerance=.05,collision=lambda x:collision(x,state),warm_start=q,seed=100+i*10+k)
    if not solved.success:reason=f'{stage}: no collision-free IK found';break
    nextq=solved.positions
    if q is not None:
     samples=max(1,int(np.ceil(np.max(abs(nextq-q))/.035)))
     for f in np.linspace(0,1,samples+1)[1:]:
      candidate=q+(nextq-q)*f
      if collision(candidate,state):reason=f'{stage}: interpolated joint path collision';break
      frames.append(dict(q=candidate.tolist(),state=state,stage=stage,block=block_at.tolist()))
     if reason:break
    else:frames.append(dict(q=nextq.tolist(),state=state,stage=stage,block=block_at.tolist()))
    q=nextq;worst_pos=max(worst_pos,solved.position_error);worst_rot=max(worst_rot,solved.orientation_error)
   if reason:break
   previous=target.copy()
  travel=float(sum(np.linalg.norm(np.array(b['q'])-a['q']) for a,b in zip(frames,frames[1:])))
  row=dict(model=name,task=task,success=reason is None,failure=reason,checked_configurations=checks,frames=frames,joint_path_length_rad=travel,max_position_error_m=worst_pos,max_orientation_error_rad=worst_rot,elapsed_s=time.monotonic()-started,pick=pick.tolist(),place=place.tolist(),height=height,obstacle=obstacle)
  (out/f'{name}_{task}_trajectory.json').write_text(json.dumps(row,indent=2)+'\n');summary.append({k:v for k,v in row.items() if k!='frames'})
  (out/'demo_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(name,task,row['success'],reason,'frames',len(frames),'seconds',round(row['elapsed_s'],1),flush=True)
