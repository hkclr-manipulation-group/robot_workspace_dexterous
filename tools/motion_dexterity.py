"""Reproducible finite trajectory suite, full-turn geometry; no hardware timing.

12 common positions, six approach directions, four allowed starting rolls.
MOT-01 translation; MOT-02 fixed-position tilt; MOT-03 translate and rotate.
Tracks unwrapped joint coordinates. Path interiors are checked at <=1 degree
joint increments, with refinement if pose tracking or collision checks fail.
This is resolution-based checking, not a continuous collision certificate.
"""
import os
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial.transform import Rotation
from full_rotation_workspace import load,NAMES,ROOT,STATE,previous

OUT=ROOT/'robot_workspace_dexterous/workspace_results/motion_dexterity_v1'/STATE
CENTRES=np.array([[.2,.2,.2],[.2,-.2,.2],[-.2,.2,.2],[-.2,-.2,.2],
 [.6,0,.2],[-.6,0,.2],[0,.6,.2],[0,-.6,.2],[.2,0,.6],[-.2,0,.6],[.2,0,-.2],[-.2,0,-.2]])
TRANS=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1.]])
POSITION_TOL=.002;ROTATION_TOL=.02;MAX_STEP=np.deg2rad(1)
ORIENTATIONS=Rotation.random(64,random_state=np.random.default_rng(20261112)).as_quat()[:,[3,0,1,2]]

def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def model_data(name):
 arm,checker,ee,ignores,p,_=load(name)
 archive=np.load(ROOT/'robot_workspace_dexterous/workspace_results/full_rotation/six'/STATE/(name+'.npz'))
 indices=[int(np.argmin(np.linalg.norm(p-x,axis=1))) for x in CENTRES]
 assert np.allclose(p[indices],CENTRES)
 return arm,checker,ee,ignores,archive,indices

def distances(checker,poses,ignores):
 """Minimum triangle surface separation; containment checked by check() first."""
 checker._update(poses);excluded={tuple(sorted(x)) for x in ignores}
 best=np.inf;pair=None
 for i,j in checker._pairs:
  a,b=checker.names[i],checker.names[j]
  if tuple(sorted((a,b))) in excluded:continue
  # AABB distance is a conservative lower bound for triangle distance.
  gap=np.maximum(np.maximum(checker._low[i]-checker._high[j],checker._low[j]-checker._high[i]),0)
  if np.linalg.norm(gap)>=best:continue
  d=float(checker.fcl.distance(checker.objects[a],checker.objects[b],checker.fcl.DistanceRequest(),checker.fcl.DistanceResult()))
  if not np.isfinite(d) or d<0:raise RuntimeError('Invalid distance on a collision-free sample')
  if d<best:best,pair=d,[a,b]
 assert pair is not None
 return best,pair

def pose_error(arm,q,ee,p,r):
 t=arm.forward(q)[ee]
 return float(np.linalg.norm(t[:3,3]-p)),float(Rotation.from_matrix(t[:3,:3]@r.T).magnitude())

def target(centre,direction,roll,kind,index,s):
 r0=Rotation.from_quat(previous.ORIENTS[direction*4+roll][[1,2,3,0]]).as_matrix()
 p=centre.copy();r=r0
 if kind in ('MOT-01','MOT-03'):p=p+.12*s*TRANS[(index+direction)%6]
 if kind in ('MOT-02','MOT-03'):
  # Alternate tool roll and tilt, with equal rotation amplitude.
  axis=np.array([1.,0,0]) if (index+direction)%2==0 else np.array([0.,0,1])
  r=r0@Rotation.from_rotvec(axis*(np.pi/2)*s).as_matrix()
 return p,r

def local_solve(arm,ee,q,p,r):
 # Bounds are centred on the unwrapped seed, not the +/-pi display seam.
 arm.lower[:]=q-.5;arm.upper[:]=q+.5
 o=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
 result=arm.solve(p,ee,o,warm_start=q,seeds=1,max_iterations=60,
   position_tolerance=POSITION_TOL,orientation_tolerance=ROTATION_TOL)
 return result.positions if result.success else None

def follow(arm,checker,ee,ignores,q0,centre,direction,roll,kind,index,resolution=MAX_STEP):
 nodes=[(0.,q0.copy())];failure=None
 def advance(s0,q,s1,depth):
  nonlocal failure
  p,r=target(centre,direction,roll,kind,index,s1)
  q1=local_solve(arm,ee,q,p,r)
  reason='tracking_budget'
  if q1 is not None:
   count=max(2,int(np.ceil(np.max(np.abs(q1-q))/resolution)))
   reason=None
   for t in np.linspace(0,1,count+1)[1:]:
    s=s0+(s1-s0)*t;qi=q+t*(q1-q);pt,rt=target(centre,direction,roll,kind,index,s)
    pe,re=pose_error(arm,qi,ee,pt,rt)
    if pe>POSITION_TOL or re>ROTATION_TOL:reason='interior_pose_error';break
    if checker.check(arm.forward(qi),ignores=ignores,clearance=1e-6):reason='path_collision';break
   if reason is None:return [(s1,q1)]
  # Reduce local motion; do not jump to unrelated global IK solutions.
  if depth<3:
   mid=(s0+s1)/2;left=advance(s0,q,mid,depth+1)
   if left is not None:
    right=advance(mid,left[-1][1],s1,depth+1)
    if right is not None:return left+right
  failure=dict(category=reason or 'tracking_budget',fraction=float(s1))
  return None
 for s in np.linspace(0,1,17)[1:]:
  step=advance(nodes[-1][0],nodes[-1][1],float(s),0)
  if step is None:return None,failure
  nodes.extend(step)
 return nodes,None

def follow_redundancy(arm,c,ee,ignore,q0,centre,direction,roll,index):
 """Track 0.5 rad nominal null-space travel, both signs; seven axes only."""
 p,r=target(centre,direction,roll,'MOT-04',index,0)
 for sign in [1.,-1.]:
  nodes=[(0.,q0.copy())];last=None;failed=False
  for step in range(1,21):
   q=nodes[-1][1];_,_,vt=np.linalg.svd(arm.jacobian(q,ee),full_matrices=True);v=vt[-1]
   if last is None:v*=sign
   elif v@last<0:v=-v
   last=v;nextq=local_solve(arm,ee,q+.025*v,p,r)
   if nextq is None:failed=True;break
   n=max(2,int(np.ceil(np.max(np.abs(nextq-q))/MAX_STEP)))
   for t in np.linspace(0,1,n+1)[1:]:
    qi=q+t*(nextq-q);pe,re=pose_error(arm,qi,ee,p,r)
    if pe>POSITION_TOL or re>ROTATION_TOL or c.check(arm.forward(qi),ignores=ignore,clearance=1e-6):failed=True;break
   if failed:break
   nodes.append((step/20,nextq))
  if not failed and np.linalg.norm(nodes[-1][1]-q0)>=.25:return nodes,None
 return None,dict(category='redundancy_tracking_or_collision',fraction=float(len(nodes)/21))

def measure(arm,c,ee,ignore,nodes,centre,direction,roll,kind,index):
 records=[];sigmas=[];minimum=np.inf;minpair=None;minfraction=None;maxp=0.;maxr=0.
 samples=[nodes[0]]
 for (s0,q0),(s1,q1) in zip(nodes,nodes[1:]):
  count=max(2,int(np.ceil(np.max(np.abs(q1-q0))/MAX_STEP)))
  samples.extend((s0+(s1-s0)*t,q0+(q1-q0)*t) for t in np.linspace(0,1,count+1)[1:])
 for s,q in samples:
  poses=arm.forward(q);assert not c.check(poses,ignores=ignore,clearance=1e-6)
  d,pair=distances(c,poses,ignore)
  if d<minimum:minimum,minpair,minfraction=d,pair,s
  sigma=float(np.linalg.svd(arm.jacobian(q,ee),compute_uv=False)[-1]);sigmas.append(sigma)
  pt,rt=target(centre,direction,roll,kind,index,s);pe,re=pose_error(arm,q,ee,pt,rt);maxp=max(maxp,pe);maxr=max(maxr,re)
  records.append([s,*q])
 # Equal parameter intervals: constant translation/rotation speed in these tasks.
 sample_s=np.array([x[0] for x in samples]);regular_sigma=np.interp(np.linspace(0,1,101),sample_s,sigmas)
 return dict(sigma_min=min(sigmas),sigma_p05=float(np.quantile(regular_sigma,.05)),clearance_min_m=minimum,
   clearance_pair=minpair,clearance_fraction=float(minfraction),position_error_max_m=maxp,orientation_error_max_rad=maxr,
   joint_travel_rad=float(np.abs(np.diff(np.array([q for _,q in nodes]),axis=0)).sum()),checked_samples=len(samples)),np.array(records)

def run_model(name):
 np.seterr(all='ignore');start=time.monotonic();folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 arm,c,ee,ignore,old,indices=model_data(name);paths=[];witnesses={}
 # A saved static solution is a warm start, not automatic path acceptance.
 # Up to four allowed rolls, one previously found branch per roll.
 for index,centre in enumerate(CENTRES):
  for direction in range(6):
   starts=[]
   for roll in range(4):
    j=direction*4+roll
    if not old['accepted'][indices[index],j]:continue
    q=old['solutions'][indices[index],j].copy();p,r=target(centre,direction,roll,'MOT-01',index,0)
    q=local_solve(arm,ee,q,p,r)
    if q is not None and not c.check(arm.forward(q),ignores=ignore,clearance=1e-6):starts.append((roll,q))
   kinds=['MOT-01','MOT-02','MOT-03']+(['MOT-04'] if len(arm.joint_names)==7 else [])
   for kind in kinds:
    pid=f'{kind}-P{index:02d}-D{direction}';row=dict(id=pid,kind=kind,centre=index,direction=direction,start_branches=len(starts),success=False,attempts=0)
    if not starts:row['failure']='no_refined_cached_start'
    for roll,q in starts:
     row['attempts']+=1
     if kind=='MOT-04':nodes,failure=follow_redundancy(arm,c,ee,ignore,q,centre,direction,roll,index)
     else:nodes,failure=follow(arm,c,ee,ignore,q,centre,direction,roll,kind,index)
     if nodes is None:row['failure']=failure['category'];row['failed_fraction']=failure['fraction'];continue
     metrics,trace=measure(arm,c,ee,ignore,nodes,centre,direction,roll,kind,index)
     row.update(metrics,success=True,roll=roll,failure=None)
     if kind=='MOT-04':
      elbow=next(j for j in arm._joints if j.get('name')==arm.joint_names[3]).find('child').get('link')
      row['joint4_origin_displacement_m']=float(np.linalg.norm(arm.forward(nodes[-1][1])[elbow][:3,3]-arm.forward(nodes[0][1])[elbow][:3,3]))
     witnesses[pid]=trace
     break
    paths.append(row)
  total=288 if len(arm.joint_names)==7 else 216
  write_json(folder/'progress.json',dict(model=name,completed_paths=len(paths),total_paths=total,elapsed_s=time.monotonic()-start))
  print(STATE,name,'paths',len(paths),'/',total,flush=True)
 np.savez_compressed(folder/'paths.npz',**witnesses)
 write_json(folder/'paths.json',paths)
 write_json(folder/'complete.json',dict(status='complete',elapsed_s=time.monotonic()-start))
 return name

def protocol():
 return dict(name='motion_dexterity_v1',state=STATE,models=NAMES,centres_m=CENTRES.tolist(),
  paths_per_model=216,extra_seven_axis_paths=72,kinds=['MOT-01','MOT-02','MOT-03','MOT-04 (7 DOF only)'],translation_m=.12,rotation_rad=np.pi/2,
  redundancy='20 null-space predictor/projector steps of 0.025 rad; both signs tried; final joint displacement >=0.25 rad',
  starts='one cached branch per allowed roll, refined; at most four starts, no exhaustive branch search',
  angular_policy='unwrapped full-turn joints, no native angular stops',position_tolerance_m=POSITION_TOL,
  orientation_tolerance_rad=ROTATION_TOL,max_joint_sampling_step_rad=MAX_STEP,
  initial_path_intervals=16,max_refinement_depth=3,collision='same meshes/exemptions; FCL; non-exempt distance',
  J_normalisation_m=1.,timing='not evaluated',scope='12 common anchor positions; not full-workspace success rate')

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 spec=protocol();spec['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 spec['static_npz_sha256']={n:hashlib.sha256((ROOT/'robot_workspace_dexterous/workspace_results/full_rotation/six'/STATE/(n+'.npz')).read_bytes()).hexdigest() for n in NAMES}
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec,'Protocol changed: use a new output directory'
 write_json(OUT/'protocol.json',spec)
 names=json.loads(os.environ.get('SPARK_MOTION_MODELS',json.dumps(NAMES)))
 todo=[n for n in names if not (OUT/n/'complete.json').exists()]
 with ProcessPoolExecutor(max_workers=int(os.environ.get('SPARK_MOTION_WORKERS','4')),mp_context=mp.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(run_model,n) for n in todo]):print('completed',STATE,f.result(),flush=True)
 if all((OUT/n/'complete.json').exists() for n in NAMES):write_json(OUT/'complete.json',dict(status='complete'))
if __name__=='__main__':main()
