"""Original non-JIT FK/Jacobian, double-resolution path and SO(3) checks."""
import os,json,hashlib,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial.transform import Rotation
from collision_shpere_generation.kinematics import RobotKinematics
from full_rotation_workspace import ROOT,NAMES,STATE,load,previous
from motion_dexterity import OUT,CENTRES,target,distances,write_json,POSITION_TOL,ROTATION_TOL,MAX_STEP
AUDIT=ROOT/'robot_workspace_dexterous/workspace_results/orientation_audit_v1'/STATE

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(name):
 np.seterr(all='ignore');fast,c,ee,ignore,_,_=load(name);locks={}
 if name in previous.base.EXTERNAL:locks=json.loads((previous.base.OUT/'models'/f'{name}_{STATE}/model.json').read_text())['locks']
 arm=RobotKinematics(fast.urdf,fast.base_link,locked_joints=locks)
 folder=OUT/name;rows=json.loads((folder/'paths.json').read_text());traces=np.load(folder/'paths.npz')
 assert len(rows)==(288 if len(arm.joint_names)==7 else 216)
 assert set(traces.files)=={r['id'] for r in rows if r['success']}
 result={}
 for row in rows:
  if not row['success']:continue
  trace=traces[row['id']];s=trace[:,0];q=trace[:,1:]
  assert q.shape[1]==len(arm.joint_names) and np.isfinite(trace).all()
  assert s[0]==0 and s[-1]==1 and (np.diff(s)>0).all()
  assert np.abs(np.diff(q,axis=0)).max()<=MAX_STEP+1e-10
  # Interleave every midpoint for a second, finer collision/pose audit.
  refined=np.empty((2*len(trace)-1,trace.shape[1]));refined[::2]=trace;refined[1::2]=(trace[1:]+trace[:-1])/2
  values=[];clearance=np.inf;pair=None;valid=True;reason=None;maxpe=0.;maxre=0.
  for record in refined:
   fraction=float(record[0]);qi=record[1:];poses=arm.forward(qi)
   pt,rt=target(CENTRES[row['centre']],row['direction'],row['roll'],row['kind'],row['centre'],fraction)
   pe=float(np.linalg.norm(poses[ee][:3,3]-pt));re=float(Rotation.from_matrix(poses[ee][:3,:3]@rt.T).magnitude());maxpe=max(maxpe,pe);maxre=max(maxre,re)
   if pe>POSITION_TOL+1e-9 or re>ROTATION_TOL+1e-9:valid=False;reason='refined_pose_error';break
   if c.check(poses,ignores=ignore,clearance=1e-6):valid=False;reason='refined_collision';break
   d,links=distances(c,poses,ignore)
   if d<clearance:clearance=d;pair=links
   values.append(float(np.linalg.svd(arm.jacobian(qi,ee),compute_uv=False)[-1]))
  item=dict(accepted_at_refined_resolution=valid,reason=reason)
  if valid:
   uniform=np.interp(np.linspace(0,1,101),refined[:,0],values)
   item.update(sigma_min=min(values),sigma_p05=float(np.quantile(uniform,.05)),clearance_min_m=float(clearance),clearance_pair=pair,
     checked_samples=len(refined),position_error_max_m=maxpe,orientation_error_max_rad=maxre)
  result[row['id']]=item
 report=dict(status='complete',model=name,independent_FK='RobotKinematics (not JIT)',max_joint_sampling_step_rad=MAX_STEP/2,
  path_npz_sha256=sha(folder/'paths.npz'),path_json_sha256=sha(folder/'paths.json'),paths=result,
  original_successes=sum(r['success'] for r in rows),refined_successes=sum(x['accepted_at_refined_resolution'] for x in result.values()))
 write_json(folder/'validation.json',report)
 # Static SO(3) sample validation, including the independent-seed repeat.
 af=AUDIT/name;z=np.load(af/'samples.npz');assert int(z['completed_centres'])==12;checked=0
 for flags,qs in [(z['accepted'],z['solutions']),(z['repeat_accepted'],z['repeat_solutions'])]:
  for i,j in zip(*np.where(flags)):
   v=qs[i,j];assert np.all(np.abs(v)<=np.pi+1e-9);poses=arm.forward(v);t=poses[ee]
   rot=Rotation.from_quat(z['orientations_wxyz'][j][[1,2,3,0]]).as_matrix()
   assert np.linalg.norm(t[:3,3]-z['positions'][i])<=.005+1e-9
   assert Rotation.from_matrix(t[:3,:3]@rot.T).magnitude()<=.08+1e-9
   assert not c.check(poses,ignores=ignore,clearance=1e-6);checked+=1
 write_json(af/'validation.json',dict(status='PASS',accepted_solutions_checked=checked,npz_sha256=sha(af/'samples.npz')))
 print('validated',STATE,name,report['refined_successes'],'/',len(rows),flush=True);return name

def main():
 with ProcessPoolExecutor(max_workers=int(os.environ.get('SPARK_MOTION_WORKERS','4')),mp_context=mp.get_context('spawn')) as pool:
  jobs=[pool.submit(validate,n) for n in NAMES if not (OUT/n/'validation.json').exists() or not (AUDIT/n/'validation.json').exists()]
  for f in as_completed(jobs):f.result()
 write_json(OUT/'validation_complete.json',dict(status='complete'));write_json(AUDIT/'validation_complete.json',dict(status='complete'))
if __name__=='__main__':main()
