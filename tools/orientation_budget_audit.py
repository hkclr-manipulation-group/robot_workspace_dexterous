"""Uniform SO(3) sampling / IK-budget sensitivity at the 12 motion anchors.
Not a replacement for the existing 2,197-position workspace volume study.
"""
import os,json,hashlib,time,multiprocessing as mp,fcntl
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from motion_dexterity import CENTRES,ORIENTATIONS,write_json
from full_rotation_workspace import ROOT,NAMES,STATE,load
OUT=ROOT/'robot_workspace_dexterous/workspace_results/orientation_audit_v1'/STATE

def run_model(name):
 folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 with (folder/'worker.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (folder/'complete.json').exists():return name
  return compute_model(name)

def compute_model(name):
 np.seterr(all='ignore');arm,c,ee,ignore,_,_=load(name);folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 a=np.zeros((12,64),bool);q=np.full((12,64,len(arm.joint_names)),np.nan);attempts=np.zeros((12,64),int)
 repeat=np.zeros((12,16),bool);rq=np.full((12,16,len(arm.joint_names)),np.nan)
 for i,p in enumerate(CENTRES):
  for j,o in enumerate(ORIENTATIONS):
   r=arm.solve(p,ee,o,seeds=8,max_iterations=120,collision=lambda v:c.check(arm.forward(v),ignores=ignore,clearance=1e-6),seed=20261112+i*64+j)
   if r.success:a[i,j]=True;q[i,j]=r.positions
   attempts[i,j]=r.attempts
  for j,o in enumerate(ORIENTATIONS[:16]):
   r=arm.solve(p,ee,o,seeds=8,max_iterations=120,collision=lambda v:c.check(arm.forward(v),ignores=ignore,clearance=1e-6),seed=30261112+i*64+j)
   if r.success:repeat[i,j]=True;rq[i,j]=r.positions
  np.savez_compressed(folder/'samples.npz',positions=CENTRES,orientations_wxyz=ORIENTATIONS,accepted=a,solutions=q,attempts=attempts,repeat_accepted=repeat,repeat_solutions=rq,completed_centres=i+1)
  write_json(folder/'progress.json',dict(completed_centres=i+1,total_centres=12))
  print(STATE,name,'SO3',i+1,'/12',flush=True)
 write_json(folder/'complete.json',dict(status='complete'));return name

def main():
 spec=dict(state=STATE,models=NAMES,positions_m=CENTRES.tolist(),orientations_wxyz=ORIENTATIONS.tolist(),
  rotation_sampling='Haar-uniform SO(3), seed 20261112, nested prefixes 16/32/64',
  ik='zero initial then seven random starts; max_nfev 120; first four versus all eight budget',
  repeat='first 16 rotations repeated with IK seeds offset by 10000000',
  position_tolerance_m=.005,orientation_tolerance_rad=.08,scope='12 fixed anchor positions; not full workspace coverage',
  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 write_json(OUT/'protocol.json',spec)
 names=json.loads(os.environ.get('SPARK_MOTION_MODELS',json.dumps(NAMES)))
 with ProcessPoolExecutor(max_workers=int(os.environ.get('SPARK_MOTION_WORKERS','4')),mp_context=mp.get_context('spawn')) as pool:
  jobs=[pool.submit(run_model,n) for n in names if not (OUT/n/'complete.json').exists()]
  for f in as_completed(jobs):print('completed',STATE,f.result(),flush=True)
 if all((OUT/n/'complete.json').exists() for n in NAMES):write_json(OUT/'complete.json',dict(status='complete'))
if __name__=='__main__':main()
