"""Separate geometric benchmark: complete joint revolutions, collision-filtered."""
import os
os.environ['SPARK_EXTRA_MODELS']='["dual_v2_1_left","dual_v2_2_left"]'
os.environ.pop('SPARK_DIRECTION_MODELS',None)
import json,time,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import six_direction_grasp as previous
STATE=previous.STATE;SET=os.environ.get('SPARK_FULL_SET','six');assert SET in ('six','sixteen')
ROOT=previous.ROOT;NAMES=previous.NAMES;OUT=previous.base.WS/'workspace_results/full_rotation'/SET/STATE
ORIENTS=previous.ORIENTS if SET=='six' else np.load(previous.base.OUT/'grid.npz')['orientations_wxyz']
COUNT=len(ORIENTS);CACHE={}
def full_turn(arm):
 for name in arm.joint_names:
  j=next(j for j in arm._joints if j.get('name')==name)
  assert j.get('type') in ('revolute','continuous'),'Only arm rotation joints may be expanded'
 arm.lower[:]=-np.pi;arm.upper[:]=np.pi
 for name in arm.joint_names:arm._limits[name]=(-np.pi,np.pi)
 return arm

def load(name):
 if name not in CACHE:
  arm,c,ee,ignore,p,blocked=previous.load(name);full_turn(arm);CACHE[name]=(arm,c,ee,ignore,p,blocked)
 return CACHE[name]

def cell(task):
 np.seterr(all='ignore');name,i=task;arm,c,ee,ignore,p,blocked=load(name);accepted=np.zeros(COUNT,bool);qout=np.full((COUNT,len(arm.joint_names)),np.nan);sigma=np.full(COUNT,np.nan);condition=np.full(COUNT,np.nan)
 if not blocked[i]:
  for j,o in enumerate(ORIENTS):
   result=arm.solve(p[i],ee,o,seeds=8,max_iterations=120,position_tolerance=.005,orientation_tolerance=.08,collision=lambda q:c.check(arm.forward(q),ignores=ignore,clearance=1e-6),seed=i*COUNT+j)
   if result.success:
    accepted[j]=True;qout[j]=result.positions;s=np.linalg.svd(arm.jacobian(result.positions,ee),compute_uv=False);sigma[j]=s[-1];condition[j]=s[0]/s[-1] if s[-1] else np.inf
 return name,i,accepted,qout,sigma,condition

def main():
 OUT.mkdir(parents=True,exist_ok=True);p=np.load(previous.base.OUT/'grid.npz')['positions'];data={};tasks=[];models={}
 for name in NAMES:
  arm,c,ee,ignore,_,_=load(name);n=len(arm.joint_names);d=dict(accepted=np.zeros((len(p),COUNT),bool),solutions=np.full((len(p),COUNT,n),np.nan),sigma_min=np.full((len(p),COUNT),np.nan),condition=np.full((len(p),COUNT),np.nan),processed=np.zeros(len(p),bool));file=OUT/(name+'.npz')
  if file.exists():
   with np.load(file) as a:
    for k in d:d[k]=a[k]
  data[name]=d;tasks.extend((name,int(i)) for i in np.where(~d['processed'])[0]);models[name]=dict(joints=list(arm.joint_names),lower=arm.lower.tolist(),upper=arm.upper.tolist(),source_urdf=str(arm.urdf.relative_to(ROOT)),source_sha256=hashlib.sha256(arm.urdf.read_bytes()).hexdigest(),ee=ee,ignored_pairs=ignore,open_mesh_links=c.open_mesh_links)
 protocol=dict(joint_policy='All arm revolute joints [-pi, pi]; source limits overridden at runtime',state=STATE,orientation_set=SET,orientation_count=COUNT,grid_points=len(p),collision='FCL mesh; existing rigid/adjacent exclusions retained',seeds=8,max_nfev=120,position_tolerance_m=.005,orientation_tolerance_rad=.08,singularity='Recorded only, not rejected',models=models)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 def save():
  for name,d in data.items():
   tmp=OUT/(name+'.partial.npz');np.savez_compressed(tmp,positions=p,orientations_wxyz=ORIENTS,reachable_orientations=d['accepted'].sum(1),**d);os.replace(tmp,OUT/(name+'.npz'))
  progress={n:int(d['processed'].sum()) for n,d in data.items()};(OUT/'progress.json').write_text(json.dumps(progress,indent=2));print(SET,STATE,progress,flush=True)
 tasks.sort(key=lambda t:(t[1],NAMES.index(t[0])));last=time.time()
 with ProcessPoolExecutor(max_workers=int(os.environ.get('SPARK_FULL_WORKERS','2')),mp_context=mp.get_context('spawn')) as pool:
  for name,i,a,q,s,c in pool.map(cell,tasks,chunksize=2):
   d=data[name];d['accepted'][i]=a;d['solutions'][i]=q;d['sigma_min'][i]=s;d['condition'][i]=c;d['processed'][i]=True
   if time.time()-last>30:save();last=time.time()
 save();(OUT/'complete.json').write_text(json.dumps({'status':'complete'}))
if __name__=='__main__':main()
