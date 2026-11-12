"""Six world-axis approaches, four tool rolls; collision-filtered holding IK."""
import os
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,itertools,time
import numpy as np
import yaml,trimesh
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from fast_reference_kinematics import FastReferenceKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
import compare_nine_workspace as base
STATE=os.environ.get('SPARK_DIRECTION_STATE','open')
assert STATE in ('bare','open')
ROOT=base.ROOT;OUT=base.WS/'workspace_results'/f'six_direction_{STATE}';NAMES=base.OURS+base.EXTERNAL
if os.environ.get('SPARK_DIRECTION_MODELS'):
 NAMES=json.loads(os.environ['SPARK_DIRECTION_MODELS']);OUT=base.WS/'workspace_results'/f'dual_single_direction_{STATE}'
DIRECTIONS=['Top (-Z)','Bottom (+Z)','Left (-Y)','Right (+Y)','Front (-X)','Back (+X)']
AXES=np.array([[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[-1,0,0],[1,0,0]],float)
ORIENTS=[]
for z in AXES:
 x=np.array([1.,0,0]) if abs(z[0])<.5 else np.array([0.,1,0]);y=np.cross(z,x);frame=np.column_stack((x,y,z))
 for roll in [0,90,180,270]:
  r=frame@Rotation.from_euler('z',roll,degrees=True).as_matrix();q=Rotation.from_matrix(r).as_quat();ORIENTS.append(q[[3,0,1,2]])
ORIENTS=np.array(ORIENTS);CACHE={}

def load(name):
 if name in CACHE:return CACHE[name]
 if name in base.EXTERNAL:
  _,checker,meta,ignores,*_=base.load(name,STATE);p=base.OUT/'models'/f'{name}_{STATE}/robot.urdf';arm=FastReferenceKinematics(p,meta['base'],locked_joints=meta['locks']);ee=meta['ee']
 else:
  folder=base.OLD/'models'/f'{name}_{STATE}';cfg=yaml.safe_load((base.OLD/'configs'/f'{name}_{STATE}.yaml').read_text())['robot'];arm=FastReferenceKinematics(folder/'robot.urdf',cfg['base_link']);ee=cfg['ee_links'][0];manifest=yaml.safe_load((folder/'collision_meshes.yaml').read_text());meshes={}
  for link,entries in manifest['links'].items():
   parts=[]
   for entry in entries:
    old=entry['file'];relative=old.split('/rt_ws/')[-1];path=ROOT/relative
    if not path.exists():
     matches=list(base.WS.rglob(Path(old).name));assert matches,old;path=matches[0]
    m=trimesh.load(path,force='mesh');m.apply_scale(entry['scale']);t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',entry['rpy']).as_matrix();t[:3,3]=entry['xyz'];m.apply_transform(t);parts.append(m)
   meshes[link]=trimesh.util.concatenate(parts)
  checker=MeshCollisionChecker(meshes);ignores=[(a,b) for a,bs in cfg['self_collision_ignore'].items() for b in bs]
 positions=np.load(base.OUT/'grid.npz')['positions'];chain=arm._chains[ee];first=next(i for i,j in enumerate(chain) if j.get('name') in arm.joint_names);p=arm.forward(arm.zero_positions());centre=p[chain[first].find('child').get('link')][:3,3];radius=sum(np.linalg.norm(arm._origins[j.get('name')][:3,3]) for j in chain[first+1:]);blocked=np.linalg.norm(positions-centre,axis=1)>radius+.005
 CACHE[name]=(arm,checker,ee,ignores,positions,blocked);return CACHE[name]

def cell(task):
 np.seterr(all='ignore');name,i=task;arm,checker,ee,ignores,positions,blocked=load(name);ok=np.zeros(24,bool);qs=np.full((24,len(arm.joint_names)),np.nan);sigma=np.full(24,np.nan);condition=np.full(24,np.nan)
 if not blocked[i]:
  for j,o in enumerate(ORIENTS):
   r=arm.solve(positions[i],ee,o,seeds=8,max_iterations=120,position_tolerance=.005,orientation_tolerance=.08,collision=lambda q:checker.check(arm.forward(q),ignores=ignores,clearance=1e-6),seed=i*24+j)
   if r.success:
    ok[j]=True;qs[j]=r.positions;s=np.linalg.svd(arm.jacobian(r.positions,ee),compute_uv=False);sigma[j]=s[-1];condition[j]=s[0]/s[-1] if s[-1]>0 else np.inf
 return name,i,ok,qs,sigma,condition

def main():
 OUT.mkdir(exist_ok=True);positions=np.load(base.OUT/'grid.npz')['positions'];data={};tasks=[]
 for name in NAMES:
  p=OUT/(name+'.npz');n=len(load(name)[0].joint_names)
  data[name]=dict(accepted=np.zeros((2197,24),bool),solutions=np.full((2197,24,n),np.nan),sigma_min=np.full((2197,24),np.nan),condition=np.full((2197,24),np.nan),processed=np.zeros(2197,bool))
  if p.exists():
   with np.load(p) as a:
    for k in data[name]:data[name][k]=a[k]
  tasks.extend((name,int(i)) for i in np.where(~data[name]['processed'])[0])
 # Round robin models, so progress is not hidden behind one slow arm.
 tasks.sort(key=lambda t:(t[1],NAMES.index(t[0])))
 def save():
  for name,d in data.items():
   tmp=OUT/(name+'.partial.npz');np.savez_compressed(tmp,positions=positions,orientations_wxyz=ORIENTS,**d);os.replace(tmp,OUT/(name+'.npz'))
  progress={n:int(d['processed'].sum()) for n,d in data.items()};(OUT/'progress.json').write_text(json.dumps(progress,indent=2));print(progress,flush=True)
 last=time.time()
 with ProcessPoolExecutor(max_workers=int(os.environ.get("SPARK_DIRECTION_WORKERS","5")),mp_context=mp.get_context('spawn')) as pool:
  for name,i,ok,q,s,c in pool.map(cell,tasks,chunksize=2):
   d=data[name];d['accepted'][i]=ok;d['solutions'][i]=q;d['sigma_min'][i]=s;d['condition'][i]=c;d['processed'][i]=True
   if time.time()-last>30:save();last=time.time()
 save();(OUT/'complete.json').write_text(json.dumps({'status':'complete','directions':DIRECTIONS,'rolls_deg':[0,90,180,270],'grid_points':2197,'singularity_note':'J=[Jv/(1m);Jw]; condition >=100 flags sampled near-singular configurations, not intrinsically singular positions'},indent=2))
if __name__=='__main__':main()
