"""Native FK workspace sampling and bounded folded-envelope search.
Uses source URDF limits, vendor triangle collision meshes and visual mesh hulls
for bounding boxes. Results are finite-sample estimates, not global optima.
"""
import argparse,copy,json,itertools,hashlib,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc
from scipy.optimize import minimize
from collision_shpere_generation.kinematics import RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
from collision_shpere_generation.generator import _origin_matrix
import trimesh
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'robot_workspace_dexterous/workspace_results/external_references'
NAMES=['external_piper','external_ur5','external_panda']

def load(name,state):
 folder=ROOT/'collision_shpere_generation/models'/name
 meta=json.loads((folder/'source_manifest.json').read_text())
 urdf=folder/('robot.urdf' if state=='bare' else 'equipped.urdf')
 arm=RobotKinematics(urdf,meta['base_link'],locked_joints={} if state=='bare' else meta['locked_joints'])
 checker=MeshCollisionChecker.from_urdf(urdf)
 tree=ET.parse(urdf).getroot();visual={}
 for link in tree.findall('link'):
  parts=[]
  for element in link.findall('visual') or link.findall('collision'):
   tag=element.find('geometry/mesh');m=trimesh.load(folder/tag.get('filename'),force='mesh',process=False)
   m.vertices*=np.fromstring(tag.get('scale','1 1 1'),sep=' ');m.apply_transform(_origin_matrix(element));parts.append(m)
  if parts:visual[link.get('name')]=trimesh.util.concatenate(parts)
 # Ignore rigid assemblies and direct moving-joint neighbours. Keep all pairs
 # separated by >=2 movable joints, including the mounted tool and lower arm.
 graph={k:[] for k in arm.links}
 for j in tree.findall('joint'):
  a,b=j.find('parent').get('link'),j.find('child').get('link')
  weight=int(j.get('name') in arm.joint_names)
  graph[a].append((b,weight));graph[b].append((a,weight))
 ignores=[]
 for a in checker.names:
  stack=[(a,None,0)]
  while stack:
   b,parent,d=stack.pop()
   if b in checker.names and a<b and d<=1:ignores.append((a,b))
   for child,w in graph[b]:
    if child!=parent:stack.append((child,b,d+w))
 return arm,checker,visual,meta,ignores

def analyze(name,power):
 folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 records={}
 for state in ['bare','open']:
  start=time.time();arm,checker,visual,meta,ignores=load(name,state)
  n=len(arm.joint_names);assert n==(7 if name=='external_panda' else 6)
  rng=qmc.Sobol(n,scramble=True,seed=20261111)
  qs=qmc.scale(rng.random_base2(power),arm.lower,arm.upper)
  hulls={k:m.vertices[ConvexHull(m.vertices).vertices] for k,m in visual.items()}
  def points(k,p):return np.einsum('ij,nj->ni',p[k][:3,:3],hulls[k])+p[k][:3,3]
  ref=arm.forward(arm.zero_positions());floor=min(points(k,ref)[:,2].min() for k in hulls if arm.is_static(k))
  def envelope(q):
   p=arm.forward(q);lo=[];hi=[]
   for k in hulls:
    v=points(k,p);lo.append(v.min(0));hi.append(v.max(0))
   return np.min(lo,0),np.max(hi,0)
  def collides(q):return checker.check(arm.forward(q),ignores=ignores,clearance=1e-6)
  positions=[];clear=[];candidates=[]
  # Flange is used for both states, isolating tool-induced collisions.
  for i,q in enumerate(qs):
   p=arm.forward(q);positions.append(p[meta['flange_link']][:3,3].copy());ok=not collides(q);clear.append(ok)
   if ok:
    lo,hi=envelope(q)
    if lo[2]>=floor-1e-7:candidates.append((float(np.prod(hi-lo)),q.copy()))
   if (i+1)%1024==0:print(name,state,'FK',i+1,len(qs),'clear',sum(clear),flush=True)
  choices=[sorted(set([float(lo),float(hi)]+[float(a) for a in np.deg2rad([-180,-90,0,90,180]) if lo<=a<=hi])) for lo,hi in zip(arm.lower,arm.upper)]
  # Rank quarter-turn/boundary configurations geometrically before checking
  # collisions; exact same state constraints apply as to the Sobol candidates.
  ranked=[]
  for q in itertools.product(*choices):
   q=np.array(q);lo,hi=envelope(q)
   if lo[2]>=floor-1e-7:ranked.append((float(np.prod(hi-lo)),q))
  ranked.sort(key=lambda v:v[0]);found=0
  for volume,q in ranked:
   if not collides(q):
    candidates.append((volume,q));found+=1
    if found>=8:break
  if not candidates:raise RuntimeError('No floor-clear collision-free folded candidate')
  candidates.sort(key=lambda v:v[0]);best_volume,best_q=candidates[0]
  def objective(q):
   nonlocal best_volume,best_q
   lo,hi=envelope(q);volume=float(np.prod(hi-lo))
   if lo[2]<floor-1e-7:return 10+100*(floor-lo[2])+volume
   if collides(q):return 10+volume
   if volume<best_volume:best_volume,best_q=volume,q.copy()
   return volume
  for _,initial in candidates[:2]:minimize(objective,initial,method='Powell',bounds=list(zip(arm.lower,arm.upper)),options={'maxfev':450,'xtol':1e-4,'ftol':1e-6})
  lo,hi=envelope(best_q);hits=checker.collisions(arm.forward(best_q),ignores=ignores)
  assert not hits and lo[2]>=floor-1e-7 and np.all(best_q>=arm.lower-1e-9) and np.all(best_q<=arm.upper+1e-9)
  positions=np.array(positions);clear=np.array(clear)
  np.savez_compressed(folder/(state+'.npz'),joints=qs,flange_positions=positions,collision_free=clear)
  voxels=np.floor(positions/.05).astype(int)
  row=dict(state=state,samples=len(qs),collision_free_samples=int(clear.sum()),occupied_50mm_voxels_before=len(np.unique(voxels,axis=0)),occupied_50mm_voxels_after=len(np.unique(voxels[clear],axis=0)),joint_names=list(arm.joint_names),limits_deg=np.rad2deg(np.c_[arm.lower,arm.upper]).tolist(),ignored_pairs=ignores,open_collision_mesh_links=checker.open_mesh_links,source_revision=meta['source_revision'],urdf_sha256=meta['urdf_sha256']['robot.urdf' if state=='bare' else 'equipped.urdf'],gripper_note=meta['gripper_note'],fold=dict(degrees=np.rad2deg(best_q).tolist(),dimensions_mm=np.ceil((hi-lo)*1000).astype(int).tolist(),exact_dimensions_mm=((hi-lo)*1000).tolist(),bounds_mm=(np.array([lo,hi])*1000).tolist(),volume_l=best_volume*1000,contacts=[],below_base_mm=max(0,float((floor-lo[2])*1000)),quarter_boundary_candidates=len(ranked),search='Sobol + in-limit quarter-turn/boundary enumeration + two bounded Powell refinements; best found, not certified global minimum'),elapsed_seconds=time.time()-start)
  records[state]=row;(folder/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
  print('DONE',name,state,row['fold']['dimensions_mm'],round(row['fold']['volume_l'],2),'L',flush=True)
 return records

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--model',choices=NAMES,required=True);parser.add_argument('--power',type=int,default=13);args=parser.parse_args()
 np.seterr(all='ignore');analyze(args.model,args.power)
