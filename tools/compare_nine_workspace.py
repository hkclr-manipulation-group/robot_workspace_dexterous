"""Common 20 cm grid: subsample six archived 1 m arms and solve three references.
External tools retain native size and the documented tool reference frames.
Panda uses its native Hand TCP at 103.4 mm; other tools use the 155 mm reference. Holding uses a 30 mm cube at that point.
CPU and archived CUDA IK differ; this is a matched-condition comparison, not
bitwise solver equivalence. Unaccepted finite-seed targets remain inconclusive.
"""
import os
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,copy,hashlib,json,time,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from collision_shpere_generation.kinematics import RobotKinematics
if os.environ.get("SPARK_BENCHMARK_JIT") == "1":
 from fast_reference_kinematics import FastReferenceKinematics as RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
ROOT=Path(__file__).resolve().parents[2];WS=ROOT/'robot_workspace_dexterous';OLD=WS/'workspace_results/six_model_comparison';OUT=WS/'workspace_results/nine_model_comparison'
OURS=['spark2_v1','spark2_v2_2','P2AB-1','P2AB-2','P2AB-3','P2AB-4'];EXTERNAL=['external_piper','external_ur5','external_panda'];STATES=['bare','open','holding']
CACHE={}

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 first=np.load(OLD/'output/P2AB-2_bare/L66.npz');p=first['positions'];keep=np.all(np.isclose(p/.2,np.round(p/.2),atol=1e-5),axis=1);positions=p[keep].astype(float);assert len(positions)==2197
 source_meta=json.loads((OLD/'output/P2AB-2_bare/L66_summary.json').read_text());orientations=np.array(source_meta['orientation_samples_wxyz']);assert orientations.shape==(16,4)
 np.savez(OUT/'grid.npz',positions=positions,orientations_wxyz=orientations)
 sources=json.loads((OLD/'sources.json').read_text())
 for name in OURS:
  for state in STATES:
   file=OLD/'output'/f'{name}_{state}'/Path(sources[f'{name}_{state}']).name
   a=np.load(file);np.testing.assert_allclose(a['positions'],p,atol=1e-6)
   counts=a['reachable_orientations'][keep];dest=OUT/f'{name}_{state}';dest.mkdir(exist_ok=True)
   np.savez_compressed(dest/'workspace.npz',positions=positions,reachable_orientations=counts,orientations_wxyz=orientations)
   summary(name,state,counts,dict(backend='archived CUDA/cuRobo',original_npz=str(file.relative_to(ROOT)),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),grid_points=2197))
 for name in EXTERNAL:
  source=ROOT/'collision_shpere_generation/models'/name;meta=json.loads((source/'source_manifest.json').read_text());bare=ET.parse(source/'robot.urdf').getroot();arm_links={l.get('name') for l in bare.findall('link')};arm_joints={j.get('name') for j in bare.findall('joint')}
  length=sum(np.linalg.norm(np.fromstring(o.get('xyz','0 0 0'),sep=' ')) for o in bare.findall('joint/origin'));scale=1/length
  for state in STATES:
   dest=OUT/'models'/f'{name}_{state}';dest.mkdir(parents=True,exist_ok=True)
   r=copy.deepcopy(bare) if state=='bare' else ET.parse(source/'equipped.urdf').getroot()
   for joint in r.findall('joint'):
    if joint.get('name') in arm_joints and joint.find('origin') is not None:
     o=joint.find('origin');o.set('xyz',' '.join(map(str,np.fromstring(o.get('xyz','0 0 0'),sep=' ')*scale)))
   for link in r.findall('link'):
    for o in link.findall('.//origin'):
     if link.get('name') in arm_links:o.set('xyz',' '.join(map(str,np.fromstring(o.get('xyz','0 0 0'),sep=' ')*scale)))
    for mesh in link.findall('.//mesh'):
     mesh.set('filename',os.path.relpath((source/mesh.get('filename')).resolve(),dest))
     if link.get('name') in arm_links:mesh.set('scale',' '.join(map(str,np.fromstring(mesh.get('scale','1 1 1'),sep=' ')*scale)))
   locks={} if state=='bare' else meta['locked_joints'].copy();ee=meta['flange_link'];fingers=[]
   if state!='bare':
    mount={'external_piper':'gripper_base','external_ur5':'gripper_base_link','external_panda':'panda_hand'}[name]
    fingers={'external_piper':['link7','link8'],'external_ur5':['gripper_L1','gripper_L2'],'external_panda':['panda_leftfinger','panda_rightfinger']}[name]
    ee='review_grasp_tcp';link=ET.SubElement(r,'link',name=ee);joint=ET.SubElement(r,'joint',name='review_tcp_joint',type='fixed');ET.SubElement(joint,'parent',link=mount);ET.SubElement(joint,'child',link=ee);ET.SubElement(joint,'origin',xyz='0 0 0.1034' if name=='external_panda' else '0 0 0.155',rpy='0 0 0')
    if state=='holding':
     locks={'external_piper':{'joint7':.015,'joint8':-.015},'external_ur5':{'gripper_J1':.032697},'external_panda':{'panda_finger_joint1':.015}}[name]
     cube=dest/'payload.stl';trimesh.creation.box(extents=[.03]*3).export(cube)
     for tag in ['visual','collision']:element=ET.SubElement(link,tag);geom=ET.SubElement(element,'geometry');ET.SubElement(geom,'mesh',filename='payload.stl')
   ET.indent(r);ET.ElementTree(r).write(dest/'robot.urdf',encoding='utf-8',xml_declaration=True)
   m=dict(model=name,state=state,base=meta['base_link'],ee=ee,locks=locks,fingers=fingers,scale=float(scale),original_chain_length_m=float(length),normalized_chain_length_m=1.,source_revision=meta['source_revision'],urdf_sha256=hashlib.sha256((dest/'robot.urdf').read_bytes()).hexdigest(),tool_reference='OEM Panda Hand TCP: 103.4 mm along hand Z' if name=='external_panda' else '155 mm along native gripper-base Z; physical grasp centre needs confirmation',radius_upper_bound_m=1. if state=='bare' else 1.+.001+.155+.11)
   (dest/'model.json').write_text(json.dumps(m,indent=2)+'\n')


def summary(name,state,counts,extra):
 R=int((counts>0).sum());values=[int((counts>=k).sum()) for k in range(1,17)]
 r=dict(model=name,state=state,status='complete',R=R,D=values[-1],threshold_points=values,threshold_percent=[v/R*100 if R else 0. for v in values],**extra)
 folder=OUT/f'{name}_{state}';folder.mkdir(exist_ok=True);(folder/'summary.json').write_text(json.dumps(r,indent=2)+'\n');return r


def load(name,state):
 key=(name,state)
 if key in CACHE:return CACHE[key]
 folder=OUT/'models'/f'{name}_{state}';meta=json.loads((folder/'model.json').read_text());arm=RobotKinematics(folder/'robot.urdf',meta['base'],locked_joints=meta['locks']);checker=MeshCollisionChecker.from_urdf(folder/'robot.urdf');r=ET.parse(folder/'robot.urdf').getroot()
 graph={n:[] for n in arm.links}
 for j in r.findall('joint'):
  a,b=j.find('parent').get('link'),j.find('child').get('link');w=int(j.get('name') in arm.joint_names);graph[a].append((b,w));graph[b].append((a,w))
 ignores=[]
 for a in checker.names:
  stack=[(a,None,0)]
  while stack:
   b,parent,d=stack.pop()
   if b in checker.names and a<b and d<=1:ignores.append((a,b))
   for c,w in graph[b]:
    if c!=parent:stack.append((c,b,d+w))
 if state=='holding':
  ignores=[(a,b) for a,b in ignores if meta['ee'] not in (a,b)]
  ignores.extend((meta['ee'],f) for f in meta['fingers'])
 g=np.load(OUT/'grid.npz');positions=g['positions'];orientations=g['orientations_wxyz'];blocked=np.linalg.norm(positions,axis=1)>meta['radius_upper_bound_m']+.005
 p=arm.forward(arm.zero_positions())
 # A safe chain sphere centred at the first moving-joint origin. The origin
 # is fixed; downstream rotations preserve each translation's norm. This
 # is a necessary reach bound, not a sampled or learned reach estimate.
 chain=arm._chains[meta['ee']]
 first=next(i for i,j in enumerate(chain) if j.get('name') in arm.joint_names)
 centre=p[chain[first].find('child').get('link')][:3,3]
 radius=sum(np.linalg.norm(arm._origins[j.get('name')][:3,3]) for j in chain[first+1:])
 blocked|=np.linalg.norm(positions-centre,axis=1)>radius+.005
 for k,m in checker.meshes.items():
  if not arm.is_static(k):continue
  m=m.copy();m.apply_transform(p[k]);m.merge_vertices()
  candidate=np.where(np.all((positions>=m.bounds[0]-1e-6)&(positions<=m.bounds[1]+1e-6),axis=1))[0]
  if len(candidate):
   if m.is_watertight:blocked[candidate]|=m.contains(positions[candidate])
   _,distance,_=m.nearest.on_surface(positions[candidate]);blocked[candidate]|=distance<=1e-6
 CACHE[key]=(arm,checker,meta,ignores,positions,orientations,blocked);return CACHE[key]


def solve_cell(task):
 np.seterr(all='ignore');name,state,i=task;arm,checker,meta,ignores,positions,orientations,blocked=load(name,state)
 accepted=np.zeros(16,bool);solutions=np.full((16,len(arm.joint_names)),np.nan)
 if not blocked[i]:
  for j,o in enumerate(orientations):
   result=arm.solve(positions[i],meta['ee'],o,seeds=8,max_iterations=120,position_tolerance=.005,orientation_tolerance=.08,collision=lambda q:checker.check(arm.forward(q),ignores=ignores,clearance=1e-6),seed=i*16+j)
   if result.success:accepted[j]=True;solutions[j]=result.positions
 return i,accepted,solutions,bool(blocked[i])


def run(name,states,workers):
 grid=np.load(OUT/'grid.npz');positions=grid['positions'];orients=grid['orientations_wxyz']
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
  for state in states:
   dest=OUT/f'{name}_{state}';dest.mkdir(exist_ok=True);archive=dest/'workspace.npz';n=7 if name=='external_panda' else 6
   accepted=np.zeros((len(positions),16),bool);solutions=np.full((len(positions),16,n),np.nan);done=np.zeros(len(positions),bool);blocked=np.zeros(len(positions),bool)
   if archive.exists():
    a=np.load(archive)
    if 'processed' in a:accepted=a['accepted'];solutions=a['solutions'];done=a['processed'];blocked=a['blocked']
   if done.all() and (dest/'summary.json').exists():
    print('REUSE',name,state,flush=True);continue
   pending=np.where(~done)[0];start=time.time();last=start
   # cells are independent: completion order cannot affect seed or solution.
   for i,a,q,b in pool.map(solve_cell,[(name,state,int(i)) for i in pending],chunksize=8):
    accepted[i]=a;solutions[i]=q;done[i]=True;blocked[i]=b
    if time.time()-last>20 or done.all():
     temporary=dest/'workspace.partial.npz'
     np.savez_compressed(temporary,positions=positions,orientations_wxyz=orients,reachable_orientations=accepted.sum(1),accepted=accepted,solutions=solutions,processed=done,blocked=blocked)
     os.replace(temporary,archive)
     progress=dict(model=name,state=state,processed=int(done.sum()),total=len(done),reachable_so_far=int(accepted.any(1).sum()),elapsed_seconds=time.time()-start);(dest/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(progress,flush=True);last=time.time()
   if done.all():
    _,checker,meta,ignores,*_=load(name,state)
    r=summary(name,state,accepted.sum(1),dict(backend='CPU scipy bounded least-squares + FCL' + ('; checkpoint resumed with validated JIT FK/Jacobian' if os.environ.get('SPARK_BENCHMARK_JIT') == '1' else ''),grid_points=len(positions),iterations=120,seeds=8,position_tolerance_m=.005,orientation_tolerance_rad=.08,source=meta,ignored_pairs=ignores,open_mesh_links=checker.open_mesh_links,elapsed_seconds=time.time()-start));print('DONE',name,state,r['R'],r['D'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model',choices=EXTERNAL);p.add_argument('--states',nargs='+',choices=STATES,default=STATES);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
 if args.prepare:prepare()
 if args.model:run(args.model,args.states,args.workers)
