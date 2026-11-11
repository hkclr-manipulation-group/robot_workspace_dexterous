"""Finite-seed native Y=0 pose coverage for imported external references."""
import sys,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_external_references import load,OUT
name=sys.argv[1];np.seterr(all='ignore')
arm,checker,visual,meta,ignores=load(name,'bare')
positions=np.array([[x,0,z] for z in np.arange(-.3,.71,.2) for x in np.arange(-.6,.61,.2)])
xyzw=Rotation.random(16,random_state=20261111).as_quat();orientations=xyzw[:,[3,0,1,2]]
counts=np.zeros(len(positions),int);passed=np.zeros((len(positions),16),bool);solutions=np.full((len(positions),16,len(arm.joint_names)),np.nan)
blocked=np.zeros(len(positions),bool);ref=arm.forward(arm.zero_positions())
for k,m in checker.meshes.items():
 if not arm.is_static(k):continue
 m=m.copy();m.apply_transform(ref[k]);m.merge_vertices()
 if m.is_watertight:blocked|=m.contains(positions)
 _,dist,_=m.nearest.on_surface(positions);blocked|=dist<=1e-6
start=time.time()
for i,p in enumerate(positions):
 if not blocked[i]:
  for j,o in enumerate(orientations):
   result=arm.solve(p,meta['flange_link'],o,seeds=8,max_iterations=120,position_tolerance=.005,orientation_tolerance=.08,collision=lambda q:checker.check(arm.forward(q),ignores=ignores,clearance=1e-6),seed=20261111+i*16+j)
   if result.success:passed[i,j]=True;solutions[i,j]=result.positions
 counts[i]=int(passed[i].sum())
 print(name,'IK',i+1,len(positions),'orientations',counts[i],'elapsed',round(time.time()-start),flush=True)
 np.savez_compressed(OUT/name/'ik_section.npz',positions=positions,orientations_wxyz=orientations,reachable_orientations=counts,passed=passed,solutions=solutions,processed_points=i+1,blocked_by_fixed_mesh=blocked)
summary=dict(model=name,status='complete',positions=len(positions),orientations=16,seeds=8,max_iterations=120,position_tolerance_m=.005,orientation_tolerance_rad=.08,joint_limits='source URDF native',reachable_points=int((counts>0).sum()),at_least_8=int((counts>=8).sum()),all_16=int((counts==16).sum()),blocked_points=int(blocked.sum()),elapsed_seconds=time.time()-start,collision='vendor triangle meshes + FCL/closed-mesh containment; rigid/adjacent pairs ignored',note='Finite seed section only; failures do not prove unreachable. Separate from archived 1 m chain statistics.')
(OUT/name/'ik_section.json').write_text(json.dumps(summary,indent=2)+'\n')
