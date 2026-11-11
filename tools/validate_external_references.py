"""Recheck imported assets, FK samples, folds and every accepted IK solution."""
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from analyze_external_references import load,OUT,NAMES,ROOT
np.seterr(all='ignore');report={}
for name in NAMES:
 folder=ROOT/'collision_shpere_generation/models'/name;meta=json.loads((folder/'source_manifest.json').read_text());results=json.loads((OUT/name/'summary.json').read_text());checked=0
 for filename,sha in meta['urdf_sha256'].items():assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==sha
 for state in ['bare','open']:
  arm,checker,visual,m,ignores=load(name,state);a=np.load(OUT/name/(state+'.npz'));qs=a['joints'];assert len(qs)==8192 and np.isfinite(qs).all();assert np.all(qs>=arm.lower) and np.all(qs<=arm.upper)
  for i in np.linspace(0,len(qs)-1,32,dtype=int):
   p=arm.forward(qs[i]);np.testing.assert_allclose(p[m['flange_link']][:3,3],a['flange_positions'][i],atol=1e-10)
   assert bool(a['collision_free'][i])==(not checker.check(p,ignores=ignores,clearance=1e-6))
  for objective in ['fold','compact_fold']:
   f=results[state][objective];q=np.deg2rad(f['degrees']);assert np.all(q>=arm.lower-1e-9) and np.all(q<=arm.upper+1e-9);p=arm.forward(q)
   assert not checker.check(p,ignores=ignores,clearance=1e-6)
   vertices=np.concatenate([np.einsum('ij,nj->ni',p[k][:3,:3],v.vertices)+p[k][:3,3] for k,v in visual.items()]);size=np.ptp(vertices,axis=0)
   np.testing.assert_array_equal(np.ceil(size*1000).astype(int),f['dimensions_mm']);np.testing.assert_allclose(size*1000,f['exact_dimensions_mm'],atol=1e-7);assert abs(np.prod(size)*1000-f['volume_l'])<1e-7
   floor=min((np.einsum('ij,nj->ni',p[k][:3,:3],v.vertices)+p[k][:3,3])[:,2].min() for k,v in visual.items() if arm.is_static(k));assert vertices[:,2].min()>=floor-1e-7
 arm,checker,visual,m,ignores=load(name,'bare');a=np.load(OUT/name/'ik_section.npz');assert int(a['processed_points'])==42
 for i,j in zip(*np.where(a['passed'])):
  q=a['solutions'][i,j];assert np.all(q>=arm.lower-1e-9) and np.all(q<=arm.upper+1e-9);p=arm.forward(q);pose=p[m['flange_link']];assert np.linalg.norm(pose[:3,3]-a['positions'][i])<=.005+1e-9
  target=Rotation.from_quat(a['orientations_wxyz'][j][[1,2,3,0]]).as_matrix();assert Rotation.from_matrix(pose[:3,:3]@target.T).magnitude()<=.08+1e-9
  assert not checker.check(p,ignores=ignores,clearance=1e-6);checked+=1
 assert np.array_equal(a['passed'].sum(1),a['reachable_orientations'])
 report[name]=dict(status='PASS',fk_samples_rechecked=64,folds_rechecked=4,accepted_ik_solutions_rechecked=checked)
 print(name,report[name],flush=True)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
