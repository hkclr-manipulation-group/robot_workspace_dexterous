"""Validate common sampling and every accepted external solution before plotting."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compare_nine_workspace import OUT,OLD,OURS,EXTERNAL,STATES,load
np.seterr(all='ignore');grid=np.load(OUT/'grid.npz');positions=grid['positions'];orientations=grid['orientations_wxyz'];report={}
assert len(positions)==2197 and len(np.unique(np.round(positions,6),axis=0))==2197
for name in OURS:
 for state in STATES:
  folder=OUT/f'{name}_{state}';r=json.loads((folder/'summary.json').read_text());source=Path(__file__).resolve().parents[2]/r['original_npz'];assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
  old=np.load(source);mask=np.all(np.isclose(old['positions']/.2,np.round(old['positions']/.2),atol=1e-5),axis=1);a=np.load(folder/'workspace.npz');np.testing.assert_allclose(a['positions'],positions,atol=1e-8);np.testing.assert_array_equal(a['reachable_orientations'],old['reachable_orientations'][mask]);np.testing.assert_allclose(old['positions'][mask],positions,atol=1e-8)
  report[f'{name}_{state}']=dict(status='PASS',archived_exact_grid_subset=True)
for name in EXTERNAL:
 for state in STATES:
  folder=OUT/f'{name}_{state}';r=json.loads((folder/'summary.json').read_text());assert r['status']=='complete';a=np.load(folder/'workspace.npz');assert a['processed'].all();np.testing.assert_allclose(a['positions'],positions,atol=1e-8);np.testing.assert_allclose(a['orientations_wxyz'],orientations,atol=1e-10);np.testing.assert_array_equal(a['accepted'].sum(1),a['reachable_orientations']);assert int(a['accepted'].any(1).sum())==r['R'];assert int(a['accepted'].all(1).sum())==r['D']
  arm,checker,meta,ignores,*_=load(name,state);max_p=max_r=0.;checked=0
  assert hashlib.sha256((OUT/'models'/f'{name}_{state}/robot.urdf').read_bytes()).hexdigest()==meta['urdf_sha256']
  for i,j in zip(*np.where(a['accepted'])):
   q=a['solutions'][i,j];assert np.all(q>=arm.lower-1e-9) and np.all(q<=arm.upper+1e-9);p=arm.forward(q);pose=p[meta['ee']];pe=float(np.linalg.norm(pose[:3,3]-positions[i]));target=Rotation.from_quat(orientations[j][[1,2,3,0]]).as_matrix();re=float(Rotation.from_matrix(pose[:3,:3]@target.T).magnitude());assert pe<=.005+1e-9 and re<=.08+1e-9;assert not checker.check(p,ignores=ignores,clearance=1e-6);max_p=max(max_p,pe);max_r=max(max_r,re);checked+=1
  report[f'{name}_{state}']=dict(status='PASS',accepted_poses_rechecked=checked,max_position_error_m=max_p,max_orientation_error_rad=max_r)
  print(name,state,report[f'{name}_{state}'],flush=True)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('All 27 datasets validated')
