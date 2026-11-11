from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from collision_shpere_generation.kinematics import RobotKinematics
from six_direction_grasp import OUT,NAMES,ORIENTS,load,base,STATE
np.seterr(all='ignore');report={}
for name in NAMES:
 fast,c,ee,ignores,positions,_=load(name);locks={}
 if name in base.EXTERNAL:locks=json.loads((base.OUT/'models'/f'{name}_{STATE}/model.json').read_text())['locks']
 arm=RobotKinematics(fast.urdf,fast.base_link,locked_joints=locks);a=np.load(OUT/(name+'.npz'));assert a['processed'].all();np.testing.assert_allclose(a['positions'],positions);np.testing.assert_allclose(a['orientations_wxyz'],ORIENTS);count=0
 for i,j in zip(*np.where(a['accepted'])):
  q=a['solutions'][i,j];assert np.all(q>=arm.lower-1e-9) and np.all(q<=arm.upper+1e-9);p=arm.forward(q);pose=p[ee];assert np.linalg.norm(pose[:3,3]-positions[i])<=.005+1e-9
  target=Rotation.from_quat(ORIENTS[j][[1,2,3,0]]).as_matrix();assert Rotation.from_matrix(pose[:3,:3]@target.T).magnitude()<=.08+1e-9;assert not c.check(p,ignores=ignores,clearance=1e-6)
  s=np.linalg.svd(arm.jacobian(q,ee),compute_uv=False);np.testing.assert_allclose(s[-1],a['sigma_min'][i,j],atol=1e-10);np.testing.assert_allclose(s[-1]/s[0],1/a['condition'][i,j],rtol=1e-5,atol=1e-12);count+=1
 report[name]=dict(status='PASS',accepted_configurations=count);print(name,count,'PASS',flush=True)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
