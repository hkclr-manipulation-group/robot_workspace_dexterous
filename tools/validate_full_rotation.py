"""Independent FK/collision/SVD checks for unrestricted-joint samples."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from collision_shpere_generation.kinematics import RobotKinematics
from full_rotation_workspace import OUT,NAMES,ORIENTS,load,full_turn,previous,STATE
np.seterr(all='ignore');report={}
for name in NAMES:
 fast,c,ee,ignore,p,_=load(name);locks={}
 if name in previous.base.EXTERNAL:locks=json.loads((previous.base.OUT/'models'/f'{name}_{STATE}/model.json').read_text())['locks']
 arm=full_turn(RobotKinematics(fast.urdf,fast.base_link,locked_joints=locks));a=np.load(OUT/(name+'.npz'));assert a['processed'].all();np.testing.assert_allclose(a['positions'],p);np.testing.assert_allclose(a['orientations_wxyz'],ORIENTS)
 for i,j in zip(*np.where(a['accepted'])):
  q=a['solutions'][i,j];assert np.all(q>=-np.pi-1e-9) and np.all(q<=np.pi+1e-9);poses=arm.forward(q);t=poses[ee];assert np.linalg.norm(t[:3,3]-p[i])<=.005+1e-9;target=Rotation.from_quat(ORIENTS[j][[1,2,3,0]]).as_matrix();assert Rotation.from_matrix(t[:3,:3]@target.T).magnitude()<=.08+1e-9;assert not c.check(poses,ignores=ignore,clearance=1e-6);s=np.linalg.svd(arm.jacobian(q,ee),compute_uv=False);np.testing.assert_allclose(s[-1],a['sigma_min'][i,j],atol=1e-10);np.testing.assert_allclose(s[-1]/s[0],1/a['condition'][i,j],rtol=1e-5,atol=1e-12)
 report[name]=dict(status='PASS',accepted_configurations=int(a['accepted'].sum()));print(name,report[name],flush=True)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
