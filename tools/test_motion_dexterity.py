"""Analytic checks for the finite motion evaluator; run as a standalone unittest."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from collision_shpere_generation.kinematics import RobotKinematics
from collision_shpere_generation.mesh_collision import MeshCollisionChecker
import motion_dexterity as motion

class MotionChecks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();p=Path(self.temp.name)/'robot.urdf'
  p.write_text('<robot name="seam"><link name="base"/><link name="tip"/><joint name="j" type="revolute"><parent link="base"/><child link="tip"/><axis xyz="0 0 1"/><limit lower="-3.141592653589793" upper="3.141592653589793" effort="1" velocity="1"/></joint></robot>')
  self.arm=RobotKinematics(p,'base')
 def tearDown(self):self.temp.cleanup()
 def test_full_turn_seam_is_not_a_stop(self):
  q=motion.local_solve(self.arm,'tip',np.array([3.]),np.zeros(3),Rotation.from_euler('z',-3.).as_matrix())
  self.assertIsNotNone(q);self.assertGreater(q[0],np.pi);self.assertAlmostEqual(q[0],2*np.pi-3,places=5)
 def test_mesh_separation_matches_analytic_boxes(self):
  c=MeshCollisionChecker({name:trimesh.creation.box(extents=[.2]*3) for name in ['a','b']})
  t=np.eye(4);u=np.eye(4);u[0,3]=.5
  d,pair=motion.distances(c,{'a':t,'b':u},[]);self.assertAlmostEqual(d,.3,places=6);self.assertEqual(set(pair),{'a','b'})
 def test_path_interior_is_checked(self):
  class Clear:
   def check(self,*args,**kwargs):return False
  class Obstacle:
   def check(self,poses,**kwargs):
    angle=Rotation.from_matrix(poses['tip'][:3,:3]).as_rotvec()[2]%(2*np.pi)
    return abs(angle-3.2)<.015
  def analytic_target(*args):return np.zeros(3),Rotation.from_euler('z',3+.4*args[-1]).as_matrix()
  with patch.object(motion,'target',analytic_target):
   args=('tip',[],np.array([3.]),np.zeros(3),0,0,'MOT-02',0)
   nodes,_=motion.follow(self.arm,Clear(),*args);self.assertIsNotNone(nodes);self.assertGreater(nodes[-1][1][0],np.pi)
   nodes,failure=motion.follow(self.arm,Obstacle(),*args);self.assertIsNone(nodes);self.assertEqual(failure['category'],'path_collision')
if __name__=='__main__':unittest.main()
