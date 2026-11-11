"""Optional JIT FK/Jacobian adapter for offline benchmarks.
The bounded SciPy IK solver and its seeds/tolerances remain unchanged.
Validate against RobotKinematics before using results.
"""
import numpy as np
from numba import njit
from collision_shpere_generation.kinematics import RobotKinematics

@njit(cache=True)
def matmul4(a,b):
 c=np.zeros((4,4))
 for i in range(4):
  for j in range(4):
   for k in range(4):c[i,j]+=a[i,k]*b[k,j]
 return c

@njit(cache=True)
def fk(q,base,rootidx,nlinks,parents,children,origins,axes,kinds,drivers,mults,offsets):
 poses=np.zeros((nlinks,4,4));poses[rootidx]=base;frames=np.zeros_like(origins)
 for j in range(len(parents)):
  frame=matmul4(poses[parents[j]],origins[j]);frames[j]=frame
  v=offsets[j]+(q[drivers[j]]*mults[j] if drivers[j]>=0 else 0.)
  motion=np.eye(4)
  if kinds[j]==2:
   for i in range(3):motion[i,3]=axes[j,i]*v
  elif kinds[j]==1:
   x,y,z=axes[j];c=np.cos(v);s=np.sin(v);t=1-c
   motion[0,0]=c+x*x*t;motion[0,1]=x*y*t-z*s;motion[0,2]=x*z*t+y*s
   motion[1,0]=y*x*t+z*s;motion[1,1]=c+y*y*t;motion[1,2]=y*z*t-x*s
   motion[2,0]=z*x*t-y*s;motion[2,1]=z*y*t+x*s;motion[2,2]=c+z*z*t
  poses[children[j]]=matmul4(frame,motion)
 return poses,frames

@njit(cache=True)
def jac(poses,frames,axes,kinds,drivers,mults,chain,tip,n):
 result=np.zeros((6,n))
 for j in chain:
  idx=drivers[j];mult=mults[j]
  if idx<0 or mult==0:continue
  axis=np.zeros(3)
  for i in range(3):
   for k in range(3):axis[i]+=frames[j,i,k]*axes[j,k]
  if kinds[j]==2:
   for i in range(3):result[i,idx]+=mult*axis[i]
  else:
   d=poses[tip,:3,3]-frames[j,:3,3]
   result[0,idx]+=mult*(axis[1]*d[2]-axis[2]*d[1]);result[1,idx]+=mult*(axis[2]*d[0]-axis[0]*d[2]);result[2,idx]+=mult*(axis[0]*d[1]-axis[1]*d[0])
   for i in range(3):result[i+3,idx]+=mult*axis[i]
 return result

class FastReferenceKinematics(RobotKinematics):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self._linkindex={k:i for i,k in enumerate(self.links)};self._jindex={j.get('name'):i for i,j in enumerate(self._joints)}
  self._parent=np.array([self._linkindex[j.find('parent').get('link')] for j in self._joints]);self._child=np.array([self._linkindex[j.find('child').get('link')] for j in self._joints]);self._origin=np.array([self._origins[j.get('name')] for j in self._joints]);self._axis=np.array([self._axes[j.get('name')] for j in self._joints]);self._kind=np.array([2 if j.get('type')=='prismatic' else 0 if j.get('type')=='fixed' else 1 for j in self._joints]);d=[self._drivers[j.get('name')] for j in self._joints];self._driver=np.array([x[0] for x in d]);self._mult=np.array([x[1] for x in d]);self._offset=np.array([x[2] for x in d]);self._chainindices={k:np.array([self._jindex[j.get('name')] for j in chain],dtype=np.int64) for k,chain in self._chains.items()};self._fast_ready=True;self._cached_q=None
 def _state(self,q):
  if not hasattr(self,'_fast_ready'):return super()._state(q)
  q=np.asarray(q,dtype=float)
  if q.shape!=(len(self.joint_names),) or not np.isfinite(q).all():raise ValueError('Invalid joint vector')
  if self._cached_q is not None and np.array_equal(q,self._cached_q):return self._cached_state
  a,b=fk(q,self._base_inverse,self._linkindex[self.root_link],len(self.links),self._parent,self._child,self._origin,self._axis,self._kind,self._driver,self._mult,self._offset)
  self._fast_arrays=(a,b);self._cached_q=q.copy();self._cached_state=({k:a[i] for k,i in self._linkindex.items()},{k:b[i] for k,i in self._jindex.items()});return self._cached_state
 def jacobian(self,q,link):
  self._state(q);a,b=self._fast_arrays
  return jac(a,b,self._axis,self._kind,self._driver,self._mult,self._chainindices[link],self._linkindex[link],len(self.joint_names))
