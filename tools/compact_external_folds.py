"""Find compact storage candidates by minimising the longest envelope edge.
Keep the volume-first candidates separately for a transparent objective comparison.
"""
import sys,json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution,minimize
from analyze_external_references import load,OUT
np.seterr(all='ignore');name=sys.argv[1];path=OUT/name/'summary.json';data=json.loads(path.read_text())
for state in ['bare','open']:
 arm,checker,visual,meta,ignores=load(name,state);hulls={k:m.vertices[ConvexHull(m.vertices).vertices] for k,m in visual.items()};ref=arm.forward(arm.zero_positions())
 def points(k,p):return np.einsum('ij,nj->ni',p[k][:3,:3],hulls[k])+p[k][:3,3]
 floor=min(points(k,ref)[:,2].min() for k in hulls if arm.is_static(k))
 def envelope(q):
  p=arm.forward(q);bounds=np.array([[points(k,p).min(0),points(k,p).max(0)] for k in hulls]);return bounds[:,0].min(0),bounds[:,1].max(0)
 a=np.load(OUT/name/(state+'.npz'));best_score=float('inf');best_q=None
 # Start from verified valid samples, not only the volume-optimal shape.
 for q in np.concatenate([a['joints'][a['collision_free']],np.deg2rad([data[state]['fold']['degrees']])]):
  lo,hi=envelope(q);score=float(max(hi-lo))
  if lo[2]>=floor-1e-7 and score<best_score:best_score,best_q=score,q.copy()
 def objective(q):
  global best_score,best_q
  lo,hi=envelope(q);score=float(max(hi-lo))
  if lo[2]<floor-1e-7:return 10+floor-lo[2]+score
  if checker.check(arm.forward(q),ignores=ignores,clearance=1e-6):return 10+score
  if score<best_score:best_score,best_q=score,q.copy()
  return score
 for seed in [20261113,20261114]:
  differential_evolution(objective,list(zip(arm.lower,arm.upper)),seed=seed,maxiter=100,popsize=12,x0=best_q,polish=False,tol=1e-5)
 minimize(objective,best_q,method='Powell',bounds=list(zip(arm.lower,arm.upper)),options={'maxfev':600})
 lo,hi=envelope(best_q);assert not checker.check(arm.forward(best_q),ignores=ignores,clearance=1e-6)
 data[state]['compact_fold']=dict(degrees=np.rad2deg(best_q).tolist(),dimensions_mm=np.ceil((hi-lo)*1000).astype(int).tolist(),exact_dimensions_mm=((hi-lo)*1000).tolist(),bounds_mm=(np.array([lo,hi])*1000).tolist(),volume_l=float(np.prod(hi-lo)*1000),contacts=[],below_base_mm=max(0,float((floor-lo[2])*1000)),search='Minimum longest base-axis envelope edge; feasible Sobol starts + two differential evolution runs (100 generations/popsize12) + bounded Powell. Best found, not certified global minimum.')
 path.write_text(json.dumps(data,indent=2)+'\n');print('COMPACT',name,state,data[state]['compact_fold']['dimensions_mm'],round(data[state]['compact_fold']['volume_l'],2),flush=True)
