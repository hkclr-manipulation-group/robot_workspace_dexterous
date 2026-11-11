"""Refine native fold candidates using constrained evolutionary search."""
import sys,json,time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from analyze_external_references import load,OUT
np.seterr(all='ignore');name=sys.argv[1];path=OUT/name/'summary.json';data=json.loads(path.read_text())
for state in ['bare','open']:
 arm,checker,visual,meta,ignores=load(name,state);hulls={k:m.vertices[ConvexHull(m.vertices).vertices] for k,m in visual.items()};ref=arm.forward(arm.zero_positions())
 def points(k,p):return np.einsum('ij,nj->ni',p[k][:3,:3],hulls[k])+p[k][:3,3]
 floor=min(points(k,ref)[:,2].min() for k in hulls if arm.is_static(k))
 best=data[state]['fold'];best_q=np.deg2rad(best['degrees']);best_vol=best['volume_l']/1000
 def envelope(q):
  p=arm.forward(q);lo=[];hi=[]
  for k in hulls:
   v=points(k,p);lo.append(v.min(0));hi.append(v.max(0))
  return np.min(lo,0),np.max(hi,0)
 def objective(q):
  global best_q,best_vol
  lo,hi=envelope(q);vol=float(np.prod(hi-lo))
  if lo[2]<floor-1e-7:return 1+vol+floor-lo[2]
  if checker.check(arm.forward(q),ignores=ignores,clearance=1e-6):return 1+vol
  if vol<best_vol:best_q,best_vol=q.copy(),vol
  return vol
 start=time.time()
 for seed in [20261111,20261112]:
  differential_evolution(objective,list(zip(arm.lower,arm.upper)),seed=seed,maxiter=100,popsize=12,x0=best_q,polish=False,tol=1e-5,atol=0)
  print(name,state,'DE',seed,round(best_vol*1000,2),flush=True)
 lo,hi=envelope(best_q)
 assert not checker.check(arm.forward(best_q),ignores=ignores,clearance=1e-6)
 best.update(degrees=np.rad2deg(best_q).tolist(),dimensions_mm=np.ceil((hi-lo)*1000).astype(int).tolist(),exact_dimensions_mm=((hi-lo)*1000).tolist(),bounds_mm=(np.array([lo,hi])*1000).tolist(),volume_l=best_vol*1000,search=best['search']+'; two differential-evolution runs (100 generations, population multiplier 12)',refinement_seconds=time.time()-start)
 path.write_text(json.dumps(data,indent=2)+'\n');print('REFINED',name,state,best['dimensions_mm'],flush=True)
