from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
out=Path(__file__).parent
rows=[]
for state in ['open','holding']:
 models=[]
 for i in range(1,5):
  source=out if i==3 else out.parent/'gripper_demo'
  path=source/'output'/f'P2AB-{i}_{state}'
  summary=json.loads((path/'grasp_tcp_summary.json').read_text())
  assert summary['status']=='complete'
  assert summary['collision_model']['self_collision_enabled']
  with np.load(path/'grasp_tcp.npz') as d:
   p=d['positions'].copy(); s=d['dexterity'].copy(); r=d['reachable_orientations'].copy()
  R=int((r>0).sum()); D=int((r==16).sum())
  assert R==summary['rws_cells'] and D==summary['dws_cells']
  rows.append(dict(model=f'P2AB-{i}',state=state,R=R,D=D,D_R_percent=100*D/R if R else 0))
  models.append((p,s,r))
 levels=np.unique(models[0][0][:,1])
 levels=[y for y in levels if any(np.any((p[:,1]==y)&(r>0)) for p,s,r in models)]
 fig,axes=plt.subplots(4,1,figsize=(26,12),layout='compressed')
 for i,(ax,(p,s,r)) in enumerate(zip(axes,models),1):
  for k,y in enumerate(levels):
   center=k*1.3
   corners=np.array([[-1.2,-1.2],[1.2,-1.2],[1.2,1.2],[-1.2,1.2]])
   ax.add_patch(Polygon(np.column_stack((center+.43*corners[:,0],corners[:,1]+.3*corners[:,0])),facecolor='#f5f8fb',edgecolor='#c4ced9',lw=.6,zorder=0))
   mask=(p[:,1]==y)&(r>0)
   art=ax.scatter(center+.43*p[mask,0],p[mask,2]+.3*p[mask,0],c=s[mask],s=7,cmap='turbo',vmin=0,vmax=1)
   ax.text(center,-1.76,f'{0 if abs(y)<1e-6 else float(y):+.1f}',ha='center',fontsize=8)
  ax.set_xlim(-.6,(len(levels)-1)*1.3+.6); ax.set_ylim(-1.95,1.65);ax.set_aspect('equal');ax.axis('off')
  ax.set_title(f'P2AB-{i} | R = {int((r>0).sum()):,} | D (16/16) = {int((r==16).sum()):,}',loc='left',fontsize=14)
 fig.colorbar(art,ax=axes.tolist(),label='Reachable orientation fraction (16 samples)',shrink=.8,pad=.01)
 fig.suptitle(f'Spark2 gripper: {state} | arm chain length = 1 m, original-size gripper; P2AB-3 mount revised\nXZ slices along Y (labels in m), 0.1 m interval | same XZ scale | display separation is illustrative',fontsize=16)
 fig.savefig(out/f'{state}_workspace_slices.pdf',bbox_inches='tight',pad_inches=.06)
 fig.savefig(out/f'{state}_workspace_slices.png',dpi=160,bbox_inches='tight',pad_inches=.06)
 plt.close(fig)
(out/'workspace_counts.json').write_text(json.dumps(rows,indent=2))
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,key,title in zip(axes,['R','D'],['Total reachable points R','Fully dexterous points D (16/16)']):
 for j,state in enumerate(['open','holding']):
  vals=[r[key] for r in rows if r['state']==state]
  bars=ax.bar(np.arange(4)+(j-.5)*.36,vals,.36,label=state,color=['#3182bd','#f28e2b'][j])
  ax.bar_label(bars,padding=3)
 ax.set_xticks(np.arange(4),[f'P2AB-{i}' for i in range(1,5)]);ax.set_title(title);ax.set_ylabel('Grid points');ax.margins(y=.2);ax.legend();ax.spines[['top','right']].set_visible(False)
fig.suptitle('Equal-length arms + Spark2 gripper | 0.1 m grid, 16 orientations, 8 IK seeds\nR: at least one accepted orientation; D: all 16 accepted orientations')
fig.savefig(out/'workspace_counts.png',dpi=180)
print(json.dumps(rows,indent=2))
