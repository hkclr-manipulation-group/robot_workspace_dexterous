from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
root=out/'output'
records=json.loads((out/'normalization.json').read_text())
assert all(abs(item['normalized_chain_length_m']-1.0)<1e-9 for item in records)
models=[]
for i in range(1,5):
 ee='L'+'6'*i
 with np.load(root/f'P2AB-{i}_preview'/f'{ee}.npz') as d:
  models.append((d['positions'].copy(),d['dexterity'].copy(),d['reachable_orientations'].copy()))
levels=np.unique(models[0][0][:,1])
for p,s,r in models:
 np.testing.assert_array_equal(np.unique(p[:,1]),levels)
fig,axes=plt.subplots(4,1,figsize=(25,12),layout='compressed')
from matplotlib.patches import Polygon
for i,(ax,model) in enumerate(zip(axes,models),1):
 p,s,r=model
 kept_levels=[y for y in levels if any(np.any((m[0][:,1]==y)&(m[2]>0)) for m in models)]
 for k,y in enumerate(kept_levels):
  center=k*1.25
  corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
  ax.add_patch(Polygon(np.column_stack((center+.43*corners[:,0],corners[:,1]+.3*corners[:,0])),facecolor='#f5f8fb',edgecolor='#c4ced9',lw=.6,zorder=0))
  mask=(p[:,1]==y)&(r>0)
  art=ax.scatter(center+.43*p[mask,0],p[mask,2]+.3*p[mask,0],c=s[mask],s=8,cmap='turbo',vmin=0,vmax=1)
  ax.text(center,-1.62,f'{0.0 if abs(y)<1e-6 else float(y):+.1f}',ha='center',fontsize=9)
 ax.set_xlim(-.65,(len(kept_levels)-1)*1.25+.65);ax.set_ylim(-1.9,1.55);ax.set_aspect('equal');ax.set_anchor('W');ax.axis('off')
 ax.set_title(f'P2AB-{i} | normalized chain length = 1 m | R={int((r>0).sum())}, D={int((r==16).sum())}',loc='left',fontsize=13)
fig.colorbar(art,ax=axes.tolist(),label='Orientation coverage',shrink=.8,pad=.01,ticks=[0,.25,.5,.75,1])
fig.suptitle('Equal-length models: joint-offset chain length normalized to 1 m\nRecomputed IK + STL collision | Y interval: 0.1 m | Same XZ scale | Display gaps are not physical distances',fontsize=16)
fig.savefig(out/'p2ab_equal_length_slices.png',dpi=180,bbox_inches='tight',pad_inches=.06)
fig.savefig(out/'p2ab_equal_length_slices.pdf',bbox_inches='tight',pad_inches=.06)
plt.close(fig)
