from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Polygon
out=Path(__file__).parent;old=out.parent
names=['spark2_v1','spark2_v2_2']+[f'P2AB-{i}' for i in range(1,5)];labels=['Spark2 v1','Spark2 v2.2']+names[2:]
states=['bare','open','holding'];titles=['Bare arm (flange target)','Open gripper (grasp TCP)','Holding 30 mm cube (grasp TCP)']
colors=['#6a51a3','#222222','#0072B2','#E69F00','#009E73','#CC79A7'];data={};summary=[];bins=[];provenance={};reference=None
for name in names:
 for state in states:
  root=out
  folder=root/'output'/f'{name}_{state}';files=list(folder.glob('*_summary.json'));assert len(files)==1,folder
  meta=json.loads(files[0].read_text());assert meta['status']=='complete' and meta['collision_model']['self_collision_enabled']
  file=folder/(files[0].name.replace('_summary.json','.npz'))
  with np.load(file) as a:p=a['positions'].copy();r=a['reachable_orientations'].copy().astype(int);s=a['dexterity'].copy()
  assert len(r)==15625 and meta['orientation_samples']==16 and meta['ik_seeds']==8
  current=(p,meta['orientation_samples_wxyz'])
  if reference is None:reference=current
  else:
   np.testing.assert_allclose(reference[0],p,atol=1e-6)
   np.testing.assert_allclose(reference[1],current[1],atol=1e-7)
  counts=np.bincount(r,minlength=17);R=int((r>0).sum());D=int(counts[16]);cum=np.array([counts[k:].sum()/R*100 for k in range(16,0,-1)])
  assert R==meta['rws_cells'] and D==meta['dws_cells'] and np.isclose(cum[-1],100)
  data[name,state]=(p,s,r,cum);provenance[f'{name}_{state}']=str(file)
  summary.append(dict(model=name,state=state,R=R,D=D,at_least_50_percent=float(cum[8]),at_least_75_percent=float(cum[4]),full_coverage_percent=float(cum[0]),mean_orientation_coverage_percent=float(s[r>0].mean()*100)))
  for t,v in zip(range(16,0,-1),cum):bins.append(dict(model=name,state=state,threshold=t,threshold_coverage_percent=t/16*100,cumulative_percent=float(v),cumulative_points=int(counts[t:].sum()),R=R))
for filename,rows in [('summary.csv',summary),('cumulative_percent.csv',bins)]:
 with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(out/'sources.json').write_text(json.dumps(provenance,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(19,6),sharey=True,layout='constrained')
for ax,state,title in zip(axes,states,titles):
 for name,label,color in zip(names,labels,colors):
  ax.plot(np.arange(16),data[name,state][3],label=label,color=color,lw=2.7 if name=='spark2_v2_2' else 2,marker='o',ms=3)
 ax.set_title(title);ax.set_xticks(np.arange(16),[f'≥{t}' for t in range(16,0,-1)],rotation=55);ax.set_xlabel('Required directions (out of 16)');ax.set_ylim(-2,103);ax.grid(alpha=.2);ax.yaxis.set_major_formatter(PercentFormatter(100))
axes[0].set_ylabel('Cumulative share of reachable points');axes[0].legend(loc='upper left',fontsize=10)
fig.suptitle('Six-model comparison | cumulative orientation coverage\n1 m arm chain length • same STL collision policy • 10 cm grid • 16 orientations • 8 IK seeds',fontsize=17)
fig.savefig(out/'cumulative_curves.png',dpi=180);fig.savefig(out/'cumulative_curves.pdf');plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(18,10),layout='constrained')
for ax,state,title in zip(axes,states,titles):
 matrix=np.array([data[n,state][3] for n in names]);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
 ax.set_yticks(range(6),labels);ax.set_xticks(range(16),[f'≥{t}/16' for t in range(16,0,-1)]);ax.set_title(title,loc='left',weight='bold')
 for y in range(6):
  for x in range(16):ax.text(x,y,(f'{matrix[y,x]:.2f}%' if 0 < matrix[y,x] < 1 else f'{matrix[y,x]:.1f}%'),ha='center',va='center',fontsize=9,color='white' if matrix[y,x]>55 else '#172b3a')
fig.colorbar(im,ax=axes,label='Share of reachable points (%)',shrink=.9,pad=.01)
fig.suptitle('Cumulative percentages | denominator: reachable points only (0/16 excluded)',fontsize=17)
fig.savefig(out/'cumulative_heatmap.png',dpi=180);fig.savefig(out/'cumulative_heatmap.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
for ax,key,title in zip(axes,['R','D'],['Total reachable points R','All 16 directions reachable D']):
 for j,(state,col) in enumerate(zip(states,['#aab4be','#3182bd','#f28e2b'])):
  vals=[next(r[key] for r in summary if r['model']==n and r['state']==state) for n in names]
  bars=ax.bar(np.arange(6)+(j-1)*.25,vals,.25,label=state,color=col);ax.bar_label(bars,padding=3,fontsize=9,rotation=90)
 ax.set_xticks(range(6),labels);ax.set_ylabel('Grid points');ax.set_title(title);ax.margins(y=.25);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].legend();fig.suptitle('Equal arm chain length (1 m) | bare target: flange; gripper target: grasp TCP',fontsize=15)
fig.savefig(out/'point_counts.png',dpi=180);fig.savefig(out/'point_counts.pdf');plt.close(fig)
# Preserve the user's original per-model cumulative bar chart style.
for selected,suffix in [(['bare'],'bare'),(['open','holding'],'gripper')]:
 fig,axes=plt.subplots(3,2,figsize=(18,14),layout='constrained',sharey=True)
 for ax,name,label in zip(axes.flat,names,labels):
  totals=[]
  for j,state in enumerate(selected):
   p,scores,r,cum=data[name,state];totals.append(int(sum(r>0)))
   shift=0 if len(selected)==1 else (j-.5)*.4
   bars=ax.bar(np.arange(16)+shift,cum,.65 if len(selected)==1 else .4,color=['#3182bd','#f28e2b'][j],label={'bare':'Bare arm','open':'Open gripper','holding':'Holding 30 mm cube'}[state])
   ax.bar_label(bars,labels=[f'{v:.2f}%' for v in cum],rotation=90,padding=3,fontsize=8)
  ax.set_title(f'{label} | R = '+ ' / '.join(f'{v:,}' for v in totals),loc='left',fontsize=12)
  ax.set_xticks(np.arange(16),[f'≥{t}/16' for t in range(16,0,-1)],rotation=55,ha='right');ax.set_ylim(0,118)
  ax.set_ylabel('Cumulative share of reachable points (%)');ax.yaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 axes.flat[0].legend(loc='upper left')
 fig.suptitle('Six-model cumulative comparison | '+('bare arms' if suffix=='bare' else 'open gripper / holding cube')+'\n≥16/16 → ≥1/16 | denominator: reachable points only | equal arm chain length 1 m',fontsize=17)
 fig.savefig(out/f'cumulative_bars_{suffix}.png',dpi=180);fig.savefig(out/f'cumulative_bars_{suffix}.pdf');plt.close(fig)

# Same section locations and geometry scale for all states and models.
levels=np.unique(data[names[0],states[0]][0][:,1]);levels=[y for y in levels if any(np.any((p[:,1]==y)&(r>0)) for p,s,r,c in data.values())]
for state,title in zip(states,titles):
 fig,axes=plt.subplots(6,1,figsize=(26,16),layout='compressed')
 for ax,name,label in zip(axes,names,labels):
  p,s,r,c=data[name,state]
  for k,y in enumerate(levels):
   center=k*1.25;corners=np.array([[-1.2,-1.2],[1.2,-1.2],[1.2,1.2],[-1.2,1.2]])
   ax.add_patch(Polygon(np.column_stack((center+.43*corners[:,0],corners[:,1]+.3*corners[:,0])),facecolor='#f5f8fb',edgecolor='#c4ced9',lw=.55,zorder=0))
   mask=np.isclose(p[:,1],y,atol=1e-5)&(r>0);im=ax.scatter(center+.43*p[mask,0],p[mask,2]+.3*p[mask,0],c=s[mask]*100,s=7,cmap='turbo',vmin=0,vmax=100)
   ax.text(center,-1.74,f'{0 if abs(y)<1e-5 else y:+.1f}',ha='center',fontsize=8)
  ax.set_xlim(-.6,(len(levels)-1)*1.25+.6);ax.set_ylim(-1.94,1.65);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{label} | R={sum(r>0):,} | D={sum(r==16):,}',loc='left',fontsize=14)
 fig.colorbar(im,ax=axes.tolist(),label='Orientation coverage (%)',shrink=.85,pad=.01)
 fig.suptitle(f'{title} | six equal-length arms\nXZ sections along Y every 0.1 m | equal scale | illustrative separation between sections',fontsize=17)
 fig.savefig(out/f'workspace_{state}.png',dpi=160,bbox_inches='tight');plt.close(fig)
print(json.dumps(summary,indent=2))
