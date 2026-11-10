from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
old=out.parent/'gripper_demo'
rows=[];data={};sources={}
for i in range(1,5):
 for state in ['open','holding']:
  root=out if i==3 else old
  path=root/'output'/f'P2AB-{i}_{state}'/'grasp_tcp.npz'
  with np.load(path) as a:
   r=a['reachable_orientations'].astype(int);n=int(a['orientation_count'])
  assert n==16 and np.all((r>=0)&(r<=16))
  counts=np.bincount(r,minlength=17);assert counts.sum()==15625
  data[i,state]=counts;sources[f'P2AB-{i}_{state}']=str(path)
  for k,count in enumerate(counts):
   rows.append(dict(model=f'P2AB-{i}',state=state,reachable_directions=k,total_directions=16,coverage_percent=k/16*100,points=int(count),fraction_of_grid=float(count/len(r))))
with (out/'coverage_histogram.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(out/'coverage_histogram.json').write_text(json.dumps(dict(sources=sources,bins=rows),indent=2))
for first,suffix in [(0,'all_bins'),(1,'reachable_bins')]:
 fig,axes=plt.subplots(2,2,figsize=(18,10),layout='constrained',sharey=True)
 ymax=max(max(v[first:]) for v in data.values())
 for i,ax in enumerate(axes.flat,1):
  for j,state in enumerate(['open','holding']):
   counts=data[i,state];x=np.arange(first,17)
   bars=ax.bar(x+(j-.5)*.4,counts[first:],.4,color=['#3182bd','#f28e2b'][j],label='Open gripper' if j==0 else 'Holding 30 mm cube')
   ax.bar_label(bars,labels=[str(v) if v else '' for v in counts[first:]],rotation=90,padding=3,fontsize=8)
  a,b=data[i,'open'],data[i,'holding']
  ax.set_title(f'P2AB-{i} | R: {sum(a[1:]):,} / {sum(b[1:]):,} | D: {a[16]} / {b[16]} (open / holding)',fontsize=12)
  ax.set_xticks(np.arange(first,17));ax.set_xlabel('Reachable directions out of 16');ax.set_ylabel('Grid points');ax.set_ylim(0,ymax*1.24);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
  if first==1:ax.text(.98,.94,f'0/16 excluded here: {a[0]:,} / {b[0]:,}',ha='right',va='top',transform=ax.transAxes,fontsize=10)
 axes.flat[0].legend(loc='upper right',bbox_to_anchor=(1,.86))
 scope='All grid points, including 0/16' if first==0 else 'Reachable points only: detailed distribution (1/16 to 16/16)'
 fig.suptitle(f'{scope}\n1 m arm chain length + original-size Spark2 gripper | 10 cm grid | P2AB-3: revised upward mount',fontsize=16)
 fig.savefig(out/f'coverage_{suffix}.png',dpi=170);fig.savefig(out/f'coverage_{suffix}.pdf');plt.close(fig)
print(json.dumps({str(i):{s:dict(R=int(data[i,s][1:].sum()),D=int(data[i,s][16])) for s in ['open','holding']} for i in range(1,5)},indent=2))
