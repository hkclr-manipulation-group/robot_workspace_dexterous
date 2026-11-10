from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
out=Path(__file__).parent
rows=list(csv.DictReader((out/'coverage_histogram.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(18,10),layout='constrained',sharey=True)
export=[]
for i,ax in enumerate(axes.flat,1):
 totals=[]
 for j,state in enumerate(['open','holding']):
  selected=[r for r in rows if r['model']==f'P2AB-{i}' and r['state']==state and int(r['reachable_directions'])>0]
  selected.sort(key=lambda r:int(r['reachable_directions']))
  counts=np.array([int(r['points']) for r in selected]);total=counts.sum();totals.append(total)
  pct=100*counts/total;assert np.isclose(pct.sum(),100)
  x=np.arange(1,17)
  bars=ax.bar(x+(j-.5)*.4,pct,.4,color=['#3182bd','#f28e2b'][j],label='Open gripper' if j==0 else 'Holding 30 mm cube')
  ax.bar_label(bars,labels=[f'{v:.2f}%' for v in pct],rotation=90,padding=3,fontsize=8)
  for r,v in zip(selected,pct):export.append(dict(model=r['model'],state=state,coverage_percent=r['coverage_percent'],points=r['points'],total_reachable_points=int(total),percent_of_reachable=float(v)))
 ax.set_title(f'P2AB-{i} | reachable points: {totals[0]:,} / {totals[1]:,} (open / holding)',fontsize=12)
 ax.set_xticks(x,[f'{k/16*100:g}%' for k in x],rotation=55,ha='right')
 ax.set_xlabel('Orientation coverage (1/16 to 16/16)');ax.set_ylabel('Share of reachable points (%)');ax.yaxis.set_major_formatter(PercentFormatter(100));ax.set_ylim(0,19)
 ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
axes.flat[0].legend(loc='upper right')
fig.suptitle('Distribution among reachable points only (0/16 excluded)\nEach color sums to 100% per model | 1 m arm chain length + Spark2 gripper | revised P2AB-3 mount',fontsize=16)
fig.savefig(out/'coverage_reachable_percent.png',dpi=170);fig.savefig(out/'coverage_reachable_percent.pdf')
with (out/'coverage_reachable_percent.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=export[0]);w.writeheader();w.writerows(export)
print('Saved percentage figure, PDF and CSV; all eight distributions sum to 100%.')
