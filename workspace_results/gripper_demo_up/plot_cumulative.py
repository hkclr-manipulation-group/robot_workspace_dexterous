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
thresholds=np.arange(16,0,-1)
for i,ax in enumerate(axes.flat,1):
 totals=[]
 for j,state in enumerate(['open','holding']):
  bins={int(r['reachable_directions']):int(r['points']) for r in rows if r['model']==f'P2AB-{i}' and r['state']==state}
  total=sum(bins[k] for k in range(1,17));totals.append(total)
  counts=np.array([sum(bins[k] for k in range(t,17)) for t in thresholds]);pct=counts/total*100
  assert np.all(np.diff(pct)>=0) and np.isclose(pct[-1],100)
  bars=ax.bar(np.arange(16)+(j-.5)*.4,pct,.4,color=['#3182bd','#f28e2b'][j],label='Open gripper' if j==0 else 'Holding 30 mm cube')
  ax.bar_label(bars,labels=[f'{v:.2f}%' for v in pct],rotation=90,padding=3,fontsize=8)
  for t,n,v in zip(thresholds,counts,pct):export.append(dict(model=f'P2AB-{i}',state=state,minimum_reachable_directions=int(t),minimum_coverage_percent=float(t/16*100),cumulative_points=int(n),total_reachable_points=total,cumulative_percent=float(v)))
 ax.set_title(f'P2AB-{i} | reachable points: {totals[0]:,} / {totals[1]:,} (open / holding)',fontsize=12)
 ax.set_xticks(np.arange(16),[f'≥{t}/16' for t in thresholds],rotation=55,ha='right');ax.set_ylim(0,118)
 ax.set_xlabel('Minimum orientation coverage');ax.set_ylabel('Cumulative share of reachable points (%)');ax.yaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
axes.flat[0].legend(loc='upper left')
fig.suptitle('Cumulative coverage: ≥16/16 → ≥1/16 | unreachable points excluded\nEach bar = points meeting threshold / all reachable points | 1 m arms + Spark2 gripper | P2AB-3 mount revised',fontsize=16)
fig.savefig(out/'coverage_cumulative_percent.png',dpi=170);fig.savefig(out/'coverage_cumulative_percent.pdf')
with (out/'coverage_cumulative_percent.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=export[0]);w.writeheader();w.writerows(export)
print('Validated eight monotonic cumulative distributions; ≥1/16 = 100%.')
