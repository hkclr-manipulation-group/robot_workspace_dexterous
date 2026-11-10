from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent/'output'
out=Path(__file__).parent
r=[];d=[]
for i in range(1,5):
 ee='L'+'6'*i
 with np.load(root/f'P2AB-{i}_preview'/f'{ee}.npz') as data:
  counts=data['reachable_orientations'];n=int(data['orientation_count'])
  r.append(int((counts>0).sum()));d.append(int((counts==n).sum()))
fig,ax=plt.subplots(figsize=(10,5.7),layout='constrained')
x=np.arange(4)
a=ax.bar(x-.19,r,.38,label='Reachable: at least 1 / 16 orientations',color='#3986c6')
b=ax.bar(x+.19,d,.38,label='Dexterous: all 16 / 16 orientations',color='#e58b35')
ax.bar_label(a,padding=5,fontsize=13);ax.bar_label(b,padding=5,fontsize=13)
ax.set_xticks(x,[f'P2AB-{i}' for i in range(1,5)])
ax.set(ylabel='Total grid-point count',ylim=(0,3000),title='Length-normalized workspace: joint-offset chain length = 1 m')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right',fontsize=9)
fig.supxlabel('100 mm XYZ grid | 16 sampled orientations | Uniformly scaled geometry and kinematics | Full 3D grid',fontsize=10)
fig.savefig(out/'p2ab_normalized_counts.png',dpi=180);plt.close(fig)
