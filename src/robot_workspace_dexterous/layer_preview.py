"""Self-contained XY layer viewer; works when progress HTML is opened locally."""
import json

import numpy as np


def xy_layers(workspace, tested, center_z=0.):
    points = workspace.positions
    heights, inverse = np.unique(points[:, 2], return_inverse=True)
    layers = []
    for index, z in enumerate(heights):
        mask = inverse == index
        visible = mask & (tested > 0) & (workspace.reachable_orientations > 0)
        layers.append(dict(z=float(z), total=int(mask.sum()),
                           tested=int(np.count_nonzero(mask & (tested > 0))),
                           complete=int(np.count_nonzero(mask & (tested == workspace.orientation_count))),
                           points=np.column_stack((points[visible, :2], workspace.dexterity[visible])).tolist()))
    data = dict(layers=layers, center=int(np.argmin(abs(heights-center_z))),
                bounds=[points[:, :2].min(0).tolist(), points[:, :2].max(0).tolist()])
    return _HTML.replace('__DATA__', json.dumps(data, allow_nan=False))


_HTML = '''<section style="background:white;padding:16px;border-radius:8px">
<h2>Live XY layers</h2>
<label>Z layer <select id="xy-layer"></select></label>
<button id="xy-prev">Previous</button><button id="xy-next">Next</button>
<p id="xy-status"></p>
<canvas id="xy-canvas" width="680" height="680" style="max-width:100%;height:auto"></canvas>
<p>Orientation coverage: 0 <span style="display:inline-block;width:180px;height:12px;background:linear-gradient(to right,#440154,#21918c,#fde725)"></span> 1</p>
<p>Only accepted points in this Z layer are shown. Untested cells are unknown;
partial coverage is a lower bound. Final exported images remain centre sections.</p>
</section>
<script id="xy-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('xy-data').textContent);
  const select = document.getElementById('xy-layer');
  data.layers.forEach((layer, i) => select.add(new Option(`Z = ${layer.z.toFixed(4)} m`, i)));
  const saved = new URLSearchParams(location.hash.slice(1)).get('xy_z');
  let initial = data.center;
  if (saved !== null && Number.isFinite(Number(saved))) {
    initial = data.layers.reduce((best, layer, i) =>
      Math.abs(layer.z-Number(saved)) < Math.abs(data.layers[best].z-Number(saved)) ? i : best, 0);
  }
  select.value = initial;
  const canvas = document.getElementById('xy-canvas'), ctx = canvas.getContext('2d');
  function draw() {
    const index = Number(select.value), layer = data.layers[index];
    const hash = new URLSearchParams(location.hash.slice(1));
    hash.set('xy_z', layer.z);
    location.replace('#'+hash.toString());
    document.getElementById('xy-prev').disabled = index === 0;
    document.getElementById('xy-next').disabled = index === data.layers.length-1;
    document.getElementById('xy-status').textContent =
      `${layer.points.length} reachable; ${layer.tested}/${layer.total} tested; ${layer.complete}/${layer.total} fully tested`;
    const [lo, hi] = data.bounds, span = Math.max(hi[0]-lo[0], hi[1]-lo[1], .02)*1.06;
    const cx = (lo[0]+hi[0])/2, cy = (lo[1]+hi[1])/2;
    const px = x => 350+(x-cx)*560/span, py = y => 320-(y-cy)*560/span;
    ctx.clearRect(0,0,680,680); ctx.font = '14px sans-serif'; ctx.fillStyle = '#172b43';
    ctx.fillText(`XY section at Z = ${layer.z.toFixed(4)} m`, 180, 22);
    for (let i=0; i<=4; i++) {
      const x=cx-span/2+i*span/4, y=cy-span/2+i*span/4;
      ctx.strokeStyle='#e0e5ea'; ctx.beginPath();
      ctx.moveTo(px(x),40); ctx.lineTo(px(x),600);
      ctx.moveTo(70,py(y)); ctx.lineTo(630,py(y)); ctx.stroke();
      ctx.fillText(x.toFixed(2),px(x)-16,624); ctx.fillText(y.toFixed(2),12,py(y)+4);
    }
    ctx.fillText('X (m)',320,653); ctx.fillText('Y (m)',8,30);
    const stops=[[68,1,84],[33,145,140],[253,231,37]];
    for (const [x,y,score] of layer.points) {
      const t=Math.max(0,Math.min(1,score))*2, k=Math.min(1,Math.floor(t)), f=t-k;
      const rgb=stops[k].map((v,i)=>Math.round(v*(1-f)+stops[k+1][i]*f));
      ctx.fillStyle=`rgb(${rgb.join(',')})`; ctx.fillRect(px(x)-2,py(y)-2,4,4);
    }
  }
  select.onchange=draw;
  document.getElementById('xy-prev').onclick=()=>{select.selectedIndex--;draw();};
  document.getElementById('xy-next').onclick=()=>{select.selectedIndex++;draw();};
  draw();
})();
</script>'''
