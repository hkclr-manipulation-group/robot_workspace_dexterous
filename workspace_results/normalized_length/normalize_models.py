from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np,yaml
root=Path('/home/hkclr-manipulation-spark1/rt_ws/robot_workspace_dexterous')
out=Path('/home/hkclr-manipulation-spark1/rt_ws/robot_workspace_dexterous/workspace_results/normalized_length')
(out/'configs').mkdir(parents=True,exist_ok=True)
records=[]
for i in range(1,5):
 name=f'P2AB-{i}'; source=root/'models'/name; dest=out/'models'/name;dest.mkdir(parents=True,exist_ok=True)
 tree=ET.parse(source/'robot.urdf');r=tree.getroot()
 length=sum(np.linalg.norm(np.fromstring(j.find('origin').get('xyz','0 0 0'),sep=' ')) for j in r.findall('joint') if j.find('origin') is not None)
 scale=float(1/length)
 for origin in r.findall('.//origin'):
  xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')*scale;origin.set('xyz',' '.join(map(str,xyz)))
 tree.write(dest/'robot.urdf',encoding='utf-8',xml_declaration=True)
 manifest=yaml.safe_load((source/'collision_meshes.yaml').read_text());manifest['urdf_sha256']=hashlib.sha256((dest/'robot.urdf').read_bytes()).hexdigest()
 for entries in manifest['links'].values():
  for entry in entries:
   entry['file']=str((source/entry['file']).resolve());entry['scale']=[v*scale for v in entry['scale']];entry['xyz']=[v*scale for v in entry['xyz']]
 (dest/'collision_meshes.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
 cfg=yaml.safe_load((root/'configs'/f'{name}.yaml').read_text())
 cfg['robot']['urdf']=str(dest/'robot.urdf');cfg['robot']['collision_meshes']=str(dest/'collision_meshes.yaml')
 cfg['grid'].update(x_range=[-1.1,1.1],y_range=[-1.1,1.1],z_min=-1.1,z_max=1.1)
 (out/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 records.append(dict(model=name,original_chain_length_m=float(length),scale=float(scale),normalized_chain_length_m=1.0))
(out/'normalization.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
