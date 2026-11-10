from pathlib import Path
import copy,hashlib,json,xml.etree.ElementTree as E
import numpy as np,yaml,trimesh
out=Path(__file__).parent;repo=Path('/home/hkclr-manipulation-spark1/rt_ws/robot_workspace_dexterous');old=out.parent
names=['spark2_v2_2','spark2_v1']+[f'P2AB-{i}' for i in range(1,5)]
(out/'configs').mkdir(exist_ok=True);records=[]
for name in names:
 source=repo/'models'/name;tree=E.parse(source/'robot.urdf');root=tree.getroot()
 length=sum(np.linalg.norm(np.fromstring(j.find('origin').get('xyz','0 0 0'),sep=' ')) for j in root.findall('joint') if j.find('origin') is not None);scale=float(1/length)
 for o in root.findall('.//origin'):o.set('xyz',' '.join(map(str,np.fromstring(o.get('xyz','0 0 0'),sep=' ')*scale)))
 manifest=yaml.safe_load((source/'collision_meshes.yaml').read_text())
 for entries in manifest['links'].values():
  for e in entries:e.update(file=str((source/e['file']).resolve()),scale=[v*scale for v in e['scale']],xyz=[v*scale for v in e['xyz']])
 states=['bare','open','holding'] if name.startswith('spark2_') else ['bare']
 for state in states:
  r=copy.deepcopy(root);m=copy.deepcopy(manifest);last='arm_L6' if name.startswith('spark2_') else 'L'+'6'*int(name[-1]);base='dummy_base_link' if name.startswith('spark2_') else 'base_link';ee='arm_end_effector' if name.startswith('spark2_') else last
  ignore={}
  for j in r.findall('joint'):ignore.setdefault(j.find('parent').get('link'),[]).append(j.find('child').get('link'))
  if state!='bare':
   template=old/'gripper_demo_up/models'/f'P2AB-1_{state}';g=E.parse(template/'robot.urdf').getroot();gm=yaml.safe_load((template/'collision_meshes.yaml').read_text())
   entry=m['links'][last][0];mesh=trimesh.load(entry['file'],force='mesh');mount=float(mesh.bounds[1,2]*entry['scale'][2])+.001
   for elem in g:
    if elem.get('name','').startswith(('gripper_','grasp_tcp','mount_gripper')):
     item=copy.deepcopy(elem)
     if item.get('name')=='mount_gripper':item.find('parent').set('link',last);item.find('origin').set('xyz',f'0 0 {mount}')
     r.append(item)
   for key,entries in gm['links'].items():
    if key.startswith(('gripper_','grasp_tcp')):m['links'][key]=entries
   ignore.setdefault(last,[]).append('gripper_base_link');ignore['gripper_base_link']=['gripper_L1','gripper_L2']
   if state=='holding':ignore['grasp_tcp']=['gripper_L1','gripper_L2']
   ee='grasp_tcp'
  dest=out/'models'/f'{name}_{state}';dest.mkdir(parents=True,exist_ok=True);E.ElementTree(r).write(dest/'robot.urdf',encoding='utf-8',xml_declaration=True)
  m['urdf_sha256']=hashlib.sha256((dest/'robot.urdf').read_bytes()).hexdigest();(dest/'collision_meshes.yaml').write_text(yaml.safe_dump(m,sort_keys=False))
  cfg=yaml.safe_load((old/'gripper_demo_up/configs/P2AB-1_open.yaml').read_text());cfg['robot'].update(urdf=str((dest/'robot.urdf').resolve()),collision_meshes=str((dest/'collision_meshes.yaml').resolve()),base_link=base,ee_links=[ee],self_collision_ignore=ignore)
  (out/'configs'/f'{name}_{state}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 records.append(dict(model=name,original_chain_length_m=float(length),scale=float(scale),normalized_chain_length_m=1.,joint_limits='original URDF limits retained'))
(out/'normalization.json').write_text(json.dumps(records,indent=2));print(records)
