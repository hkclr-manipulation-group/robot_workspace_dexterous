from pathlib import Path
import copy,hashlib,json,xml.etree.ElementTree as E
import numpy as np,trimesh,yaml
from robot_workspace_dexterous.urdf_compat import origin_matrix
out=Path(__file__).parent;old=out.parent/'normalized_length';src=Path('/home/hkclr-manipulation-spark1/rt_ws/cuarm_mesh/spark2_v2_2');gr=E.parse(src/'assembly/gripper/spark2_stl.urdf').getroot()
(out/'configs').mkdir(exist_ok=True);cube=trimesh.creation.box(extents=[.03]*3);cube.export(out/'payload.stl')
for i in range(1,5):
 name=f'P2AB-{i}';last='L'+'6'*i
 for state,q in [('open',0.),('holding',.032697)]:
  dest=out/'models'/f'{name}_{state}';dest.mkdir(parents=True,exist_ok=True)
  tree=E.parse(old/'models'/name/'robot.urdf');root=tree.getroot();manifest=yaml.safe_load((old/'models'/name/'collision_meshes.yaml').read_text())
  e=manifest['links'][last][0];mesh=trimesh.load(e['file'],force='mesh');mount=float(mesh.bounds[1,2]*e['scale'][2])+.001
  if i==3: mount=float(mesh.bounds[0,2]*e['scale'][2])-.001
  ignores={}
  for joint in root.findall('joint'):ignores.setdefault(joint.find('parent').get('link'),[]).append(joint.find('child').get('link'))
  for lname,file in [('gripper_base_link','base_link.stl'),('gripper_L1','L1.stl'),('gripper_L2','L2.stl')]:
   root.append(E.Element('link',name=lname));path=src/'meshes/gripper'/file
   manifest['links'][lname]=[dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=[1.,1.,1.],xyz=[0.,0.,0.],rpy=[0.,0.,0.])]
  joint=E.SubElement(root,'joint',name='mount_gripper',type='fixed');E.SubElement(joint,'parent',link=last);E.SubElement(joint,'child',link='gripper_base_link');E.SubElement(joint,'origin',xyz=f'0 0 {mount}',rpy=('3.141592653589793 0 0' if i==3 else '0 0 0'))
  ignores[last]=['gripper_base_link']
  ignores['gripper_base_link']=['gripper_L1','gripper_L2']
  for jname in ['gripper_J1','gripper_J1_1']:
   j=copy.deepcopy(gr.find(f"joint[@name='{jname}']"));t=origin_matrix(j.find('origin'));xyz=t[:3,3]+t[:3,2]*q;j.find('origin').set('xyz',' '.join(map(str,xyz)));j.set('type','fixed')
   for tag in ['axis','limit','mimic']:
    el=j.find(tag)
    if el is not None:j.remove(el)
   root.append(j)
  root.append(E.Element('link',name='grasp_tcp'));j=E.SubElement(root,'joint',name='grasp_tcp_joint',type='fixed');E.SubElement(j,'parent',link='gripper_base_link');E.SubElement(j,'child',link='grasp_tcp');E.SubElement(j,'origin',xyz='0 0 0.155',rpy='0 0 0')
  if state=='holding':
   path=out/'payload.stl';manifest['links']['grasp_tcp']=[dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=[1.,1.,1.],xyz=[0.,0.,0.],rpy=[0.,0.,0.])]
   ignores['grasp_tcp']=['gripper_L1','gripper_L2']
  tree.write(dest/'robot.urdf',encoding='utf-8',xml_declaration=True);manifest['urdf_sha256']=hashlib.sha256((dest/'robot.urdf').read_bytes()).hexdigest();(dest/'collision_meshes.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
  cfg=yaml.safe_load((old/'configs'/f'{name}.yaml').read_text());cfg['robot'].update(urdf=str(dest/'robot.urdf'),collision_meshes=str(dest/'collision_meshes.yaml'),ee_links=['grasp_tcp'],self_collision_ignore=ignores)
  cfg['solver']['backend']='cuda';cfg['grid'].update(x_range=[-1.2,1.2],y_range=[-1.2,1.2],z_min=-1.2,z_max=1.2)
  (out/'configs'/f'{name}_{state}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
  print(name,state,'mount',mount)
