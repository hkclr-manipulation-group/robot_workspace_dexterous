"""Extract independent left 7-DOF chains with traceable native/normalised meshes."""
from pathlib import Path
import xml.etree.ElementTree as E
import copy,json,hashlib,os
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2];WS=ROOT/'robot_workspace_dexterous';OUT=WS/'workspace_results/nine_model_comparison/models'
template=WS/'workspace_results/six_model_comparison/models/spark2_v2_2_open';tool=E.parse(template/'robot.urdf').getroot();tm=yaml.safe_load((template/'collision_meshes.yaml').read_text())['links']
for version in ['2_1','2_2']:
 name=f'dual_v{version}_left';source=WS/'models'/f'dual_v{version}_no_gripper';full=E.parse(source/'robot.urdf').getroot();manifest=yaml.safe_load((source/'collision_meshes.yaml').read_text());root='arm_base_link_l';links={root};joints=[]
 while True:
  added=[j for j in full.findall('joint') if j.find('parent').get('link') in links and j.find('child').get('link') not in links]
  if not added:break
  joints+=added;links.update(j.find('child').get('link') for j in added)
 assert all(l.endswith('_l') for l in links);assert len([j for j in joints if j.get('type')!='fixed'])==7
 length=sum(np.linalg.norm(np.fromstring(j.find('origin').get('xyz','0 0 0'),sep=' ')) for j in joints if j.find('origin') is not None)
 for state in ['bare','open']:
  for native in [True,False]:
   dest=(WS/'workspace_results/dual_single_native/models' if native else OUT)/f'{name}_{state}';dest.mkdir(parents=True,exist_ok=True);scale=1 if native else 1/length;r=E.Element('robot',name=name);assets={}
   for old in full.findall('link'):
    n=old.get('name')
    if n not in links:continue
    l=E.SubElement(r,'link',name=n)
    for e in manifest['links'].get(n,[]):
     file=(source/e['file']).resolve();assert hashlib.sha256(file.read_bytes()).hexdigest()==e['sha256'];assets[str(file.relative_to(ROOT))]=e['sha256']
     for tag in ['visual','collision']:
      v=E.SubElement(l,tag);E.SubElement(v,'origin',xyz=' '.join(map(str,np.array(e['xyz'])*scale)),rpy=' '.join(map(str,e['rpy'])));g=E.SubElement(v,'geometry');E.SubElement(g,'mesh',filename=os.path.relpath(file,dest),scale=' '.join(map(str,np.array(e['scale'])*scale)))
   for old in joints:
    j=copy.deepcopy(old);o=j.find('origin')
    if o is not None:o.set('xyz',' '.join(map(str,np.fromstring(o.get('xyz','0 0 0'),sep=' ')*scale)))
    r.append(j)
   ee='arm_end_effector_l'
   if state=='open':
    for old in tool:
     if old.get('name','').startswith(('gripper_','grasp_tcp','mount_gripper')):
      e=copy.deepcopy(old)
      if e.get('name')=='mount_gripper':e.find('parent').set('link',ee);e.find('origin').set('xyz','0 0 0.001')
      if e.tag=='link':
       for entry in tm.get(e.get('name'),[]):
        file=ROOT/entry['file'].split('/rt_ws/')[-1];assert file.exists()
        for tag in ['visual','collision']:
         v=E.SubElement(e,tag);E.SubElement(v,'origin',xyz=' '.join(map(str,entry['xyz'])),rpy=' '.join(map(str,entry['rpy'])));g=E.SubElement(v,'geometry');E.SubElement(g,'mesh',filename=os.path.relpath(file,dest),scale=' '.join(map(str,entry['scale'])))
      r.append(e)
    ee='grasp_tcp'
   E.indent(r);E.ElementTree(r).write(dest/'robot.urdf',encoding='utf-8',xml_declaration=True)
   meta=dict(model=name,state=state,base=root,ee=ee,locks={},fingers=['gripper_L1','gripper_L2'] if state=='open' else [],scale=scale,original_chain_length_m=length,normalized_chain_length_m=length if native else 1.,radius_upper_bound_m=(length if native else 1.)+(.2 if state=='open' else 0),source_revision='local URDF hash below',source_urdf_sha256=hashlib.sha256((source/'robot.urdf').read_bytes()).hexdigest(),urdf_sha256=hashlib.sha256((dest/'robot.urdf').read_bytes()).hexdigest(),source_assets=assets,mount_note='Independent left arm; torso/right arm removed; reference gripper +1 mm flange spacer; physical mounting needs review')
   (dest/'model.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(name,'7 DOF',length,'m native chain')
