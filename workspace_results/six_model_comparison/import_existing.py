from pathlib import Path
import shutil,json,hashlib
out=Path(__file__).parent.resolve();old=out.parent
shutil.copy2(old/'gripper_demo_up/payload.stl',out/'payload.stl')
for i in range(1,5):
 source=old/('gripper_demo_up' if i==3 else 'gripper_demo')
 for state in ['open','holding']:
  name=f'P2AB-{i}_{state}'
  shutil.copytree(source/'models'/name,out/'models'/name,dirs_exist_ok=True)
  target=out/'output'/name;target.mkdir(parents=True,exist_ok=True)
  for p in (source/'output'/name).iterdir():
   if p.is_file():shutil.copy2(p,target/p.name)
  config=out/'configs'/f'{name}.yaml';config.write_text((source/'configs'/config.name).read_text().replace(str(source),str(out)))
  for p in list((out/'models'/name).rglob('*'))+list(target.iterdir()):
   if p.is_file() and p.suffix in ('.yaml','.json','.urdf'):
    s=p.read_text();p.write_text(s.replace(str(source),str(out)))
  summary=target/'grasp_tcp_summary.json';d=json.loads(summary.read_text());c=d['collision_model'];c['original_manifest_sha256']=c['manifest_sha256'];c['manifest_sha256']=hashlib.sha256(Path(c['file']).read_bytes()).hexdigest();d['reused_from']=str(source/'output'/name);summary.write_text(json.dumps(d,indent=2))
print('Imported 8 unchanged gripper cases into consolidated comparison folder.')
