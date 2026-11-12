from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[2];TOOLS=Path(__file__).resolve().parent
# Prioritise the six-direction study before the supplementary 16-pose study.
for orientation_set in ['six','sixteen']:
 jobs=[]
 for state in ['bare','open']:
  env=dict(os.environ,SPARK_DIRECTION_STATE=state,SPARK_FULL_SET=orientation_set,SPARK_FULL_WORKERS='5')
  jobs.append((env,subprocess.Popen([sys.executable,str(TOOLS/'full_rotation_workspace.py')],cwd=ROOT,env=env)))
 for env,p in jobs:
  if p.wait():raise SystemExit('Calculation failed; incomplete data not published')
  subprocess.run([sys.executable,str(TOOLS/'validate_full_rotation.py')],cwd=ROOT,env=env,check=True)
subprocess.run([sys.executable,str(TOOLS/'publish_full_rotation.py')],cwd=ROOT,check=True)
