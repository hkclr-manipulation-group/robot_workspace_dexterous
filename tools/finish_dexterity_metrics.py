"""Finish the running finite motion suite, audit SO(3), validate and publish."""
from pathlib import Path
import os,subprocess,sys,time,json
ROOT=Path(__file__).resolve().parents[2];TOOLS=Path(__file__).resolve().parent
MOTION=ROOT/'robot_workspace_dexterous/workspace_results/motion_dexterity_v1'
# The two motion jobs have already been launched; do not start duplicates.
while not all((MOTION/s/'complete.json').exists() for s in ['bare','open']):time.sleep(10)
for script in ['orientation_budget_audit.py','validate_motion_dexterity.py']:
 jobs=[]
 for state in ['bare','open']:
  env=dict(os.environ,SPARK_DIRECTION_STATE=state,SPARK_MOTION_WORKERS='4');env.pop('SPARK_MOTION_MODELS',None)
  log=open('/tmp/spark-'+script.replace('.py','')+'-'+state+'.log','w')
  jobs.append((subprocess.Popen([sys.executable,str(TOOLS/script)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log))
 for p,log in jobs:
  if p.wait():raise SystemExit(script+' failed; see /tmp/spark-* logs')
  log.close()
subprocess.run([sys.executable,str(ROOT/'spark2_design_docs/review/motion_metrics/publish.py')],check=True,cwd=ROOT)
