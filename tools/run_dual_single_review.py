"""Run independent dual-left-arm benchmarks without touching completed nine-model data."""
from pathlib import Path
import os,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2];TOOLS=Path(__file__).resolve().parent;names=['dual_v2_1_left','dual_v2_2_left'];env=dict(os.environ,SPARK_EXTRA_MODELS=json.dumps(names),SPARK_DIRECTION_MODELS=json.dumps(names),SPARK_DIRECTION_WORKERS='3',SPARK_BENCHMARK_JIT='1')
jobs=[]
for state in ['bare','open']:
 jobs.append(subprocess.Popen([sys.executable,str(TOOLS/'six_direction_grasp.py')],cwd=ROOT,env=dict(env,SPARK_DIRECTION_STATE=state)))
for name in names:jobs.append(subprocess.Popen([sys.executable,str(TOOLS/'compare_nine_workspace.py'),'--model',name,'--states','bare','open','--workers','2'],cwd=ROOT,env=env))
for p in jobs:
 if p.wait():raise SystemExit('A benchmark failed; summaries were not published')
for state in ['bare','open']:subprocess.run([sys.executable,str(TOOLS/'validate_six_direction_grasp.py')],cwd=ROOT,env=dict(env,SPARK_DIRECTION_STATE=state),check=True)
subprocess.run([sys.executable,str(ROOT/'spark2_design_docs/review/dual_single/publish.py')],cwd=ROOT,env=env,check=True)
