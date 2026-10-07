from pathlib import Path
import subprocess,sys,os,signal,json,fcntl,traceback
ROOT=Path(__file__).resolve().parent
def save(d):
 p=ROOT/'runner_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (ROOT/'runner_status.json').exists(),'No duplicate launch'
 plan=json.loads((ROOT/'execution_plan.json').read_text())
 save(dict(stage='wm19_independent_coverage_running',optimizer_updates=0,hardware_enabled=False))
 with (ROOT/'sequence.log').open('ab') as log:
  child=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=child.wait(timeout=plan['maximum_wall_seconds'])
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
  assert rc==0,rc
 save(dict(stage='wm19_independent_coverage_complete',optimizer_updates=0,hardware_enabled=False))
if __name__=='__main__':
 try:main()
 except Exception:save(dict(stage='wm19_independent_coverage_failed',error=traceback.format_exc(),optimizer_updates=0,hardware_enabled=False));raise
