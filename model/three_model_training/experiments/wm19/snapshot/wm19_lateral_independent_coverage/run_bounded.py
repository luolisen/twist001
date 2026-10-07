from pathlib import Path
import subprocess,sys,os,signal,json,fcntl,time,traceback
ROOT=Path(__file__).resolve().parent
def save(v):
 p=ROOT/'runner_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'runner_status.json').exists()
 plan=json.loads((ROOT/'execution_plan.json').read_text());deadline=time.monotonic()+plan['maximum_wall_seconds']
 for script in ['fresh_vla.py','audit.py']:
  save(dict(stage='wm16_lateral_independent_data_'+script.removesuffix('.py')+'_running',training_started=False,WM_model_calls=0,optimizer_updates=0,hardware_enabled=False))
  with (ROOT/(script+'.log')).open('ab') as log:
   child=subprocess.Popen([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=child.wait(timeout=max(.1,deadline-time.monotonic()))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
   assert rc==0,(script,rc)
 gate=json.loads((ROOT/'quality_gate.json').read_text());save(dict(stage='wm16_lateral_independent_settled_vla_data_complete',data_gate_passed=gate['passed'],training_started=False,WM_model_calls=0,optimizer_updates=0,hardware_enabled=False))
if __name__=='__main__':
 try:main()
 except Exception:save(dict(stage='wm16_lateral_independent_settled_vla_data_failed',error=traceback.format_exc(),training_started=False,WM_model_calls=0,optimizer_updates=0,hardware_enabled=False));raise
