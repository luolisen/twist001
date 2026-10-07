from pathlib import Path
import json,subprocess,sys,time,os,signal,fcntl,traceback,hashlib
D=Path(__file__).resolve().parent;T=D.parent/'wm19_eight_candidate_tournament'
def save(path,x):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(path)
def run(name,cap,stage):
 with (D/(name+'.log')).open('ab') as f:
  p=subprocess.Popen([sys.executable,'-u',str(D/name)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+cap
  while p.poll() is None:
   progress={}
   pp=D/('score_status.json' if name=='score.py' else 'status.json')
   if pp.exists():
    try:progress=json.loads(pp.read_text())
    except Exception:pass
   save(D/'runner_status.json',dict(stage=stage,child_pid=p.pid,progress=progress,maximum_seconds=cap))
   save(T/'status.json',dict(stage=stage,winner='c05',candidate='c05',total_candidate_epochs=102,total_updates=36516,postfit_progress=progress,hardware_enabled=False))
   if time.monotonic()>deadline:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise TimeoutError(name)
   time.sleep(5)
  assert p.returncode==0,(name,p.returncode)
def main():
 lock=(D/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run('verify.py',2400,'wm19_winner_verification');assert json.loads((D/'verification_report.json').read_text())['passed']
 run('score.py',1800,'wm19_winner_independent_scoring')
 gate=json.loads((D/'quality_gate.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 save(D/'completion_integrity_check.json',dict(verification_passed=True,scored_candidates=504,weight_sha256=gate['weight_sha256'],verification_report_sha256=sha(D/'verification_report.json'),score_report_sha256=sha(D/'score_report.json'),quality_gate_sha256=sha(D/'quality_gate.json'),physical_reliability_subgate_passed=gate['physical_reliability_subgate_passed'],optimizer_updates=0,hardware_enabled=False))
 save(D/'status.json',dict(stage='wm19_winner_verification_and_scoring_complete',candidates_done=504,physical_reliability_subgate_passed=gate['physical_reliability_subgate_passed'],hardware_enabled=False))
 save(T/'status.json',dict(stage='wm19_winner_independent_scoring_complete',winner='c05',total_candidate_epochs=102,total_updates=36516,physical_reliability_subgate_passed=gate['physical_reliability_subgate_passed'],hardware_enabled=False))
if __name__=='__main__':
 try:main()
 except Exception:
  save(D/'status.json',dict(stage='wm19_postfit_failed',error=traceback.format_exc(),automatic_second_tournament=False,hardware_enabled=False));save(T/'status.json',dict(stage='wm19_postfit_failed',hardware_enabled=False));raise
