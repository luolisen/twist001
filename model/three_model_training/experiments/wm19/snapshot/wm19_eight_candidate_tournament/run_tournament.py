from pathlib import Path
import json,subprocess,sys,time,os,signal,fcntl,traceback,hashlib,shutil
D=Path(__file__).resolve().parent;S=D.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 f=D/n;t=f.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(f)
def run(cmd,log,cap):
 with (D/log).open('ab') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:r=p.wait(timeout=cap)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  assert r==0,(cmd,r)
def main():
 lock=(D/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (D/'status.json').exists()
 plan=json.loads((D/'execution_plan.json').read_text());deadline=time.monotonic()+plan['fit_maximum_wall_seconds']
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 for n,h in plan['code_sha256'].items():assert sha(D/n)==h,n
 ids=[c['id'] for c in plan['candidates']]
 for cid in ids:
  save('status.json',dict(stage='wm19_all_candidate_zero_update_preflight',candidate=cid,optimizer_updates=0))
  run([sys.executable,'-u',str(D/cid/'train.py'),'0','preflight'],cid+'_preflight.log',min(1200,max(.1,deadline-time.monotonic())))
  assert json.loads((D/cid/'preflight_complete.json').read_text())['passed']
 total_epochs=0;total_updates=0;rounds=[]
 for ri,(target,survivors) in enumerate(zip([6,12,24,30],[4,2,1,1])):
  leaderboard=[]
  for cid in ids:
   before=json.loads((D/cid/'fit/report.json').read_text())['epochs'] if (D/cid/'fit/report.json').exists() else 0
   save('status.json',dict(stage='wm19_tournament_training',round=ri+1,candidate=cid,target_epoch=target,total_candidate_epochs=total_epochs,total_updates=total_updates))
   run([sys.executable,'-u',str(D/cid/'train.py'),str(target)],cid+'_train.log',min(7200,max(.1,deadline-time.monotonic())))
   report=json.loads((D/cid/'fit/report.json').read_text());assert report['epochs']==target
   total_epochs+=target-before;total_updates+=(target-before)*358;assert total_epochs<=102 and total_updates<=36516
   leaderboard.append(dict(candidate=cid,cumulative_epochs=target,last_epoch_key=report['history'][-1]['selection_key'],best_epoch=report['best_epoch'],best_key=min(x['selection_key'] for x in report['history'])))
  leaderboard.sort(key=lambda x:(x['last_epoch_key'],x['candidate']));ids=[x['candidate'] for x in leaderboard[:survivors]]
  rounds.append(dict(round=ri+1,target_epoch=target,leaderboard=leaderboard,survivors=ids));save('leaderboard.json',dict(rounds=rounds,total_candidate_epochs=total_epochs,total_updates=total_updates))
 assert total_epochs==102 and total_updates==36516 and len(ids)==1
 winner=ids[0];report=json.loads((D/winner/'fit/report.json').read_text())
 save('winner.json',dict(candidate=winner,weight_sha256=sha(D/winner/'fit/best.pt'),best_epoch=report['best_epoch'],best_checkpoint_updates=report['best_checkpoint_updates'],total_candidate_epochs=total_epochs,total_updates=total_updates,independent_used_for_selection=False,threshold=.5))
 save('status.json',dict(stage='wm19_tournament_winner_frozen_pending_scoring',winner=winner,total_candidate_epochs=total_epochs,total_updates=total_updates,scoring_started=False,hardware_enabled=False))
 # Separate frozen-winner scoring is registered after winner/checkpoint integrity review; no test-driven retraining.
if __name__=='__main__':
 try:main()
 except Exception:save('status.json',dict(stage='wm19_tournament_failed',error=traceback.format_exc(),automatic_second_tournament=False,hardware_enabled=False));raise
