from pathlib import Path
import json,subprocess,sys,time,os,signal,fcntl,traceback,hashlib,datetime
D=Path(__file__).resolve().parent;S=D.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 f=D/n;t=f.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(f)
def read(cid,kind):
 f=D/cid/'fit'/kind
 return json.loads(f.read_text()) if f.exists() else {}
def existing_alive(pid):
 r=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],capture_output=True,text=True)
 return r.returncode==0 and '/train.py' in r.stdout and not r.stdout.lstrip().startswith('Z')
def kill_old(pid):
 r=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
 if r.returncode==0 and str(D/'run_tournament.py') in r.stdout:os.kill(pid,signal.SIGKILL)
def main():
 lock=(D/'parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((D/'parallel_execution_plan.json').read_text());assert not (D/'parallel_status.json').exists()
 for name,h in plan['code_sha256'].items():assert sha(D/name)==h,name
 for name,h in plan['frozen_evidence_sha256'].items():assert sha(S/name)==h,name
 deadline=plan['deadline_unix'];ids=[c['id'] for c in plan['candidates']];rounds=[];throughput=[]
 for ri,(target,survivors) in enumerate(zip([6,12,24,30],[4,2,1,1])):
  children={};handles=[];pending=[]
  for cid in ids:
   completed=read(cid,'report.json').get('epochs',0)
   if completed==target:continue
   assert completed<target
   owned=next((x for x in plan['inherited_children'] if x['candidate']==cid),None) if ri==0 else None
   if owned:children[cid]=dict(pid=owned['pid'],process=None,started=time.time())
   else:pending.append(cid)
  # Initial stage uses all eight concurrent candidates; later stages use all survivors.
  for cid in pending:
   f=(D/(cid+'_parallel_train.log')).open('ab');handles.append(f)
   proc=subprocess.Popen([sys.executable,'-u',str(D/cid/'train.py'),str(target)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   children[cid]=dict(pid=proc.pid,process=proc,started=time.time())
  save('parallel_status.json',dict(stage='wm19_parallel_training',round=ri+1,target_epoch=target,active_candidates=list(children),concurrency_limit=8))
  sample_at=time.time();sample_updates=sum(read(c,'status.json').get('updates',0) for c in ids)
  while children:
   assert time.time()<deadline,'Registered total fit wall deadline exceeded'
   active=[]
   for cid,item in list(children.items()):
    proc=item['process'];done=(proc.poll() is not None) if proc else (read(cid,'report.json').get('epochs')==target or not existing_alive(item['pid']))
    if done:
     if proc:assert proc.returncode==0,(cid,proc.returncode)
     rep=read(cid,'report.json');assert rep.get('epochs')==target,(cid,rep.get('epochs'),target)
     del children[cid]
    else:
     assert time.time()-item['started']<=7200,(cid,'child timeout')
     st=read(cid,'status.json');active.append(dict(candidate=cid,pid=item['pid'],epoch=st.get('epoch'),updates=st.get('updates')))
   updates=sum(read(c,'status.json').get('updates',0) for c in ids)
   status=dict(stage='wm19_parallel_tournament_training',round=ri+1,target_epoch=target,active_candidates=active,concurrency_limit=8,current_scope_updates=updates,completed_candidate_epochs=sum(read(c['id'],'report.json').get('epochs',0) for c in plan['candidates']),hardware_enabled=False)
   save('status.json',status);save('parallel_status.json',status)
   if time.time()-sample_at>=60:
    dt=time.time()-sample_at;throughput.append(dict(round=ri+1,seconds=dt,active_candidates=len(active),aggregate_optimizer_updates=updates-sample_updates,aggregate_updates_per_second=(updates-sample_updates)/dt,comparable_serial_baseline_not_measured=True));save('parallel_throughput.json',dict(samples=throughput,memory_bytes=plan['memory_bytes'],eight_processes_not_guaranteed_fastest=True));sample_at=time.time();sample_updates=updates
   time.sleep(1)
  for f in handles:f.close()
  if ri==0:kill_old(plan['old_sequential_parent_pid'])
  board=[]
  for cid in ids:
   r=read(cid,'report.json');board.append(dict(candidate=cid,cumulative_epochs=target,last_epoch_key=r['history'][-1]['selection_key'],best_epoch=r['best_epoch'],best_key=min(x['selection_key'] for x in r['history'])))
  board.sort(key=lambda x:(x['last_epoch_key'],x['candidate']));ids=[x['candidate'] for x in board[:survivors]]
  rounds.append(dict(round=ri+1,target_epoch=target,leaderboard=board,survivors=ids));epochs=sum(read(c['id'],'report.json').get('epochs',0) for c in plan['candidates']);assert epochs<=102
  save('leaderboard.json',dict(rounds=rounds,total_candidate_epochs=epochs,total_updates=epochs*358))
 winner=ids[0];rep=read(winner,'report.json');epochs=sum(read(c['id'],'report.json')['epochs'] for c in plan['candidates']);assert epochs==102
 save('winner.json',dict(candidate=winner,weight_sha256=sha(D/winner/'fit/best.pt'),best_epoch=rep['best_epoch'],best_checkpoint_updates=rep['best_checkpoint_updates'],total_candidate_epochs=epochs,total_updates=epochs*358,independent_used_for_selection=False,threshold=.5))
 save('status.json',dict(stage='wm19_tournament_winner_frozen_pending_scoring',winner=winner,total_candidate_epochs=102,total_updates=36516,scoring_started=False,hardware_enabled=False));save('parallel_status.json',dict(stage='parallel_tournament_complete',active_candidates=[]))
if __name__=='__main__':
 try:main()
 except Exception:
  try:
   plan=json.loads((D/'parallel_execution_plan.json').read_text())
   lines=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
   for line in lines:
    if str(D) in line and '/train.py ' in line:
     pid=int(line.split()[0])
     try:os.killpg(pid,signal.SIGTERM)
     except ProcessLookupError:pass
   kill_old(plan['old_sequential_parent_pid'])
  finally:save('status.json',dict(stage='wm19_parallel_tournament_failed',error=traceback.format_exc(),automatic_second_tournament=False,hardware_enabled=False))
  raise
