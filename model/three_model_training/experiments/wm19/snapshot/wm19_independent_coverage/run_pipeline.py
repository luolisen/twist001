"""One preregistered independent data preparation, never fitting or model selection."""
from pathlib import Path
import sys,json,fcntl,subprocess,time,traceback,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'wm_consistent_v7'))
from physical_labels import sha,atomic,np,CONTACT_TYPES
ROOT=Path(__file__).resolve().parent

def main():
 lock=open(ROOT/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'quality_gate.json').exists():raise RuntimeError('Existing completed independent set; do not overwrite')
 plan=json.load(open(ROOT/'execution_plan.json'));sequence_start=time.monotonic()
 preflight=json.load(open(ROOT.parent/'wm15_local_event_preflight/report.json'));assert preflight['source_parameters_unchanged'] and preflight['optimizer_updates']==0 and preflight['source_cache_hashes_verified']==17672
 for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(ROOT.parent/n)==h,n
 assert json.load(open(ROOT.parent/'wm11_visibility_data/quality_gate.json'))['data_gate_passed']
 assert json.load(open(ROOT.parent/'wm11_visibility_model_preflight/report.json'))['structural_and_supervision_preflight_passed']
 proper=json.load(open(ROOT.parent/'wm12_proper_event_objective_preflight/report.json'))
 assert proper['model_parameters_unchanged'] and proper['optimizer_updates']==0 and proper['samples']==96 and proper['exact_per_update_mass_used']
 protected=json.load(open(ROOT.parent/'wm13_event_protection_preflight/report.json'))
 assert protected['parameters_unchanged'] and protected['optimizer_updates']==0 and protected['actual_production_batches']==4
 assert protected['negative_critical_candidate_batches']==0 and protected['negative_retention_candidate_batches']==0
 assert all(x['max_output_difference']==0 for x in protected['mask_checks'])
 for script,stage,cap in [('new_fixtures/generate.py','wm19_independent_physical_initial',600),('synchronize_tests.py','wm19_independent_physical_history',600),('extra_fixtures.py','wm19_independent_self_retention',900),('fresh_vla.py','wm19_independent_actual_VLA',1500)]:
  atomic(ROOT/'pipeline_status.json',dict(stage=stage,pid=os.getpid(),compute_host='M2 Max',started_at=time.time(),training_started=False,hardware_enabled=False))
  with (ROOT/(Path(script).stem+'.log')).open('ab') as log:
   child=subprocess.Popen([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:result=child.wait(timeout=max(1,min(cap,3600-(time.monotonic()-sequence_start))))
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
   assert result==0,(script,result)
 manifests={key:json.load(open(ROOT/file)) for key,file in [('physical','synchronized_test_manifest.json'),('extra','extra_manifest.json'),('vla','vla_manifest.json')]}
 audit=0;counts={k:dict(positive=0,negative=0,new_positive=0,new_negative=0) for k in CONTACT_TYPES};records=[]
 for family,manifest in manifests.items():
  for r in manifest['records']:
   assert sha(ROOT/r['cache'])==r['sha256'],r['cache']
   if r.get('private_audit_cache'):assert sha(ROOT/r['private_audit_cache'])==r['private_audit_sha256']
   with np.load(ROOT/r['cache']) as a:
    assert a['history_images'].shape==(3,2,128,128,3) and a['history_state'].shape==(3,21) and a['actions'].shape==(8,7)
    assert a['history_valid'].shape==(3,) and a['history_valid'][-1]==1
    assert np.isin(a['history_valid'],[0,1]).all() and np.isfinite(a['history_state']).all() and np.isfinite(a['actions']).all()
    assert np.allclose(a['history_relative_times_s'],[-.08,-.04,0])
    for k,name in enumerate(CONTACT_TYPES):
     c=counts[name];yes=bool(a['target_contacts'][k]);new=bool(a['target_new_contacts'][k]);c['positive']+=int(yes);c['negative']+=int(not yes);c['new_positive']+=int(new);c['new_negative']+=int(not new)
   records.append(dict(r,phase_family=family,split='independent_test',fitting_allowed=False))
   audit+=1;atomic(ROOT/'audit_status.json',dict(stage='wm19_independent_cache_audit',samples_done=audit,samples_total=312,compute_host='M2 Max',training_started=False))
 atomic(ROOT/'audit_status.json',dict(stage='wm19_independent_cache_audit_complete',samples_done=audit,samples_total=312,compute_host='M2 Max',training_started=False))
 phys=manifests['physical'];extra=manifests['extra'];vla=manifests['vla']
 # Never score a model or alter thresholds here; coverage and integrity only.
 missing=[]
 for k in ('object_nonfinger','object_object','object_tray_wall','robot_tray','robot_floor'):
  if phys['coverage'][k]['positives']<3 or phys['coverage'][k]['negatives']<3:missing.append(k)
 for k in ('robot_tray','robot_floor'):
  rs=phys['records'];new_count=0
  for r in rs:
   with np.load(ROOT/r['cache']) as a:new_count+=int(a['target_new_contacts'][CONTACT_TYPES.index(k)])
  if new_count<3 or len(rs)-new_count<3:missing.append(k+'_new')
 if not extra['coverage_gate_passed']:missing.append('robot_self_added_or_start_grasp_retention')
 if not vla['generation_gate_passed']:missing.append('fresh_VLA_generation')
 initial=json.load(open(ROOT/'new_fixtures/coverage_manifest.json'))
 for record in initial['records']:assert sha(ROOT/'new_fixtures'/record['cache'])==record['sha256']
 observations=json.load(open(ROOT/'blind/manifest.json'))
 for record in observations['records']:assert sha(ROOT/record['cache'])==record['cache_sha256']
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(ROOT.parent/n)==h,n
 for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
 atomic(ROOT/'independent_manifest.json',dict(records=records,physical_coverage=phys['coverage'],self_retention_coverage=extra['counts'],all_category_counts=counts,no_resampling=True,fitting_allowed=False,execution_plan_sha256=sha(ROOT/'execution_plan.json')))
 failures=extra['failures']+phys.get('construction_failures',[])+vla['bootstrap_failures']
 gate=dict(stage='wm19_independent_coverage_complete',samples=audit,physical_samples=len(phys['records']),self_retention_samples=len(extra['records']),VLA_observations=vla['observations'],VLA_candidates=vla['candidates'],coverage_gate_passed=not missing and not failures and audit==312,missing=missing,failures=failures,physical_coverage=phys['coverage'],self_retention_coverage=extra['counts'],all_category_counts=counts,cache_hashes_verified=True,frozen_dependencies_unchanged=True,code_sha256=plan['code_sha256'],manifest_sha256=sha(ROOT/'independent_manifest.json'),training_started=False,model_scored=False,compute_host='M2 Max',hardware_enabled=False,full_task_acceptance_passed=False,scope='New physical coverage and actual candidate ground truth only, not trained accuracy, Jev or full task. Any failed gate blocks fitting.')
 atomic(ROOT/'quality_gate.json',gate);atomic(ROOT/'pipeline_status.json',gate)
if __name__=='__main__':
 try:main()
 except Exception:atomic(ROOT/'pipeline_status.json',dict(stage='wm19_independent_coverage_failed',error=traceback.format_exc(),compute_host='M2 Max',training_started=False,hardware_enabled=False));raise
