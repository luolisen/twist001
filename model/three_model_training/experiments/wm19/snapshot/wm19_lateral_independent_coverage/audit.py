"""Audit one finite WM adaptation pack, train/validation only; no WM forward."""
from pathlib import Path
import sys,json,hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parent;S=ROOT.parent
sys.path.insert(0,str(S/'wm_consistent_v7'))
from physical_labels import np,sha,atomic,CONTACT_TYPES
INPUTS=('history_images','history_state','history_valid','actions')
def main():
 plan=json.load(open(ROOT/'execution_plan.json'))
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
 blind=json.load(open(ROOT/'blind/manifest.json'));vla=json.load(open(ROOT/'vla_manifest.json'));rows=vla['records'];origins={r['observation_id']:r for r in blind['records']};errors=[];signatures=defaultdict(set);groups=defaultdict(set);positives={};manifest=[]
 if len(blind['records'])!=48 or len(rows)!=192:errors.append('48 observations/192 caches required')
 if blind['bootstrap_failures']:errors.append('bootstrap failures retained')
 if len(blind['warmup_records'])!=12 or not all(r['passed'] for r in blind['warmup_records']):errors.append('12 settled warmup gates required')
 for i,r in enumerate(rows):
  path=ROOT/r['cache'];anchor=ROOT/origins[r['observation_id']]['cache'];assert sha(path)==r['sha256'] and sha(anchor)==origins[r['observation_id']]['cache_sha256'];split='independent_test';assert r['split']==split
  groups[split].add(r['source_group']);src=origins[r['observation_id']]
  with np.load(path,allow_pickle=False) as a,np.load(anchor,allow_pickle=False) as b:
   assert np.isfinite(b['integration_state']).all() and b['history_segmentation'].shape==(3,2,128,128)
   assert np.allclose(b['history_absolute_times_s']-b['history_absolute_times_s'][-1],[-.08,-.04,0],atol=1e-10,rtol=0)
   assert np.array_equal(a['history_valid'],[1,1,1])
   assert np.array_equal(a['history_state'][-1],b['state']) and np.array_equal(a['history_images'][-1],b['images'])
   h=hashlib.sha256()
   for k in INPUTS:
    value=a[k];assert np.isfinite(value).all()
    if k!='actions':assert value.dtype==b[k].dtype and np.array_equal(value,b[k])
    h.update(k.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
   signatures[split].add(h.hexdigest());assert a['actions'].shape==(8,7)
   for key in ['object','entity']:
    flow=a['target_'+key+'_history_flow'];valid=a['target_'+key+'_history_flow_valid'];assert flow.shape==(2,2,32,32) and valid.shape==(2,32,32) and np.isfinite(flow).all() and ((valid==0)|(valid==1)).all()
    assert np.all(flow*np.repeat((1-valid)[:,None,:,:],2,axis=1)==0)
   y=np.r_[a['target_contacts'],a['target_new_contacts'],a['target_unlocalized_contacts'],a['target_retention']];assert y.shape==(29,) and ((y==0)|(y==1)).all();assert np.all(y[9:18]<=y[:9]) and np.all(y[18:27]<=y[:9])
   assert np.array_equal(a['target_contacts'],r['contacts']) and np.array_equal(a['target_new_contacts'],r['new_contacts']) and np.array_equal(a['target_retention'],r['retention'])
   positives.setdefault(split,[]).append(y.tolist())
   assert a['target_motion_delta'].shape==(10,) and a['target_future_rgb'].shape==(2,128,128,3) and a['target_contact_regions'].shape==(2,9,32,32)
   record=dict(r,stage_annotation=src['stage_annotation'],public_input_sha256=h.hexdigest(),family='settled_vla_'+src['stage_annotation'],target_contacts=y[:9].tolist(),target_new_contacts=y[9:18].tolist(),target_retention=y[27:].tolist(),actual_start_grasp=bool(r['actual_start_grasp']),actual_start_grasp_scope='Original anchor nonpositive-contact geometry: both fingers on same object without floor or tray-bottom support; GT only.',private_audit_cache=src['cache'],private_audit_sha256=src['cache_sha256'],VLA_training_allowed=False)
   manifest.append(record)
  atomic(ROOT/'audit_status.json',dict(stage='wm16_lateral_independent_data_audit',candidates_done=i+1,candidates_total=192,WM_model_calls=0,optimizer_updates=0,hardware_enabled=False))

 assert len(groups['independent_test'])==12
 hold=[r for r in manifest if r['stage_annotation'].startswith('descend') and r['candidate']=='pose_hold']
 near=[r for r in hold if r['stage_annotation']=='descend_near'];far=[r for r in hold if r['stage_annotation']=='descend_far']
 pos={r['source_group'] for r in near if r['new_contacts'][0] and not r['initial_finger_contact']};neg={r['source_group'] for r in far if not r['contacts'][0]}
 if len(pos)<3 or len(neg)<3:errors.append('Independent near/far coverage lacks>=3 positive and negative source groups')
 old=json.load(open(S/'wm14_lateral_settled_vla_data/pack_manifest.json'))
 if signatures['independent_test']&{r['public_input_sha256'] for r in old['records']}:errors.append('Exact public inputs duplicated fitting or validation data')
 assert all(r['split']=='independent_test' and not r['fitting_allowed'] for r in manifest)
 coverage=dict(near_new_finger_positive_without_initial_source_groups=len(pos),far_no_finger_source_groups=len(neg),event_positive_counts=np.array(positives['independent_test']).sum(0).astype(int).tolist())
 atomic(ROOT/'independent_manifest.json',dict(records=manifest,coverage=coverage,training_allowed=False,model_selection_allowed=False,no_resampling=True))
 gate=dict(stage='wm16_lateral_independent_coverage_complete',passed=not errors,errors=errors,coverage=coverage,generation_failure_count=len(blind['bootstrap_failures']),model_scored=False,training_started=False,hardware_enabled=False,full_task_acceptance_passed=False)
 atomic(ROOT/'quality_gate.json',gate)
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 report=dict(stage='wm16_lateral_independent_coverage_complete',data_gate_passed=gate['passed'],scenes=12,observations=48,candidates=192,coverage=coverage,WM_model_calls=0,optimizer_instances=0,optimizer_updates=0,model_scored=False,hardware_enabled=False,source_hashes_unchanged=True,independent_manifest_sha256=sha(ROOT/'independent_manifest.json'),scope='Targeted paired lateral coverage only; ordinary VLA and broad physical coverage are separate. Not full task or model reliability.')
 atomic(ROOT/'report.json',report);atomic(ROOT/'audit_status.json',report)
if __name__=='__main__':main()
