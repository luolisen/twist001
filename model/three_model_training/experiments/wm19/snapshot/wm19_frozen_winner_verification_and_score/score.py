"""Frozen selected WM19 vs WM17; fixed thresholds and separate registered scopes."""
from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parent;S=ROOT.parent
sys.path.insert(0,str(S/'wm_consistent_v7'))
from physical_labels import sha,atomic,np,CONTACT_TYPES,Image
sys.path.insert(0,str(ROOT))
from model import EntityMotionWM
import torch
INPUTS=('history_images','history_state','history_valid','actions')

def metrics(y,prob,known=None):
 actual=np.asarray(y)>.5;guess=np.asarray(prob)>=.5;known=np.ones_like(actual,bool) if known is None else np.asarray(known,bool);out={}
 for offset,name in [(0,'physical_contacts'),(9,'new_contacts'),(18,'unlocalized_contacts')]:
  out[name]={k:dict(known_labels=int(known[:,offset+j].sum()),positives=int((actual[:,offset+j]&known[:,offset+j]).sum()),negatives=int((~actual[:,offset+j]&known[:,offset+j]).sum()),false_negatives=int((actual[:,offset+j]&~guess[:,offset+j]&known[:,offset+j]).sum()),false_positives=int((~actual[:,offset+j]&guess[:,offset+j]&known[:,offset+j]).sum())) for j,k in enumerate(CONTACT_TYPES)}
 out['retention']={k:dict(known_labels=int(known[:,27+j].sum()),positives=int((actual[:,27+j]&known[:,27+j]).sum()),negatives=int((~actual[:,27+j]&known[:,27+j]).sum()),false_negatives=int((actual[:,27+j]&~guess[:,27+j]&known[:,27+j]).sum()),false_positives=int((~actual[:,27+j]&guess[:,27+j]&known[:,27+j]).sum())) for j,k in enumerate(('terminal_grasp','grip_loss'))}
 return out

def main():
 torch.set_num_threads(4);plan=json.load(open(ROOT/'execution_plan.json'));WINNER=S/'wm19_eight_candidate_tournament/c05';report=json.load(open(WINNER/'fit/report.json'));selected=WINNER/'fit/best.pt';assert sha(selected)==report['weight_sha256']
 ck=torch.load(selected,map_location='cpu',weights_only=True);assert ck['execution_plan_sha256']==sha(WINNER/'execution_plan.json') and ck['model_code_sha256']==sha(WINNER/'model.py') and ck['train_code_sha256']==sha(WINNER/'train.py')
 net=EntityMotionWM().to('mps');net.load_state_dict(ck['state_dict']);net.eval()
 spec=importlib.util.spec_from_file_location('frozen_WM17_reference',S/'wm17_joint_event_head_fit/model.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);reference=old.EntityMotionWM().to('mps');reference.load_state_dict(torch.load(S/'wm17_joint_event_head_fit/fit/best.pt',map_location='cpu',weights_only=True)['state_dict']);reference.eval()
 independent=json.load(open(S/'wm19_independent_coverage/independent_manifest.json'))['records'];scopes={}
 for kind in ('physical','extra','vla'):scopes['new_'+kind]=[r for r in independent if r['phase_family']==kind]
 scopes['paired_lateral']=[dict(r,cache=str(S/'wm19_lateral_independent_coverage'/r['cache'])) for r in json.load(open(S/'wm19_lateral_independent_coverage/independent_manifest.json'))['records']]
 results={};forecast_by_observation={};maskchecks=[];ablation=[];scoretotal=0;family_masks=set();invalid_current_rejected=False
 with torch.inference_mode():
  for scope,rows in scopes.items():
   truth=[];pred9=[];pred8=[];perrows=[];flow_sum=0.;flow_cells=0.;masked_flow=0
   for i,r in enumerate(rows):
    path=Path(r['cache']);path=path if path.is_absolute() else S/'wm19_independent_coverage'/path;assert sha(path)==r['sha256']
    with np.load(path) as a:
     b=[torch.from_numpy(a[k].copy()).unsqueeze(0).to('mps') for k in INPUTS];truth.append(np.r_[a['target_contacts'],a['target_new_contacts'],a['target_unlocalized_contacts'],a['target_retention']]);state=a['history_state'][-1].copy();commands=a['actions'].copy();flow=a['target_object_history_flow'].copy() if 'target_object_history_flow' in a else None;validflow=a['target_object_history_flow_valid'].copy() if 'target_object_history_flow_valid' in a else None
    p=net(*b);base=reference(*b);assert all(bool(torch.isfinite(v).all()) for values in (p,base) for v in values.values());probs=p['events'].sigmoid().cpu().numpy()[0];ref=base['events'].sigmoid().cpu().numpy()[0];assert np.all(probs[9:18]<=probs[:9]+1e-6) and np.all(probs[18:27]<=probs[:9]+1e-6);pred9.append(probs);pred8.append(ref)
    if flow is not None:
     error=np.abs(p['observed_flow'].cpu().numpy()[0]-flow)*validflow[:,None];flow_sum+=float(error.sum());flow_cells+=int(validflow.sum()*2);masked_flow+=int(not validflow.any())
    if scope=='new_physical':
     repeat=[x.clone() for x in b];repeat[0]=b[0][:,-1,None].expand_as(b[0]).clone();repeat[1]=b[1][:,-1,None].expand_as(b[1]).clone();rprob=net(*repeat)['events'].sigmoid().cpu().numpy()[0]
     ablation.append(dict(cache=str(path),actual_history=probs.tolist(),repeated_current=rprob.tolist(),physical_decisions_changed=int(((probs[:9]>=.5)!=(rprob[:9]>=.5)).sum()),probability_max_difference=float(np.max(np.abs(probs-rprob)))))
    family=r.get('fixture_family',r.get('family'))
    if scope in ('new_physical','new_extra') and family not in family_masks:
     family_masks.add(family)
     for pattern in ((0,0,1),(0,1,1)):
      masked=[x.clone() for x in b];masked[2][:]=torch.tensor(pattern,device='mps',dtype=b[2].dtype);dirty=[x.clone() for x in masked]
      for j,v in enumerate(pattern):
       if not v:dirty[0][:,j]=255-masked[0][:,j];dirty[1][:,j]=float('nan')
      first=net(*masked);second=net(*dirty);diff=max(float((first[k]-second[k]).abs().max()) for k in first);assert diff==0 and all(bool(torch.isfinite(x).all()) for x in second.values())
      if pattern==(0,0,1):assert not bool(first['observed_flow'].abs().any()) and not bool(first['entity_flow'].abs().any())
      maskchecks.append(dict(case=i,family=family,mask=pattern,max_output_difference=diff))
     if not invalid_current_rejected:
      bad=[x.clone() for x in b];bad[2][:,-1]=0
      try:net(*bad)
      except ValueError:invalid_current_rejected=True
      else:raise AssertionError('invalid current accepted')
    perrows.append(dict(cache=str(path),seed=r.get('seed'),source_group=r.get('source_group'),candidate=r.get('candidate',r.get('family',r.get('fixture_family'))),actual=truth[-1].tolist(),WM19=probs.tolist(),WM17=ref.tolist()))
    if scope=='new_vla':
     oid=r['observation_id']
     if oid not in forecast_by_observation and len(forecast_by_observation)<6:
      forecast_by_observation[oid]=dict(observation_id=oid,task=r['task'],state=state.tolist(),history_valid=b[2].cpu().numpy()[0].tolist(),short_term_task_memory={'verified_progress':[],'target_confirmation_frames':0},observation_is_live=False,fresh_target_confirmation=False,wm_approval_released=False,execution_allowed=False,candidates=[])
     if oid in forecast_by_observation:
      regions=p['regions'].sigmoid().cpu().numpy()[0]
      cells={name:[[int(x) for x in np.unravel_index(regions[c,k].argmax(),(32,32))] for c in range(2)] for k,name in enumerate(CONTACT_TYPES)}
      forecast_by_observation[oid]['candidates'].append(dict(name=r['candidate'],actions=commands.tolist(),physical_contacts=dict(zip(CONTACT_TYPES,probs[:9].tolist())),new_contacts=dict(zip(CONTACT_TYPES,probs[9:18].tolist())),unlocalized_contacts=dict(zip(CONTACT_TYPES,probs[18:27].tolist())),any_grasp_estimate=float(probs[27]),grip_loss_estimate=float(probs[28]),motion_delta=p['motion'].cpu().numpy()[0].tolist(),highest_contact_region_cells=cells,execution_allowed=False))
    scoretotal+=1;atomic(ROOT/'score_status.json',dict(stage='wm19_registered_scoring',scope=scope,scope_done=i+1,scope_total=len(rows),candidates_done=scoretotal,candidates_total=504,compute_host='M2 Max',weight_sha256=report['weight_sha256']))
   result=dict(scope=scope,samples=len(rows),models={'WM19':metrics(truth,pred9),'WM17':metrics(truth,pred8)},rows=perrows,WM19_weight_sha256=sha(selected),WM17_weight_sha256=sha(S/'wm17_joint_event_head_fit/fit/best.pt'),threshold=.5,observed_flow_mae_pixels=flow_sum/max(1,flow_cells) if flow_cells else None,observed_flow_truth_available=bool(flow_cells),visible_flow_scalar_cells=flow_cells,masked_flow_samples=masked_flow,independent_test=True,full_task_acceptance_passed=False,hardware_enabled=False)
   if scope=='new_extra':
    keep=[i for i,r in enumerate(rows) if r['family'].startswith('retain') and r['start_grasp']]
    result['actual_start_grasp_only_retention']={name:metrics(np.array(truth)[keep],np.array(pred)[keep])['retention'] for name,pred in [('WM19',pred9),('WM17',pred8)]};result['actual_start_grasp_samples']=len(keep)
   atomic(ROOT/(scope+'_report.json'),result);results[scope]={k:v for k,v in result.items() if k!='rows'}
  assert len(forecast_by_observation)==6 and all(len(v['candidates'])==4 for v in forecast_by_observation.values())
 assert scoretotal==504 and len(family_masks)==9 and len(maskchecks)==18 and invalid_current_rejected
 atomic(ROOT/'trained_mask_history_report.json',dict(weight_sha256=sha(selected),mask_checks=maskchecks,mask_cases=9,patterns_per_case=2,invalid_current_rejected=invalid_current_rejected,max_output_difference=0,history_ablation=ablation,physical_decisions_changed=sum(r['physical_decisions_changed'] for r in ablation),scope='Actual trained model structural invariance and newphysical history contribution, not full task or solecause',hardware_enabled=False))
 atomic(ROOT/'jev_input.json',dict(records=list(forecast_by_observation.values()),wm_weight_sha256=sha(selected),model_code_sha256=sha(ROOT/'model.py'),vla_weight_sha256=sha(S/'vla_fit/checkpoint_000150/model.safetensors'),interface_tested=False,execution_allowed=False,hardware_enabled=False,full_task_acceptance_passed=False))
 critical=('object_nonfinger','object_object','object_tray_wall','robot_tray','robot_floor');failures=[];missing_support=[]
 checkscopes={**{k:v['models']['WM19'] for k,v in results.items()},**{k:v['metrics']|{'retention':v['retention']} for k,v in report['validation'].items()}}
 for scope,m in checkscopes.items():
  for family in ('physical_contacts','new_contacts'):
   names=critical+('robot_self',) if family=='new_contacts' else critical
   for kind in names:
    v=m[family][kind]
    if not v['positives'] or not v['negatives']:missing_support.append(dict(scope=scope,family=family,kind=kind,positives=v['positives'],negatives=v['negatives']))
    if v['false_negatives'] or (v['negatives'] and v['false_positives']/v['negatives']>.1):failures.append(dict(scope=scope,family=family,kind=kind,**v))
 for scope in ('paired_lateral','lateral_validation'):
  for family in ('physical_contacts','new_contacts'):
   v=checkscopes[scope][family]['object_finger']
   if v['positives']<3 or v['negatives']<3 or v['false_negatives'] or v['false_positives']/max(1,v['negatives'])>.1:failures.append(dict(scope=scope,family=family,kind='object_finger',**v))
 for kind,v in results['new_extra']['actual_start_grasp_only_retention']['WM19'].items():
  if v['false_negatives'] or v['false_positives']:failures.append(dict(scope='new_actual_start_grasp',kind=kind,**v))
 for scope,v in report['validation'].items():
  if v.get('actual_start_grasp_retention'):
   for kind,c in v['actual_start_grasp_retention'].items():
    if c['false_negatives'] or c['false_positives']:failures.append(dict(scope=scope+'_actual_start_grasp',kind=kind,**c))
 assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in ck['state_dict'].items()),'Scoring changed loaded parameters'
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
 atomic(ROOT/'score_report.json',dict(scopes=results,scored_candidates=504,weight_sha256=sha(selected),threshold=.5,independent_used_for_selection=False,compute_host='M2 Max',hardware_enabled=False,full_task_acceptance_passed=False))
 atomic(ROOT/'quality_gate.json',dict(stage='wm19_finite_fit_and_scoring_complete',weight_sha256=sha(selected),code_sha256=plan['code_sha256'],data_fingerprints=plan['frozen_evidence_sha256'],physical_reliability_subgate_passed=not failures,failed_checks=failures,missing_scope_support=missing_support,missing_scope_is_not_coverage_pass=True,trained_mask_gate_passed=True,Jev_current_model_interface_tested=False,full_task_acceptance_passed=False,hardware_enabled=False,old_evidence_unchanged=True,automatic_second_candidate=False,pending=['real_Jev_interface_bound_to_this_model','public_RGB_target_confirmation','short_term_task_memory','complete_three_model_task_loop'],next='Review separate original/pair/entity/visibility validation and new independent physical/extra/VLA results. No automatic refit; failed physics blocks later release. Jev input is prepared but not tested.'))
if __name__=='__main__':main()
