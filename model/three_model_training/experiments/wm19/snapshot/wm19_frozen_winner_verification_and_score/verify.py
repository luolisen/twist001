from pathlib import Path
import json,hashlib,sys,importlib.util,datetime,torch,numpy as np
D=Path(__file__).resolve().parent;S=D.parent;T=S/'wm19_eight_candidate_tournament'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=D/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def main():
 torch.set_num_threads(2);p=read(D/'execution_plan.json')
 for n,h in p['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 for n,h in p['code_sha256'].items():assert sha(D/n)==h,n
 source=torch.load(S/'wm17_joint_event_head_fit/fit/best.pt',map_location='cpu',weights_only=True)['state_dict']
 sys.path.insert(0,str(T/'c05'));import train as production
 original=production.manifest('wm_temporal_v8/pack_manifest.json','wm_temporal_v8');pairs=production.manifest('wm_visible_base_v9/combined_pack_manifest.json','');entity=production.manifest('wm10_representative_motion_data/pack_manifest.json','wm10_representative_motion_data');visibility=production.manifest('wm11_visibility_data/pack_manifest.json','wm11_visibility_data');lateral=production.manifest('wm14_lateral_settled_vla_data/pack_manifest.json','wm14_lateral_settled_vla_data')
 tr=lambda rows:[r for r in rows if r['split']=='train'];bags={}
 for r in tr(pairs):bags.setdefault((r['source_group'],r['family']),[]).append(r)
 sourcehash=0
 for rows in [original,pairs,entity,visibility,lateral]:
  for row in rows:assert sha(Path(row['cache']))==row['sha256'];sourcehash+=1
 assert sourcehash==17672
 expected={'c01':12,'c02':6,'c03':24,'c04':6,'c05':30,'c06':6,'c07':12,'c08':6};reports={};epochs=0
 for cid,target in expected.items():
  C=T/cid;cp=read(C/'execution_plan.json');r=read(C/'fit/report.json');reports[cid]=r;prefix=tuple(cp['trainable_parameter_prefixes'])
  assert r['epochs']==target and r['updates']==target*358 and len(r['history'])==target
  initial=torch.load(C/'fit/initial.pt',map_location='cpu',weights_only=True)
  assert set(initial['state_dict'])==set(source) and all(torch.equal(initial['state_dict'][k],v) for k,v in source.items())
  for row in r['history']:
   epoch=row['epoch'];v=row['validation'];key=sum(20*x['critical_macro_fn']+x['critical_macro_fp']+x['normalized_motion_mse']+.1*x['event_brier']+.05*(x['observed_flow_mae_pixels']+x['entity_flow_mae_pixels']) for x in v.values())/5
   assert abs(key-row['selection_key'])<1e-10 and row['updates']==epoch*358
   m=C/('fit/row_mass_epoch_%02d.json'%epoch);assert sha(m)==row['row_mass_sha256'];stored=read(m)
   _,_,rebuilt=production.make_epoch_schedule(epoch,cp,tr(original),list(bags.items()),tr(entity),tr(visibility),tr(lateral));assert rebuilt==stored,(cid,epoch)
   assert len(stored['records'])==13407 and all(x['update_mass']>0 and abs(x['update_mass']*x['row_importance']-1/13407)<1e-12 for x in stored['records'])
   epochs+=1;save('status.json',dict(stage='wm19_postfit_integrity_verification',candidate=cid,epochs_verified=epochs,epochs_total=102,source_caches_verified=sourcehash,optimizer_updates=0))
  assert min(r['history'],key=lambda x:x['selection_key'])['epoch']==r['best_epoch']
  for name,epoch,updates in [('best',r['best_epoch'],r['best_checkpoint_updates']),('latest',target,target*358)]:
   ck=torch.load(C/('fit/'+name+'.pt'),map_location='cpu',weights_only=True)
   assert ck['epoch']==epoch and ck['updates']==updates and ck['optimizer_state']['state']
   assert ck['execution_plan_sha256']==sha(C/'execution_plan.json') and ck['model_code_sha256']==sha(C/'model.py') and ck['train_code_sha256']==sha(C/'train.py')
   assert ck['source_weight_sha256']==cp['source_weight_sha256'] and ck['row_mass_sha256']==sha(C/('fit/row_mass_epoch_%02d.json'%epoch))
   assert all(pg['lr']==cp['optimizer']['learning_rate'] for pg in ck['optimizer_state']['param_groups'])
   assert all(torch.equal(ck['state_dict'][k],v) for k,v in source.items() if not k.startswith(prefix))
  assert sha(C/'fit/best.pt')==r['weight_sha256']
 lb=read(T/'leaderboard.json');ids=list(expected)
 for ri,rr in enumerate(lb['rounds']):
  target=[6,12,24,30][ri];assert rr['target_epoch']==target
  ranking=sorted(ids,key=lambda cid:(reports[cid]['history'][target-1]['selection_key'],cid))
  assert [x['candidate'] for x in rr['leaderboard']]==ranking
  for x in rr['leaderboard']:assert x['last_epoch_key']==reports[x['candidate']]['history'][target-1]['selection_key']
  ids=ranking[:[4,2,1,1][ri]];assert rr['survivors']==ids
 winner=read(T/'winner.json');assert ids==[winner['candidate']]==['c05'] and epochs==102 and lb['total_updates']==36516
 assert winner['best_epoch']==17 and winner['best_checkpoint_updates']==6086 and sha(T/'c05/fit/best.pt')==winner['weight_sha256']
 caches=private=0
 for name in ['wm19_independent_coverage','wm19_lateral_independent_coverage']:
  B=S/name;g=read(B/'quality_gate.json');assert g.get('coverage_gate_passed',g.get('passed'));anchors={}
  for r in read(B/'independent_manifest.json')['records']:
   c=Path(r['cache']);c=c if c.is_absolute() else B/c;assert sha(c)==r['sha256'];caches+=1
   if r.get('private_audit_cache'):
    a=Path(r['private_audit_cache']);a=a if a.is_absolute() else B/a;anchors[str(a)]=r['private_audit_sha256']
  for a,h in anchors.items():assert sha(Path(a))==h
  private+=len(anchors)
 assert caches==504 and private==216
 save('verification_report.json',dict(passed=True,frozen_dependencies_verified=len(p['frozen_evidence_sha256']),source_cache_hashes_verified=17672,epochs_row_mass_verified=102,candidate_checkpoints_verified=16,initial_strict_WM17_identity=True,remaining_frozen_parameters_unchanged=True,optimizer_metadata_verified=True,latest_equal_budget_elimination_verified=True,winner=winner,independent_public_verified=504,independent_unique_private_verified=216,independent_model_calls=0,optimizer_instances=0,optimizer_updates=0,hardware_enabled=False))
if __name__=='__main__':main()
