"""One finite WM18 local-event candidate, grouped validation selection; independent sets never fit."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
from collections import defaultdict
import sys,json,math,random
ROOT=Path(__file__).resolve().parent;S=ROOT.parent.parent
sys.path.insert(0,str(S/'wm_consistent_v7'))
from physical_labels import sha,atomic,CONTACT_TYPES,np
sys.path.insert(0,str(ROOT))
from model import EntityMotionWM,SCALES
import torch
from torch.nn import functional as F
INPUTS=('history_images','history_state','history_valid','actions')
TARGETS=('target_motion_delta','target_contacts','target_new_contacts','target_unlocalized_contacts','target_retention','target_contact_regions','target_grasp_region','target_future_rgb')

def source(row):
 return Path(row['cache'])
def manifest(name,folder):
 rows=json.load(open(S/name))['records'];out=[]
 for r in rows:
  p=Path(r['cache']);out.append(dict(r,cache=str(p if p.is_absolute() else S/folder/p)))
 return out

def disk_batch(rows):
 allrows=[]
 for r in rows:
  with np.load(source(r)) as a:
   d={k:torch.from_numpy(a[k].copy()) for k in INPUTS+TARGETS}
   d['target_object_history_flow']=torch.from_numpy(a['target_object_history_flow'].copy()) if 'target_object_history_flow' in a else torch.zeros((2,2,32,32))
   d['target_object_history_flow_valid']=torch.from_numpy(a['target_object_history_flow_valid'].copy()) if 'target_object_history_flow_valid' in a else torch.zeros((2,32,32),dtype=torch.uint8)
   d['target_entity_history_flow']=torch.from_numpy(a['target_entity_history_flow'].copy()) if 'target_entity_history_flow' in a else torch.zeros((2,2,32,32))
   d['target_entity_history_flow_valid']=torch.from_numpy(a['target_entity_history_flow_valid'].copy()) if 'target_entity_history_flow_valid' in a else torch.zeros((2,32,32),dtype=torch.uint8)
   allrows.append(d)
 return {k:torch.stack([d[k] for d in allrows]).to('mps') for k in allrows[0]}

CACHE={}
def batch(rows):
 if all(r['cache'] in CACHE for r in rows):
  return {k:torch.stack([CACHE[r['cache']][k] for r in rows]).to('mps') for k in next(iter(CACHE.values()))}
 return disk_batch(rows)
def preload(rows,budget):
 global CACHE
 from concurrent.futures import ThreadPoolExecutor
 def decode(r):
  with np.load(source(r)) as a:
   d={k:torch.from_numpy(a[k].copy()) for k in INPUTS+TARGETS}
   for k,shape,dtype in [('target_object_history_flow',(2,2,32,32),torch.float32),('target_entity_history_flow',(2,2,32,32),torch.float32),('target_object_history_flow_valid',(2,32,32),torch.uint8),('target_entity_history_flow_valid',(2,32,32),torch.uint8)]:
    d[k]=torch.from_numpy(a[k].copy()) if k in a else torch.zeros(shape,dtype=dtype)
  return r['cache'],d
 used=0
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,(key,d) in enumerate(pool.map(decode,rows)):
   assert key not in CACHE
   used+=sum(v.numel()*v.element_size() for v in d.values());assert used<=budget
   CACHE[key]=d
   if i%256==0:atomic(ROOT/'cache_status.json',dict(stage='decoding_training_RAM_cache',rows_done=i+1,rows_total=len(rows),bytes=used))
 assert len(CACHE)==13407
 atomic(ROOT/'cache_status.json',dict(stage='training_RAM_cache_ready',rows_done=len(CACHE),rows_total=len(rows),bytes=used,budget=budget,validation_cached=False,independent_cached=False))


def losses(net,b,weights,pair_bags=None,row_importance=None):
 assert row_importance is not None and row_importance.shape==(b['actions'].shape[0],) and bool(torch.isfinite(row_importance).all()) and bool((row_importance>0).all())
 p=net(*[b[k] for k in INPUTS]);y=torch.cat([b[k].float() for k in ('target_contacts','target_new_contacts','target_unlocalized_contacts','target_retention')],1)
 rgb=b['target_future_rgb'].float().permute(0,1,4,2,3)/255;rgb=F.interpolate(rgb.flatten(0,1),size=(32,32),mode='area').reshape(-1,2,3,32,32)
 terms=dict(motion=F.smooth_l1_loss(p['motion']/SCALES.to('mps'),b['target_motion_delta']/SCALES.to('mps')),events=(F.binary_cross_entropy_with_logits(p['events'],y,reduction='none').mean(1)*row_importance).mean(),regions=.2*F.binary_cross_entropy_with_logits(p['regions'],b['target_contact_regions'].float(),pos_weight=torch.tensor(10.,device='mps')),grasp=.1*F.binary_cross_entropy_with_logits(p['grasp_region'],b['target_grasp_region'].float(),pos_weight=torch.tensor(10.,device='mps')),future=.2*F.l1_loss(p['future_rgb32'],rgb))
 mask=b['target_object_history_flow_valid'].float()[:,:,None]
 terms['observed_flow']=.1*(F.smooth_l1_loss(p['observed_flow'],b['target_object_history_flow'].float(),reduction='none')*mask).sum()/(mask.sum()*2).clamp_min(1)
 entitymask=b['target_entity_history_flow_valid'].float()[:,:,None]
 terms['entity_flow']=.1*(F.smooth_l1_loss(p['entity_flow'],b['target_entity_history_flow'].float(),reduction='none')*entitymask).sum()/(entitymask.sum()*2).clamp_min(1)
 rank=[]
 for start,kind in pair_bags or []:
  k=CONTACT_TYPES.index(kind);truth=b['target_contacts'][start:start+5,k]>.5;logits=p['events'][start:start+5,k]
  if bool(truth.any()) and bool((~truth).any()):rank.append(F.relu(.5-(logits[truth][:,None]-logits[~truth][None])).mean())
 terms['contrast']=.1*torch.stack(rank).mean() if rank else p['events'].sum()*0
 return sum(terms.values()),terms


def make_epoch_schedule(epoch,plan,oldtrain,baglist,entitytrain,visibilitytrain,lateraltrain):
 rng=random.Random(plan['schedule_seed']+epoch)
 original=oldtrain.copy();rng.shuffle(original);bags=baglist.copy();rng.shuffle(bags)
 auxiliaries=[]
 for source_rows in [entitytrain,visibilitytrain,lateraltrain]:
  rows=source_rows.copy();rng.shuffle(rows);auxiliaries.append(rows)
 entity,visibility,lateral=auxiliaries;scheduled=[];mass=defaultdict(float);visits=defaultdict(int)
 for bi,start in enumerate(range(0,len(original),33)):
  oldrows=original[start:start+33];key,pairrows=bags[bi%len(bags)]
  rows=oldrows+pairrows+[entity[(bi*3+j)%len(entity)] for j in range(3)]+[visibility[(bi*3+j)%len(visibility)] for j in range(3)]+[lateral[(bi*4+j)%len(lateral)] for j in range(4)]
  scheduled.append((rows,[(len(oldrows),key[1])],len(oldrows)))
 count=len(scheduled);assert count==358 and len(scheduled[-1][0])==26 and all(len(rows)==48 for rows,_,_ in scheduled[:-1])
 known={r['cache'] for r in oldtrain+[r for _,rs in baglist for r in rs]+entitytrain+visibilitytrain+lateraltrain};assert len(known)==13407
 for rows,_,_ in scheduled:
  for row in rows:mass[row['cache']]+=1/(count*len(rows));visits[row['cache']]+=1
 assert set(mass)==known and all(v>0 for v in mass.values()) and abs(sum(mass.values())-1)<1e-10
 importance={key:1/(len(known)*value) for key,value in mass.items()};error=max(abs(mass[key]*importance[key]-1/len(known)) for key in known);assert error<1e-12
 evidence=dict(epoch=epoch,unique_rows=len(known),updates=count,last_batch_rows=26,uniform_row_mass_identity_max_error=error,records=[dict(cache=key,update_mass=mass[key],row_importance=importance[key],raw_exposures=visits[key]) for key in sorted(known)])
 return scheduled,importance,evidence


def importance_tensor(rows,importance):
 return torch.tensor([importance[r['cache']] for r in rows],dtype=torch.float32,device='mps')

@torch.no_grad()
def evaluate(net,rows):
 net.eval();ys=[];ps=[];flow_sum=0.;flow_cells=0.;entity_sum=0.;entity_cells=0.;motion_errors=[]
 for offset in range(0,len(rows),48):
  b=batch(rows[offset:offset+48]);p=net(*[b[k] for k in INPUTS]);ys.append(torch.cat([b[k].float() for k in ('target_contacts','target_new_contacts','target_unlocalized_contacts','target_retention')],1).cpu().numpy());ps.append(p['events'].sigmoid().cpu().numpy())
  motion_errors.append(float(F.mse_loss(p['motion']/SCALES.to('mps'),b['target_motion_delta']/SCALES.to('mps'))))
  emask=b['target_entity_history_flow_valid'].float()[:,:,None];entity_sum+=float(((p['entity_flow']-b['target_entity_history_flow'].float()).abs()*emask).sum());entity_cells+=float(emask.sum()*2)
  mask=b['target_object_history_flow_valid'].float()[:,:,None];flow_sum+=float(((p['observed_flow']-b['target_object_history_flow'].float()).abs()*mask).sum());flow_cells+=float(mask.sum()*2)
 y=np.concatenate(ys)>.5;probs=np.concatenate(ps);guess=probs>=.5
 assert not (probs[:,9:18]>probs[:,:9]+1e-6).any() and not (probs[:,18:27]>probs[:,:9]+1e-6).any()
 metrics={}
 for off,name in [(0,'physical_contacts'),(9,'new_contacts'),(18,'unlocalized_contacts')]:
  metrics[name]={k:dict(positives=int(y[:,off+j].sum()),negatives=int((~y[:,off+j]).sum()),false_negatives=int((y[:,off+j]&~guess[:,off+j]).sum()),false_positives=int((~y[:,off+j]&guess[:,off+j]).sum())) for j,k in enumerate(CONTACT_TYPES)}
 targeted_lateral=all(r.get('family','').startswith('settled_vla_') and r.get('source_group','').startswith('lateral_vla_') for r in rows)
 critical=('object_finger',) if targeted_lateral else ('object_nonfinger','object_object','object_tray_wall','robot_tray','robot_floor')
 absent_positive_categories=[f+'/'+k for f in ('physical_contacts','new_contacts') for k in critical if metrics[f][k]['positives']==0]
 positive_rates=[v['false_negatives']/v['positives'] for f in ('physical_contacts','new_contacts') for k,v in metrics[f].items() if k in critical and v['positives']]
 negative_rates=[v['false_positives']/max(1,v['negatives']) for f in ('physical_contacts','new_contacts') for k,v in metrics[f].items() if k in critical]
 keep=[i for i,r in enumerate(rows) if (r.get('actual_start_grasp') or r.get('start_grasp'))]
 grasp_metrics={name:dict(positives=int(y[keep,27+j].sum()),negatives=len(keep)-int(y[keep,27+j].sum()),false_negatives=int((y[keep,27+j]&~guess[keep,27+j]).sum()),false_positives=int((~y[keep,27+j]&guess[keep,27+j]).sum())) for j,name in enumerate(('terminal_grasp','grip_loss'))} if keep else None
 assert positive_rates,'A source validation has no registered selectable positive categories'
 return dict(selection_categories=list(critical),missing_positive_selection_categories=absent_positive_categories,missing_positive_is_not_acceptance=True,actual_start_grasp_samples=len(keep),actual_start_grasp_retention=grasp_metrics,samples=len(rows),metrics=metrics,critical_macro_fn=float(np.mean(positive_rates)),critical_macro_fp=float(np.mean(negative_rates)),event_brier=float(np.mean((y-probs)**2)),normalized_motion_mse=float(np.mean(motion_errors)),observed_flow_mae_pixels=flow_sum/max(1,flow_cells),visible_flow_scalar_cells=int(flow_cells),entity_flow_mae_pixels=entity_sum/max(1,entity_cells),visible_entity_flow_scalar_cells=int(entity_cells),retention={name:dict(positives=int(y[:,27+j].sum()),negatives=int((~y[:,27+j]).sum()),false_negatives=int((y[:,27+j]&~guess[:,27+j]).sum()),false_positives=int((~y[:,27+j]&guess[:,27+j]).sum())) for j,name in enumerate(('terminal_grasp','grip_loss'))})

def main():
 plan=json.load(open(ROOT/'execution_plan.json'));plan['target_epoch']=int(sys.argv[1]);plan['preflight_only']=len(sys.argv)>2 and sys.argv[2]=='preflight';torch.set_num_threads(4);torch.manual_seed(plan['training_seed']);random.seed(plan['initialization_seed'])
 assert json.load(open(S/'wm19_independent_coverage/quality_gate.json'))['coverage_gate_passed']
 assert json.load(open(S/'wm19_lateral_independent_coverage/quality_gate.json'))['passed']
 assert json.load(open(S/'wm14_lateral_settled_vla_data/quality_gate.json'))['passed']
 assert json.load(open(S/'wm10_representative_motion_data/quality_gate.json'))['data_gate_passed']
 assert json.load(open(S/'wm11_visibility_data/quality_gate.json'))['data_gate_passed']
 assert json.load(open(S/'wm_visible_base_v9/combined_data_gate.json'))['data_gate_passed']
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
 original=manifest('wm_temporal_v8/pack_manifest.json','wm_temporal_v8');pairs=manifest('wm_visible_base_v9/combined_pack_manifest.json','');entity=manifest('wm10_representative_motion_data/pack_manifest.json','wm10_representative_motion_data');visibility=manifest('wm11_visibility_data/pack_manifest.json','wm11_visibility_data');lateral=manifest('wm14_lateral_settled_vla_data/pack_manifest.json','wm14_lateral_settled_vla_data')
 for i,r in enumerate(original+pairs+entity+visibility+lateral):
  assert sha(source(r))==r['sha256'],r['cache']
  if i%128==0:atomic(ROOT/'audit_status.json',dict(stage='wm18_fit_data_audit',samples_done=i,samples_total=17672,compute_host='M2 Max'))
 groups={}
 for name,records in [('original',original),('pair',pairs),('entity',entity),('visibility',visibility),('lateral',lateral)]:
  groups[name]={split:{r['source_group'] for r in records if r['split']==split} for split in ('train','validation','independent_test')}
  for a,b in [('train','validation'),('train','independent_test'),('validation','independent_test')]:assert not groups[name][a]&groups[name][b]
 independent=[]
 for directory in ['wm19_independent_coverage','wm19_lateral_independent_coverage']:
  for r in json.load(open(S/directory/'independent_manifest.json'))['records']:independent.append(dict(r,cache=str(S/directory/r['cache']),private_audit_cache=str(S/directory/r['private_audit_cache']) if r.get('private_audit_cache') else None))
 assert len(independent)==504
 fitrows=original+pairs+entity+visibility+lateral
 fit_seeds={str(r['seed']) for r in fitrows if r.get('seed') is not None}
 fit_paths={str(source(r)) for r in fitrows if r['split'] in ('train','validation')}
 assert not fit_seeds&{str(r['seed']) for r in independent};assert not fit_paths&{r['cache'] for r in independent}
 for r in independent:
  assert sha(Path(r['cache']))==r['sha256']
  if r.get('private_audit_cache'):assert sha(Path(r['private_audit_cache']))==r['private_audit_sha256']
 atomic(ROOT/'audit_status.json',dict(stage='wm18_fit_data_audit_complete',samples_done=17672,samples_total=17672,source_groups={n:{k:len(v) for k,v in splits.items()} for n,splits in groups.items()},independent_rows_in_fit=0,data_gate_passed=True,compute_host='M2 Max'))
 lateraltrain=[r for r in lateral if r['split']=='train'];assert len(lateraltrain)==512
 visibilitytrain=[r for r in visibility if r['split']=='train'];oldtrain=[r for r in original if r['split']=='train'];pairtrain=[r for r in pairs if r['split']=='train'];entitytrain=[r for r in entity if r['split']=='train'];valsets={'original_validation':[r for r in original if r['split']=='validation'],'pair_validation':[r for r in pairs if r['split']=='validation'],'entity_validation':[r for r in entity if r['split']=='validation'],'visibility_validation':[r for r in visibility if r['split']=='validation'],'lateral_validation':[r for r in lateral if r['split']=='validation']};assert len(valsets['lateral_validation'])==256
 assert (len(oldtrain),len(pairtrain),len(entitytrain),len(valsets['original_validation']),len(valsets['pair_validation']),len(valsets['entity_validation']))==(11792,440,327,3460,120,141)
 assert len(visibilitytrain)==336 and len(valsets['visibility_validation'])==168
 bags={}
 for r in pairtrain:bags.setdefault((r['source_group'],r['family']),[]).append(r)
 assert len(bags)==88 and all(len(v)==5 for v in bags.values())
 positive=np.zeros(29);n=0
 for r in oldtrain+pairtrain+entitytrain+visibilitytrain+lateraltrain:
  with np.load(source(r)) as a:positive+=np.r_[a['target_contacts'],a['target_new_contacts'],a['target_unlocalized_contacts'],a['target_retention']];n+=1
 assert n==13407
 weights=torch.ones(29,dtype=torch.float32,device='mps')

 torch.set_num_threads(plan['torch_threads']);preload(oldtrain+pairtrain+entitytrain+visibilitytrain+lateraltrain,plan['RAM_cache_budget_bytes'])
 torch.manual_seed(plan['initialization_seed']);net=EntityMotionWM().to('mps');ck=torch.load(S/'wm17_joint_event_head_fit/fit/best.pt',map_location='cpu',weights_only=True)
 prefixes=tuple(plan['trainable_parameter_prefixes']);net.load_state_dict(ck['state_dict'],strict=True)
 assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in ck['state_dict'].items())
 for name,p in net.named_parameters():p.requires_grad_(name.startswith(prefixes))
 parameters=[p for p in net.parameters() if p.requires_grad];assert len(parameters)>0
 frozen={k:v.clone() for k,v in ck['state_dict'].items() if not k.startswith(prefixes)}
 def frozen_check():
  assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in frozen.items())
  assert all(p.grad is None for name,p in net.named_parameters() if not name.startswith(prefixes))
 out=ROOT/'fit';out.mkdir(exist_ok=True)
 if not (out/'initial.pt').exists():
  torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},source_weight_sha256=sha(S/'wm17_joint_event_head_fit/fit/best.pt'),learned_local_projection_preserved=True,initialization_seed=plan['initialization_seed'],model_code_sha256=sha(ROOT/'model.py'),execution_plan_sha256=sha(ROOT/'execution_plan.json')),out/'initial.pt')
 baglist=list(bags.items());_,probe_importance,probe_mass=make_epoch_schedule(1,plan,oldtrain,baglist,entitytrain,visibilitytrain,lateraltrain);atomic(ROOT/'preflight_row_mass.json',probe_mass)
 probe_bag=next((k,rs) for k,rs in baglist if k[1]=='object_object')
 probe_entity=[next(r for r in entitytrain if r['family']==family) for family in plan['probe_entity_families']]
 probe_visibility=[next(r for r in visibilitytrain if r['family']==family) for family in plan['probe_visibility_families']]
 probe_lateral=[next(r for r in lateraltrain if r['stage_annotation']==stage and r['candidate']==variant) for stage,variant in [('descend_near','pose_hold'),('descend_far','pose_hold'),('descend_near','vla_raw'),('descend_far','vla_raw')]]
 probe_rows=oldtrain[:33]+probe_bag[1]+probe_entity+probe_visibility+probe_lateral;assert len(probe_rows)==48 and all(r['split']=='train' for r in probe_rows)
 b=batch(probe_rows);disk=disk_batch(probe_rows);assert all(torch.equal(b[k],disk[k]) for k in b);del disk;net.eval()
 from model import base
 reference=EntityMotionWM().to('mps');reference.load_state_dict(ck['state_dict']);reference.eval()
 with torch.no_grad():
  ref=reference(b['history_images'],b['history_state'],b['history_valid'],b['actions']);actual=net(b['history_images'],b['history_state'],b['history_valid'],b['actions']);identity=max(float((ref[k]-actual[k]).abs().max()) for k in ref);assert identity==0
 del reference
 initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
 loss,terms=losses(net,b,weights,[(33,probe_bag[0][1])],importance_tensor(probe_rows,probe_importance));assert all(bool(torch.isfinite(v)) for v in terms.values())
 assert any(t.requires_grad for k,t in terms.items() if k not in ('events','contrast')),'Action encoder auxiliary loss must be differentiable'
 net.zero_grad(set_to_none=True);loss.backward();assert all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
 norms={prefix:sum(float(p.grad.abs().sum()) for name,p in net.named_parameters() if name.startswith(prefix) and p.grad is not None) for prefix in prefixes}
 assert all(math.isfinite(v) and v>0 for v in norms.values()),norms
 frozen_check();assert all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items())
 atomic(ROOT/'loss_preflight_report.json',dict(actual_batch_samples=48,weights_unchanged=True,source_WM17_loaded_exactly=True,source_WM17_output_max_difference=identity,optimizer_instances=0,optimizer_updates=0,new_module_gradient_l1=norms,learned_projection_preserved=True,all_loss_finite=True,legacy_non_event_outputs_constant_wrt_trainable_parameters=False,action_encoder_auxiliary_gradients_active=True,original_event_pair_ranking_remains_active=True,terms={k:float(v.detach()) for k,v in terms.items()},event_unit_positive_weights=True,row_mass_sha256=sha(ROOT/'preflight_row_mass.json'),structural_loss_preflight_passed=True,trainable_parameter_prefixes=prefixes,compute_host='M2 Max'))
 net.zero_grad(set_to_none=True)
 if plan.get('preflight_only',False):
  atomic(ROOT/'preflight_complete.json',dict(passed=True,optimizer_instances=0,optimizer_updates=0,source_parameters_unchanged=True));return
 opt=torch.optim.AdamW(parameters,lr=plan['optimizer']['learning_rate'],weight_decay=1e-4);history=[];updates=0;best=float('inf');bestepoch=0;batches=358;start_epoch=1
 if (out/'latest.pt').exists():
  resumed=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
  assert resumed['execution_plan_sha256']==sha(ROOT/'execution_plan.json')
  net.load_state_dict(resumed['state_dict'],strict=True);opt.load_state_dict(resumed['optimizer_state']);torch.set_rng_state(resumed['torch_rng_state'])
  updates=resumed['updates'];start_epoch=resumed['epoch']+1;best=resumed['best_selection_key'];bestepoch=resumed['best_epoch']
  history=json.load(open(out/'report.json'))['history'];assert len(history)==resumed['epoch'];frozen_check()
 assert plan['target_epoch']<=30
 for epoch in range(start_epoch,plan['target_epoch']+1):
  net.eval() # Deterministic eval operations, with registered six-prefix gradients enabled.
  schedule,importance,mass=make_epoch_schedule(epoch,plan,oldtrain,baglist,entitytrain,visibilitytrain,lateraltrain);mass_path=out/('row_mass_epoch_%02d.json'%epoch);atomic(mass_path,mass)
  lossvalues=[];eventvalues=[];displacement_samples=[];exposures=0;entityexposures=0;visibilityexposures=0;lateralexposures=0;original_seen=0;gradient_summaries={p:0. for p in prefixes}
  for bi,(rows,positions,original_count) in enumerate(schedule):
   original_seen+=original_count;b=batch(rows);opt.zero_grad(set_to_none=True)
   loss,terms=losses(net,b,weights,positions,importance_tensor(rows,importance));assert bool(torch.isfinite(loss));event=terms['events'];sample_step=bi%8==0
   before=[p.detach().cpu().numpy().astype(np.float64).ravel().copy() for p in parameters] if sample_step else None
   event_reference=torch.autograd.grad(event,parameters,retain_graph=True) if sample_step else None
   auxiliary=sum(v for k,v in terms.items() if k not in ('events','contrast'))
   aux_reference=torch.autograd.grad(auxiliary,parameters,retain_graph=True,allow_unused=True) if sample_step else None
   if sample_step:assert all(bool(torch.isfinite(g).all()) for g in aux_reference if g is not None)
   loss.backward();assert all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
   for prefix in prefixes:gradient_summaries[prefix]+=sum(float(p.grad.abs().sum()) for name,p in net.named_parameters() if name.startswith(prefix) and p.grad is not None)
   grad=np.concatenate([g.detach().cpu().numpy().astype(np.float64).ravel() for g in event_reference]) if sample_step else None
   auxgrad=np.concatenate([g.detach().cpu().numpy().astype(np.float64).ravel() if g is not None else np.zeros(p.numel()) for g,p in zip(aux_reference,parameters)]) if sample_step else None
   norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert bool(torch.isfinite(norm));opt.step();updates+=1;assert updates<=10740
   if sample_step:
    delta=np.concatenate([v-p.detach().cpu().numpy().astype(np.float64).ravel() for v,p in zip(before,parameters)])
    displacement_samples.append(dict(batch_index=bi,updates=updates,event_gradient_dot_actual_before_minus_after=float(np.dot(grad,delta)),auxiliary_gradient_dot_actual_before_minus_after=float(np.dot(auxgrad,delta)),event_auxiliary_raw_gradient_dot=float(np.dot(grad,auxgrad)),raw_gradient_not_AdamW_step_guarantee=True));frozen_check()
   exposures+=5;entityexposures+=3;visibilityexposures+=3;lateralexposures+=4;lossvalues.append(float(loss.detach()));eventvalues.append(float(event.detach()))
   if bi%8==0:atomic(out/'status.json',dict(stage='wm18_local_event_training',epoch=epoch,max_epochs=plan['max_epochs'],updates=updates,batches_this_epoch=bi+1,batches_per_epoch=batches,loss=float(np.mean(lossvalues)),event_loss=float(np.mean(eventvalues)),remaining_base_parameters_frozen=True,compute_host='M2 Max'))
  frozen_check();assert all(v>0 for v in gradient_summaries.values()),gradient_summaries
  atomic(out/('displacement_epoch_%02d.json'%epoch),dict(epoch=epoch,samples=displacement_samples,event_opposing_samples=sum(x['event_gradient_dot_actual_before_minus_after']<0 for x in displacement_samples),auxiliary_opposing_samples=sum(x['auxiliary_gradient_dot_actual_before_minus_after']<0 for x in displacement_samples),gradient_l1_by_trainable_module=gradient_summaries,frozen_remaining_parameters_unchanged=True))
  val={name:evaluate(net,rows) for name,rows in valsets.items()}
  keys=[20*v['critical_macro_fn']+v['critical_macro_fp']+v['normalized_motion_mse']+.1*v['event_brier']+.05*(v['observed_flow_mae_pixels']+v['entity_flow_mae_pixels']) for v in val.values()];key=float(np.mean(keys));assert math.isfinite(key)
  row=dict(epoch=epoch,max_epochs=plan['max_epochs'],updates=updates,loss=float(np.mean(lossvalues)),event_loss=float(np.mean(eventvalues)),original_samples_this_epoch=len(oldtrain),pair_exposures_this_epoch=exposures,entity_exposures_this_epoch=entityexposures,visibility_exposures_this_epoch=visibilityexposures,lateral_exposures_this_epoch=lateralexposures,validation=val,selection_key=key,row_mass_sha256=sha(mass_path),uniform_row_mass_identity_max_error=mass['uniform_row_mass_identity_max_error'],frozen_remaining_parameters_unchanged=True);history.append(row);atomic(out/'status.json',dict(stage='wm18_local_event_training',compute_host='M2 Max',**row));print(json.dumps(row),flush=True)
  def checkpoint():return dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer_state=opt.state_dict(),updates=updates,epoch=epoch,best_epoch=bestepoch,best_selection_key=best,execution_plan_sha256=sha(ROOT/'execution_plan.json'),model_code_sha256=sha(ROOT/'model.py'),train_code_sha256=sha(ROOT/'train.py'),source_weight_sha256=sha(S/'wm17_joint_event_head_fit/fit/best.pt'),row_mass_sha256=sha(mass_path),event_objective='unit_BCE_inverse_exact_epoch_update_mass',trainable_parameter_prefixes=prefixes,frozen_remaining_parameters_unchanged=True,data_fingerprints=plan['frozen_evidence_sha256'],torch_rng_state=torch.get_rng_state(),hardware_enabled=False)
  if key<best:
   best=key;bestepoch=epoch;torch.save(checkpoint(),out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
  torch.save(checkpoint(),out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
  # Fixed tournament rung budget; no within-rung early stopping.
 selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True);net.load_state_dict(selected['state_dict']);frozen_check()
 report=dict(stage='wm18_local_event_fit_complete',epochs=len(history),best_epoch=bestepoch,best_checkpoint_updates=selected['updates'],updates=updates,weight_sha256=sha(out/'best.pt'),initial_weight_sha256=sha(out/'initial.pt'),validation={name:evaluate(net,rows) for name,rows in valsets.items()},history=history,threshold=.5,event_objective='unit_BCE_inverse_exact_epoch_update_mass',trainable_parameter_prefixes=prefixes,frozen_remaining_parameters_unchanged=True,legacy_non_event_outputs_constant=False,action_encoder_auxiliary_gradients_active=True,event_pair_ranking_active=True,candidate_count=1,independent_test_used_for_selection=False,compute_host='M2 Max',hardware_enabled=False,full_task_acceptance_passed=False)
 for n,h in plan['frozen_evidence_sha256'].items():assert sha(S/n)==h,n
 for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
 atomic(out/'report.json',report);atomic(out/'status.json',{k:v for k,v in report.items() if k!='history'})
if __name__=='__main__':main()
