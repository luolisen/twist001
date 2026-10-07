"""Bounded new self-contact/retention fixtures, private bootstrap is labels only."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'wm_consistent_v7'))
from physical_labels import np,mujoco,setup,render,public_motion,sha,atomic,snap,restore,segmentation_labels,geometry,physical_outcome,CONTACT_TYPES
from blind_scenes import ik
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent

def held(m,d,g):
 pairs=[{int(c.geom1),int(c.geom2)} for c in d.contact if c.dist<=0]
 return {o for o in g['objects'] if all({o,f} in pairs for f in g['fingers']) and not any(o in pair and pair&{g['floor'],g['bottom']} for pair in pairs)}
def init(template,seed):
 m,d,ix=setup(template);rng=np.random.default_rng(seed)
 d.qpos[ix.q]=[0,-.05,.1,0,0,0];d.qpos[ix.lq]=-.0225;d.qpos[ix.rq]=.0225;d.ctrl[ix.aids]=[0,-.05,.1,0,0,0,.045]
 for i in range(8):
  a=int(m.jnt_qposadr[m.joint('pick_cube_free' if i==0 else f'cube_free_{i}').id])
  xy=np.array([.34+.05*(i%3),.40+.05*(i//3)])+rng.uniform(-.004,.004,2)
  d.qpos[a:a+7]=[*xy,.015,1,0,0,0]
 m.light_diffuse[0]=np.array([.5]*3)*rng.uniform(.85,1.15);mujoco.mj_forward(m,d)
 return m,d,ix,rng

def record(m,d,ix,renderer,actions,path,seed,family,start=None):
 history=[];states=[]
 # Three actual observed frames with frozen preceding command, no future action applied yet.
 for n in range(3):
  mujoco.mj_forward(m,d);history.append(render(m,d,renderer)[1]);states.append(public_motion(m,d,ix))
  if n<2:
   for _ in range(40):mujoco.mj_step(m,d)
 g=geometry(m);start_held=held(m,d,g);ids=segmentation_labels(m,d,renderer);baseline=snap(m,d)
 truth=physical_outcome(m,d,ix,g,actions,renderer,ids)
 np.savez_compressed(path,history_images=np.stack(history),history_state=np.stack(states),history_valid=np.ones(3,np.uint8),history_relative_times_s=np.array([-.08,-.04,0],np.float32),actions=actions,**truth)
 private=ROOT/'private_audit'/path.name;np.savez_compressed(private,integration_state=baseline)
 return dict(cache=str(path.relative_to(ROOT)),sha256=sha(path),seed=seed,family=family,split='independent_test',source_group='independent_extra_'+str(seed),start_grasp=bool(start_held),target_contacts=truth['target_contacts'].tolist(),target_new_contacts=truth['target_new_contacts'].tolist(),target_retention=truth['target_retention'].tolist(),private_audit_cache=str(private.relative_to(ROOT)),private_audit_sha256=sha(private),inference_keys=['history_images','history_state','history_valid','actions'],fitting_allowed=False,bootstrap_annotation=start)

def main():
 plan=json.load(open(ROOT/'execution_plan.json'));source=json.load(open(ROOT.parent/'wm_contact_v5/input_plan.json'))['sources'][0]
 template=ROOT.parent/'wm_contact_v5/sources'/source['source_group']/'scene.xml';assert sha(template)==source['portable_scene_sha256']
 out=ROOT/'extra_tests';out.mkdir(exist_ok=False);(ROOT/'private_audit').mkdir(exist_ok=False);rows=[];fail=[];attempts=0
 for seed in plan['extra_seeds']:
  for kind in ('self_move','self_hold','retain_hold','retain_release'):
   attempts+=1;m,d,ix,rng=init(template,seed);renderer=mujoco.Renderer(m,360,640)
   try:
    if kind.startswith('self'):
     commands=np.repeat(d.ctrl[ix.aids][None],8,axis=0)
     if kind=='self_move':
      target=rng.uniform(m.actuator_ctrlrange[ix.aids[:6],0]*.8,m.actuator_ctrlrange[ix.aids[:6],1]*.8)
      for n in range(8):commands[n,:6]=d.ctrl[ix.aids[:6]]+(n+1)/8*(target-d.ctrl[ix.aids[:6]])
    else:
     src=np.array([.22,.20,.015])+np.r_[rng.uniform(-.01,.01,2),0]
     a=int(m.jnt_qposadr[m.joint('pick_cube_free').id]);d.qpos[a:a+7]=[*src,1,0,0,0];mujoco.mj_forward(m,d)
     rotation=Rotation.from_euler('z',-np.pi/2).as_matrix()@Rotation.from_euler('x',np.pi).as_matrix()
     # Forward physical approach, close and lift; no reset or fictitious grasp at anchor.
     for xyz,gap,count in [(src+[0,0,.105],.045,42),(src+[0,0,.015],.045,34),(src+[0,0,.015],.005,28),(src+[0,0,.12],.005,76)]:
      before=d.ctrl[ix.aids].copy();target=ik(m,ix,xyz,rotation,before[:6])
      for n in range(count):
       t=(n+1)/count;f=t*t*(3-2*t);d.ctrl[ix.aids]=np.r_[before[:6]+f*(target-before[:6]),before[6]+f*(gap-before[6])]
       for _ in range(40):mujoco.mj_step(m,d)
     mujoco.mj_forward(m,d)
     if not held(m,d,geometry(m)):raise RuntimeError('Actual elevated two-finger bootstrap grasp absent; retained failure, no resampling')
     commands=np.repeat(d.ctrl[ix.aids][None],8,axis=0)
     if kind=='retain_release':commands[:,6]=np.linspace(float(d.ctrl[ix.aids[6]]),.08,9)[1:]
    rows.append(record(m,d,ix,renderer,commands.astype('float32'),out/(str(seed)+'_'+kind+'.npz'),seed,kind))
   except Exception as e:
    private=ROOT/'private_audit'/(str(seed)+'_'+kind+'_failure.npz');np.savez_compressed(private,integration_state=snap(m,d))
    fail.append(dict(seed=seed,family=kind,error=repr(e),failure_state=str(private.relative_to(ROOT)),failure_state_sha256=sha(private)))
   finally:
    renderer.close();atomic(ROOT/'extra_status.json',dict(stage='wm19_independent_self_retention_generation',cases_done=attempts,cases_total=48,records=len(rows),failures=len(fail),compute_host='M2 Max',training_started=False))
  print('extra seed',seed,'attempts',attempts,'records',len(rows),'failures',len(fail),flush=True)
 selfrows=[r for r in rows if r['family'].startswith('self')];grasp=[r for r in rows if r['family'].startswith('retain') and r['start_grasp']]
 counts=dict(robot_self_new_positive=sum(r['target_new_contacts'][8] for r in selfrows),robot_self_new_negative=sum(not r['target_new_contacts'][8] for r in selfrows),retention_start_grasp=len(grasp),terminal_grasp_positive=sum(r['target_retention'][0] for r in grasp),terminal_grasp_negative=sum(not r['target_retention'][0] for r in grasp),grip_loss_positive=sum(r['target_retention'][1] for r in grasp),grip_loss_negative=sum(not r['target_retention'][1] for r in grasp))
 passed=not fail and all(counts[k]>=3 for k in ('robot_self_new_positive','robot_self_new_negative','terminal_grasp_positive','terminal_grasp_negative','grip_loss_positive','grip_loss_negative'))
 atomic(ROOT/'extra_manifest.json',dict(records=rows,failures=fail,counts=counts,coverage_gate_passed=passed,no_resampling=True,fitting_allowed=False,physics_unchanged=True,hardware_enabled=False,full_task_acceptance_passed=False))
if __name__=='__main__':main()
