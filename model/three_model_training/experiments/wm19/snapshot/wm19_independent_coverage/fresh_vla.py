"""Fresh preregistered actual VLA candidate ground truth; no WM fitting/scoring."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import sys,json,itertools
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'wm_consistent_v7'))
from physical_labels import np,mujoco,setup,render,public_motion,sha,atomic,snap,restore,segmentation_labels,geometry,physical_outcome,CONTACT_TYPES,category
from blind_scenes import ik
from layout_sampler import sample
from physics import context,classify
from scipy.spatial.transform import Rotation
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
ROOT=Path(__file__).resolve().parent
V5=ROOT.parent/'wm_contact_v5'
BASE=Path('/Users/imac/robot_training/camera_alignment_20261004')
def generate_blind(plan):
    template=V5/'sources'/json.loads((V5/'input_plan.json').read_text())['sources'][0]['source_group']/'scene.xml'
    out=ROOT/'blind';out.mkdir(exist_ok=False);rows=[];fail=[];warmup_records=[]
    for k,seed in enumerate(plan['future_blind_seeds']):
        m,d,ix=setup(template);renderer=mujoco.Renderer(m,360,640);ctx=context(m,0,'initial')
        perm=list(itertools.permutations(range(3)))[k%6];task='; '.join(f'put one {c} cube in compartment {perm[i]+1}' for i,c in enumerate(('red','yellow','green')))+'; do not move black, white, or the other cubes'
        folder=out/f'scene_{k:02d}';folder.mkdir()
        try:
            scatter,_=sample(seed,[.08,.33],8,selected_indices=[0,2+k%2,4+k%2]);assert scatter is not None
            d.qpos[ix.q]=[0,-.05,.1,0,0,0];d.qpos[ix.lq]=-.04;d.qpos[ix.rq]=.04;d.ctrl[ix.aids]=[0,-.05,.1,0,0,0,.08]
            for i in range(8):
                adr=int(m.jnt_qposadr[m.joint('pick_cube_free' if i==0 else f'cube_free_{i}').id]);q=Rotation.from_euler('z',float(scatter[i,2])).as_quat();d.qpos[adr:adr+7]=[*scatter[i,:2],.015,q[3],*q[:3]]
            m.light_diffuse[0]=np.array([.5]*3)*np.random.default_rng(seed).uniform(.85,1.15);mujoco.mj_forward(m,d)
            # Fixed preregistered warmup; cameras/physics/label function unchanged.
            history=[];positions=[];support=[];warm_geometry=geometry(m)
            expected_support={tuple(sorted((obj,warm_geometry['floor']))) for obj in warm_geometry['objects']}
            original_public_command=public_motion(m,d,ix)[14:].copy();original_ctrl=d.ctrl[ix.aids].copy()
            assert np.array_equal(original_ctrl.astype('float32'),original_public_command)
            warm_start=float(d.time)
            for warm_step in range(1,201):
                d.ctrl[ix.aids]=original_ctrl;mujoco.mj_step(m,d)
                if any(classify(ctx,int(c.geom1),int(c.geom2)) for c in d.contact):raise RuntimeError('warmup bootstrap contact gate failed; no resampling')
                if warm_step in (120,160,200):
                    mujoco.mj_forward(m,d);history_full,history_small=render(m,d,renderer)
                    history.append((history_small,public_motion(m,d,ix),history_full,float(d.time)))
                    positions.append(np.stack([d.geom_xpos[x].copy() for x in sorted(warm_geometry['objects'])]))
                    support.append({tuple(sorted((int(c.geom1),int(c.geom2)))) for c in d.contact if c.dist<=0 and category(warm_geometry,int(c.geom1),int(c.geom2))==5})
            warm_drift=float(np.max(np.linalg.norm(np.stack(positions)-positions[-1],axis=2)))
            assert np.allclose([h[3]-warm_start for h in history],[.12,.16,.2],atol=5e-12,rtol=0)
            assert all(pairs==expected_support for pairs in support),'missing stable eight-object support, no retry'
            assert warm_drift<=plan['initial_settling_position_drift_m'],'warmup stability failed, no extension'
            warmup_records.append(dict(seed=seed,steps=200,observed_steps=[120,160,200],time_s=float(d.time),support_pairs_each_frame=[len(pairs) for pairs in support],object_last80ms_max_position_drift_m=warm_drift,passed=True))
            xyz=d.xpos[m.geom_bodyid[ctx.geoms[0]]].copy();yaw=float((scatter[0,2]+np.pi/4)%(np.pi/2)-np.pi/4)
            rotation=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('z',-np.pi/2).as_matrix()@Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix()
            for stage,goal,count in [('initial',None,0),('approach',xyz+[0,0,.105],42),('descend',xyz+[0,0,.015],34)]:
                ctx.stage=stage
                if goal is not None:
                    q0=d.ctrl[ix.aids[:6]].copy();gap0=float(d.ctrl[ix.aids[6]]);q=ik(m,ix,goal,rotation,q0)
                    for n in range(count):
                        t=(n+1)/count;s=t*t*(3-2*t);d.ctrl[ix.aids]=np.r_[q0+s*(q-q0),gap0+s*(.045-gap0)]
                        for _ in range(40):
                            mujoco.mj_step(m,d)
                            if any(classify(ctx,int(c.geom1),int(c.geom2)) for c in d.contact):raise RuntimeError('bootstrap contact gate failed; no resampling')
                        mujoco.mj_forward(m,d);history_full,history_small=render(m,d,renderer)
                        history.append((history_small,public_motion(m,d,ix),history_full,float(d.time)));history=history[-3:]
                # Current RGB and history anchor share one exact observation, not two renders.
                full=history[-1][2];images=history[-1][0];file=folder/(stage+'.npz')
                valid=np.array([len(history)>=3,len(history)>=2,True],np.uint8)
                history_images=np.stack([history[-k][0] if len(history)>=k else images for k in (3,2,1)])
                history_state=np.stack([history[-k][1] if len(history)>=k else public_motion(m,d,ix) for k in (3,2,1)])
                assert np.array_equal(history_images[-1],images) and np.array_equal(history_state[-1],public_motion(m,d,ix))
                history_times=np.array([h[3] for h in history],np.float64);assert np.allclose(history_times-history_times[-1],[-.08,-.04,0],atol=1e-10,rtol=0)
                np.savez_compressed(file,images=images,front=full[0],wrist=full[1],state=public_motion(m,d,ix),integration_state=snap(m,d),history_absolute_times_s=history_times,history_images=history_images,history_state=history_state,history_valid=valid,history_relative_times_s=np.array([-.08,-.04,0],np.float32))
                rows.append({'observation_id':f'wm14_independent_settled_blind_{seed}_{stage}','seed':seed,'cache':str(file.relative_to(ROOT)),'cache_sha256':sha(file),'task':task,'scene':str(template),'stage_annotation':stage,'actual_anchor_time_s':float(d.time),'history_actual_times_s':history_times.tolist()})
        except Exception as e:fail.append({'seed':seed,'bootstrap_failure':repr(e)})
        finally:
            renderer.close();atomic(out/'status.json',{'stage':'wm14_independent_settled_blind_generation','scenes_done':k+1,'scenes_total':12,'observations':len(rows),'bootstrap_failures':len(fail),'compute_host':'M2 Max'})
    manifest={'initial_settling_hold_s':.2,'warmup_records':warmup_records,'records':rows,'fixed_seeds':plan['future_blind_seeds'],'bootstrap_failures':fail,'source_WM9_sha256':sha(ROOT.parent/'wm_motion_fit_v9/fit/best.pt'),'WM_model_inference_executed':False,'fitting_allowed':False,'no_resampling':True}
    atomic(out/'manifest.json',manifest);return manifest

def main():
 plan=json.load(open(ROOT/'execution_plan.json'));torch.set_num_threads(4)
 vpath=ROOT.parent/'vla_fit/checkpoint_000150';assert sha(vpath/'model.safetensors')==plan['frozen_evidence_sha256']['vla_fit/checkpoint_000150/model.safetensors']
 cfg=PreTrainedConfig.from_pretrained(vpath);cfg.device='mps';cfg.vlm_model_name=str(BASE/'smolvlm2_7b375e1');cfg.load_vlm_weights=False
 vla=SmolVLAPolicy.from_pretrained(vpath,config=cfg,strict=True).eval()
 pre,post=make_pre_post_processors(cfg,vpath,preprocessor_overrides={'device_processor':{'device':'mps'},'tokenizer_processor':{'tokenizer_name':str(BASE/'smolvlm2_7b375e1')}})
 manifest=generate_blind(plan);out=ROOT/'vla_candidates';out.mkdir(exist_ok=False);rows=[]
 for n,r in enumerate(manifest['records']):
  m,d,ix=setup(Path(r['scene']));renderer=mujoco.Renderer(m,360,640)
  assert sha(ROOT/r['cache'])==r['cache_sha256']
  with np.load(ROOT/r['cache']) as a:
   baseline=a['integration_state'].copy();front=a['front'].copy();wrist=a['wrist'].copy();motion=a['state'].copy();images=a['history_images'].copy();motor=a['history_state'].copy();valid=a['history_valid'].copy()
  m.light_diffuse[0]=np.array([.5]*3)*np.random.default_rng(r['seed']).uniform(.85,1.15);restore(m,d,baseline)
  ids=segmentation_labels(m,d,renderer);geo=geometry(m);state=motion[:7];vs=state.copy();vs[6]=np.clip(vs[6]/.08,0,1)
  frame={'observation.state':torch.from_numpy(vs),'observation.images.global':torch.from_numpy(front).permute(2,0,1).float()/255,'observation.images.wrist':torch.from_numpy(wrist).permute(2,0,1).float()/255,'task':r['task']}
  vla.reset();torch.manual_seed(20261006)
  with torch.inference_mode():raw=post(vla.predict_action_chunk(pre(frame))).detach().cpu().numpy().reshape(50,7)[:8].copy()
  raw[:,6]=np.clip(raw[:,6],0,1)*.08;raw=np.clip(raw,m.actuator_ctrlrange[ix.aids,0],m.actuator_ctrlrange[ix.aids,1]);smooth=[];prev=state.copy()
  for cmd in raw:prev=prev+np.clip(cmd-prev,-np.array([.02]*6+[.002]),np.array([.02]*6+[.002]));smooth.append(prev.copy())
  chunks=np.stack([raw,np.array(smooth),np.repeat(state[None],8,axis=0),np.repeat(motion[14:][None],8,axis=0)]).astype('float32')
  for i,name in enumerate(('vla_raw','vla_smoothed','pose_hold','command_hold')):
   restore(m,d,baseline);truth=physical_outcome(m,d,ix,geo,chunks[i],renderer,ids)
   path=out/(str(n).zfill(3)+'_'+name+'.npz');np.savez_compressed(path,history_images=images,history_state=motor,history_valid=valid,history_relative_times_s=np.array([-.08,-.04,0],np.float32),actions=chunks[i],**truth)
   rows.append(dict(cache=str(path.relative_to(ROOT)),sha256=sha(path),seed=r['seed'],source_group='independent_vla_'+str(r['seed']),split='independent_test',observation_id=r['observation_id'],candidate=name,task=r['task'],history_valid=valid.tolist(),contacts=truth['target_contacts'].tolist(),new_contacts=truth['target_new_contacts'].tolist(),retention=truth['target_retention'].tolist(),actual_VLA_generated=True,fitting_allowed=False,inference_keys=['history_images','history_state','history_valid','actions']))
  renderer.close();atomic(ROOT/'vla_status.json',dict(stage='wm19_independent_settled_fresh_VLA_candidate_truth',observations_done=n+1,observations_total=len(manifest['records']),candidates_done=len(rows),candidates_total=144,compute_host='M2 Max',training_started=False));print('VLA observations',n+1,'candidates',len(rows),flush=True)
 atomic(ROOT/'vla_manifest.json',dict(records=rows,bootstrap_failures=manifest['bootstrap_failures'],observations=len(manifest['records']),candidates=len(rows),generation_gate_passed=not manifest['bootstrap_failures'] and len(rows)==144,no_resampling=True,fitting_allowed=False,VLA_weight_sha256=sha(vpath/'model.safetensors'),scope='Actual VLA candidate truth on new bootstrapped scenes, not full task success or WM scoring',hardware_enabled=False,full_task_acceptance_passed=False))
if __name__=='__main__':main()
