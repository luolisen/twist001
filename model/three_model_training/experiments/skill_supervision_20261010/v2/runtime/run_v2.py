"""Real v2 entry. Interface mode has no simulator, control device or motion fallback.

Full mode remains gated by endpoint/path/local physics and independently validated
public carry/deposit evidence. The v1 reference run_online.py is not this entry.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import argparse,json,time,subprocess,importlib.util,traceback,hashlib
from pathlib import Path
import numpy as np
from fusion_core import atomic,sha,compact
from model_binding import load_vla,snapshot
from skill_router import SkillRouter,SPECS,digest
from skill_generators import PickGenerator,HoldGenerator,TransportGenerator,PlaceGenerator,RecoveryGenerator
from action_transaction import ActionTransaction
from gripper_continuation import Continuation
from terminal_hold import TerminalHold
from skill_model_bridge import FrozenWMBridge
from skill_session import SkillSession
from carry_evidence import public_features
from task_relative import observed_relation,predicted_relation

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def full_prerequisites(proof):
    keys=('hard_endpoint','full_path_guards','local_transport_physics','public_carry_validated',
          'public_support_validated','place_local_physics','real_model_bridges')
    missing=[k for k in keys if proof.get(k) is not True]
    if missing:raise RuntimeError('full_v2_prerequisites_missing:'+','.join(missing))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['interface','full'],required=True);a=parser.parse_args()
    plan=json.load(open(a.plan))
    if a.mode=='full':
        full_prerequisites(plan.get('full_prerequisites',{}))
        raise RuntimeError('full_physics_driver_not_enabled_before_verified_local_skills')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();service=None
    counts={'VLA':0,'WM':0,'Jev':0,'physics_steps':0,'control_commands':0}
    result={'mode':'zero_physics_real_model_interface','full_task_passed':None}
    def deadline():
        if time.monotonic()-start>plan['budgets']['wall_seconds']:raise TimeoutError('interface_total_wall_budget')
    def wait_file(path,cap):
        tick=time.monotonic()
        while not path.exists():
            deadline()
            if service.poll() is not None:raise RuntimeError('native_service_exited_before_reply')
            if time.monotonic()-tick>cap:raise TimeoutError('native_service_response_timeout')
            time.sleep(.1)
        return json.load(open(path))
    def no_motion(*args,**kwargs):raise RuntimeError('interface_mode_physics_and_control_disabled')
    try:
        for path,h in plan['code_sha256'].items():
            if sha(path)!=h:raise RuntimeError('registered_entry_source_mismatch:'+path)
        capture=Path(plan['capture']);
        if sha(capture)!=plan['capture_sha256']:raise RuntimeError('public_capture_binding_mismatch')
        z=np.load(capture);o={'time':float(z['timestamp_s']),'state':z['public_state'],
            'small':z['small_rgb'],'full':[z['front'],z['wrist']]}
        if o['time']!=0.:raise RuntimeError('interface_requires_registered_real_initial_capture')
        cal=json.load(open(plan['calibration']));g=Continuation(json.load(open(plan['gripper_rules'])))
        h=TerminalHold(json.load(open(plan['hold_rules'])));guard=g.observe(o);h.observe(o,g,guard)
        track={'front':g.association.front if guard['front_track_confirmed'] else None,
               'wrist':g.association.wrist if guard['wrist_track_confirmed'] else None,
               'conflict':guard['target_correspondence_ambiguous']}
        evidence=public_features(o,track,cal)
        atomic(out/'public_evidence.json',evidence)
        token=hashlib.sha256(json.dumps({'bbox':track['front']['bbox'],'public_visual_proposal':True},sort_keys=True).encode()).hexdigest()
        context={'task':'v2_red_to_registered_cell_0','target':token,'input_valid':guard['input_valid'],
             'target_valid':bool(track['front'] and not track['conflict']),'target_conflict':track['conflict'],
             'protection':guard.get('terminate'),'observation_time_s':o['time'],
             'last_clamp_target_m':float(o['state'][20]),'carry_evidence':evidence['carry_evidence'],
             'support_evidence':evidence['support_evidence'],'placement_evidence':evidence['placement_evidence']}
        router=SkillRouter(context['task'],token)
        pick=PickGenerator(ActionTransaction(g,h,offer_smoothed=False),np.array(plan['limits']),plan['actuator_names'])
        router.register(SPECS['Pick'].generator,pick);router.register(SPECS['Hold'].generator,HoldGenerator(h))
        # Factories are registered without asserting that unvalidated plans exist.
        router.register(SPECS['Transport'].generator,TransportGenerator)
        router.register(SPECS['Place'].generator,PlaceGenerator);router.register(SPECS['Recovery'].generator,RecoveryGenerator())
        import torch
        torch.set_num_threads(2);tick=time.monotonic()
        vla,pre,post,binding=load_vla(plan['VLA_binding']);before_vla=snapshot(vla)
        module=load_module('frozen_WM_v2',plan['WM_score_module']);wm=module.EntityMotionWM().to('mps')
        wm.load_state_dict(torch.load(plan['WM_weight'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        wm.eval().requires_grad_(False);before_wm=snapshot(wm)
        if before_wm!=json.load(open(plan['source_WM_snapshot'])):raise RuntimeError('WM_frozen_binding_mismatch')
        atomic(out/'VLA_binding.json',binding);atomic(out/'VLA_parameters_before.json',before_vla);atomic(out/'WM_parameters_before.json',before_wm)
        result['VLA_WM_initialization_seconds']=time.monotonic()-tick
        service_log=open(out/'jev_service.log','w')
        service=subprocess.Popen([plan['Jev_python'],str(Path(__file__).with_name('v2_jev_service.py')),'--plan',a.plan,'--output',str(out)],stdout=service_log,stderr=subprocess.STDOUT)
        ready=wait_file(out/'service_ready.json',120);atomic(out/'effective_Jev_service.json',ready)
        if ready['profile_internal_stages'] is not False:raise RuntimeError('internal_profiling_must_be_disabled')
        from PIL import Image
        images,states,valid,times=__import__('fusion_core').history_arrays([o])
        sheet=np.concatenate([np.concatenate(list(frame),axis=1) for frame in images],axis=0)
        Image.fromarray(sheet).save(out/'image.png')
        vla.reset();torch.manual_seed(plan['sampling_seed']);s=o['state'][:7].copy();s[6]=np.clip(s[6]/.08,0,1)
        frame={'observation.state':torch.from_numpy(s),
            'observation.images.global':torch.from_numpy(o['full'][0]).permute(2,0,1).float()/255,
            'observation.images.wrist':torch.from_numpy(o['full'][1]).permute(2,0,1).float()/255,'task':plan['task']}
        tick=time.perf_counter();counts['VLA']+=1
        from prediction_capture import predict_with_capture
        with torch.inference_mode():raw,network,pre_state=predict_with_capture(vla,pre,post,frame)
        torch.mps.synchronize();result['VLA_inference_seconds']=time.perf_counter()-tick
        physical=raw[:8].copy();physical[:,6]=np.clip(physical[:,6],0,1)*.08
        from limit_projection import project_arm_targets
        projected,projection=project_arm_targets(physical,np.array(plan['limits']))
        candidate,boundary=pick.preview(raw,projected,o,'interface/new_VLA_prediction',context['task']+'_initial')
        commands=candidate['commands'];np.savez_compressed(out/'VLA_prediction.npz',raw50=raw,network50=network,final_commands=commands)
        atomic(out/'action_binding.json',pick.tx.public_binding(candidate));atomic(out/'candidate_boundary.json',boundary)
        rpc_index=0
        def rpc(request,question='supervision'):
            nonlocal rpc_index
            if counts['Jev']>=plan['budgets']['Jev_calls']:raise RuntimeError('native_call_cap')
            i=rpc_index;rpc_index+=1;counts['Jev']+=1
            atomic(out/f'job_{i:03d}.json',{'request':request,'question':question})
            answer=wait_file(out/f'reply_{i:03d}.json',plan['response_timeout_s'])
            if not answer['ok']:raise RuntimeError('native_Jev_request_failed:'+answer['error'])
            return answer['response']
        state={'task':context['task'],'pick_policy_literal':plan['task'],'motor_state':o['state'].tolist(),
               'units':'six joint radians, measured gap/velocities/applied controls in original public 21-vector',
               'RGB_history':{'mask':valid.tolist(),'valid_times_s':[0.],'layout':'global,wrist; -80,-40ms missing black slots; current genuine RGB'},
               'public_stage_evidence':evidence,'task_relative_observation':observed_relation(cal,o['state'],track,0.)}
        base={'model':'NeoHorse-Jev-4B','state':json.dumps(compact(state),separators=(',',':')),
              'questions':{'target':{'type':'choice','instructions':'Propose whether task-relevant red target is visible in actual RGB. This is not verified identity, grasp or completion.',
                 'criteria':{'red':'Visible red target.','not_visible':'No red target visible.','uncertain':'Insufficient RGB evidence.'}}}}
        answer=rpc(base,'target');target=answer['answers']['target'];result['target_proposal']=target
        if target['choice']!='red':
            router.cancel('native_target_not_confirmed');result['stop_reason']='target_proposal_'+target['choice']
        else:
            state['native_target_proposal']=target;base['state']=json.dumps(compact(state),separators=(',',':'))
            real_wm=FrozenWMBridge(wm,torch,lambda:[o],cal)
            def forecast(cmd,observation):
                if counts['WM']>=plan['budgets']['WM_calls']:raise RuntimeError('WM_cap')
                counts['WM']+=1;f=real_wm(cmd,observation)
                f['task_relative_forecast']=predicted_relation(cal,o['state'],state['task_relative_observation'],f['motion_delta'])
                f['task_relative_forecast']['source']='WM_joint_delta_plus_static_FK'
                atomic(out/'WM_raw_and_FK_forecast.json',f)
                # Direct XYZ and adapter diagnostics stay in the exact archive;
                # native branch contains canonical FK movement plus all original risks.
                presented={k:v for k,v in f.items() if k not in ('raw_motion_and_FK_archive','wall_seconds')}
                return presented
            def archive(kind,value):atomic(out/(kind+'.json'),value)
            session=SkillSession(router,lambda:(o,context),forecast,rpc,no_motion,archive,max_wall_s=plan['budgets']['wall_seconds'],max_Jev=1)
            granted=session.authorize('Pick',commands,base,evidence)
            result['native_skill_authorized']=granted;result['stop_reason']='interface_checked_no_control' if granted else 'native_permission_declined'
            if granted:router.check(context,0.,time.monotonic(),reserve=True)
            router.cancel('zero_physics_interface_complete')
        result['registered_generators']={k:('factory_not_physically_validated' if isinstance(v,type) else type(v).__name__) for k,v in router.generators.items()}
        atomic(out/'router_events.json',router.events)
        if g.cursor or h.cursor or pick.tx.current is not None:raise RuntimeError('interface_consumed_unexecuted_plan')
        result['auxiliary_state_not_consumed']=True
        after_vla=snapshot(vla);after_wm=snapshot(wm)
        atomic(out/'VLA_parameters_after.json',after_vla);atomic(out/'WM_parameters_after.json',after_wm)
        result['VLA_parameters_unchanged']=before_vla==after_vla;result['WM_parameters_unchanged']=before_wm==after_wm
        result['real_model_interface_passed']=True
    except Exception as e:
        result.update(real_model_interface_passed=False,stop_reason='interface_exception',error=repr(e),traceback=traceback.format_exc())
    finally:
        if service is not None:
            (out/'stop_service').touch()
            try:service.wait(timeout=15)
            except subprocess.TimeoutExpired:service.terminate();service.wait(timeout=10)
            result['Jev_service_exit_code']=service.returncode
        result['calls']=counts;result['wall_seconds']=time.monotonic()-start
        atomic(out/'interface_result.json',result)

if __name__=='__main__':main()
