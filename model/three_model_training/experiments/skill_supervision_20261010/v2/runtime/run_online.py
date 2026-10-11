"""Finite online VLA/fusion research runner; no hardware or optimizer backend."""
import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'cgl')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import numpy as np
from fusion_core import (Budget, PublicMemory, atomic, sha, command_digest, filtered_candidates,
                         lookup_candidate, history_arrays, recovery_change, check_public, CONTACT_TYPES)
from fusion_core import execute_with_archive
from model_binding import load_vla, snapshot
from candidate_boundary import diagnose_candidates
from composite_stage_gate import PhaseGate, phase_of


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_plan(plan):
    if plan['hardware_enabled'] is not False or plan['optimizer_updates'] != 0:
        raise ValueError('simulation-only frozen inference required')
    if plan['action_correction'] not in ('none', 'joint_residual_half', 'explicit_arm_limit_projection'):
        raise ValueError('unregistered intervention')
    if plan.get('interface_revision', 'baseline') not in ('baseline', 'consistent_v1'):
        raise ValueError('unregistered interface revision')
    if plan['mode'] not in ('pure_vla', 'fusion'):
        raise ValueError('unknown control mode')
    for path, digest in plan['frozen_files_sha256'].items():
        if sha(path) != digest:
            raise ValueError('frozen dependency changed: ' + path)
    for filename, digest in plan['code_sha256'].items():
        if sha(Path(__file__).parent / filename) != digest:
            raise ValueError('runner code changed: ' + filename)
    if plan['command_count_per_block'] != 8 or plan['steps_per_command'] != 40:
        raise ValueError('registered 8x40 execution semantics required')
    if plan.get('joint_residual_scale', .5) != .5:
        raise ValueError('only the preregistered residual-halving intervention is allowed')


def main(plan_file=None, output_dir=None, shared=None):
    if plan_file is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--plan', required=True)
        parser.add_argument('--output', required=True)
        args = parser.parse_args()
        plan_file,output_dir=args.plan,args.output
    plan_path, output = Path(plan_file).resolve(), Path(output_dir).resolve()
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    output.mkdir(exist_ok=False, parents=True)
    atomic(output / 'effective_plan.json', plan)
    budget = Budget(plan['budgets'],parent=shared.get('budget') if shared is not None else None)
    start_wall = time.monotonic()
    service = None
    ipc_root=Path(shared['service_root']) if shared is not None else output
    run = None
    private_rows, decisions = [], []
    controller = None
    hold_controller = None
    phase_gate = None
    last_target_answer = None
    if plan.get("terminal_hold_aid"):
        from terminal_hold import TerminalHold
        hold_controller = TerminalHold(json.loads(Path(plan["terminal_hold_aid"]["rules_file"]).read_text()))
    if plan.get('gripper_execution_aid'):
        from gripper_continuation import Continuation
        controller = Continuation(json.loads(Path(plan['gripper_execution_aid']['rules_file']).read_text()))
    from action_transaction import ActionTransaction
    transactions=ActionTransaction(controller,hold_controller,
                                   offer_smoothed=not plan.get('exclude_vla_smoothed',False))
    from task_relative import observed_relation
    relation_calibration=json.loads(Path(plan['public_relation_calibration_file']).read_text()) if plan.get('dynamic_task_relation') else None
    def wall():
        if time.monotonic() - start_wall > plan['maximum_wall_seconds']:
            raise RuntimeError('maximum_wall_seconds_exhausted')
        if shared is not None and time.monotonic()-shared['start_wall']>shared['maximum_wall_seconds']:
            raise RuntimeError('suite_maximum_wall_seconds_exhausted')
    def status(stage, **extra):
        wall()
        atomic(output / 'status.json', {'stage': stage, 'counts': budget.counts, 'hardware_enabled': False, **extra})

    source_path = Path(plan['source_run_minimal'])
    sys.path.insert(0, str(source_path.parent))
    a = load_module('frozen_minimal_executor', source_path)
    a.verify()
    a.SOURCE = {k:v.copy() for k,v in np.load(a.P['expert_trace']).items()}
    a.R = output
    a.P = dict(a.P, maximum_physics_steps=plan['budgets']['physics_steps'],
               maximum_model_calls=plan['budgets']['VLA_calls'], maximum_wall_seconds=plan['maximum_wall_seconds'])
    original_outcome=a.physical_outcome
    def logged_outcome(*arguments,**kwargs):
        # Preserve the exact original outcome implementation and its sole mj_forward.
        result=original_outcome(*arguments,**kwargs)
        if run is not None:
            run.original_outcome_return={key:np.asarray(value).copy() for key,value in result.items()}
        return result
    a.physical_outcome=logged_outcome

    def public_health(t):
        g=controller.observations[-1] if controller.observations else {}
        last=controller.last_reliable
        return {'input_valid':g.get('input_valid') is True,
                'association_conflict':bool(g.get('target_correspondence_ambiguous',False)),
                'target_history_valid':last is not None and t-last<=controller.rules['maximum_unconfirmed_seconds']+1e-9,
                'protection':controller.public_stop or hold_controller.stop_reason}

    class LiveRun(a.Run):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history = []
            self.history_index = 0
            self.block_events = None
            self.original_outcome_return=None
            self.contact_pair_rows=[]
            self._aux_execution_in_progress=False
            self.history_folder = self.folder / 'public_history'
            self.history_folder.mkdir()
            atomic(self.folder/'private_geom_name_map.json',{
                'evaluation_only':True,
                'geoms':{str(i):{'geom_name':self.m.geom(i).name,'body_id':int(self.m.geom_bodyid[i]),
                                 'body_name':self.m.body(int(self.m.geom_bodyid[i])).name} for i in range(self.m.ngeom)},
                'displacement_reference_time':float(self.d.time),
                'reference_positions':{str(g):self.initial_xyz[g].tolist() for g in self.objects}})

        def capture_now(self):
            state_before = a.snap(self.m, self.d)
            full, small = a.render(self.m, self.d, self.renderer)
            if not np.array_equal(state_before, a.snap(self.m, self.d)):
                raise RuntimeError('RGB rendering altered integration state')
            observation = {'time': float(self.d.time), 'small': small.copy(),
                           'state': a.public_motion(self.m, self.d, self.ix).copy(),
                           'full': [im.copy() for im in full]}
            if self.history and abs(self.history[-1]['time'] - observation['time']) < 1e-10:
                self.history[-1] = observation
            else:
                self.history.append(observation)
            np.savez_compressed(self.history_folder / f'capture_{self.history_index:05d}.npz',
                                timestamp_s=observation['time'], small_rgb=small,
                                public_state=observation['state'], front=full[0], wrist=full[1])
            self.history_index += 1
            self.history = self.history[-3:]
            if controller is not None:
                guard = controller.observe(observation)
                atomic(output/'gripper_continuation_trace.json',controller.archived())
                if hold_controller is not None:
                    hold_guard=hold_controller.observe(observation,controller,guard)
                    atomic(output/'terminal_hold_trace.json',hold_controller.archived())
                    if hold_guard.get('terminate') and not guard.get('terminate'):
                        guard['terminate']='terminal_hold:'+hold_guard['terminate']
                if relation_calibration is not None:
                    observation['task_relative_observation']=observed_relation(relation_calibration,observation['state'],
                        {'front':controller.association.front if guard.get('front_track_confirmed') else None,
                         'wrist':controller.association.wrist if guard.get('wrist_track_confirmed') else None,
                         'conflict':guard.get('target_correspondence_ambiguous',False)},observation['time'])
                if guard.get('terminate'):
                    self.stop_reason='public_gripper_guard:'+guard['terminate']
                    if self._aux_execution_in_progress:
                        raise a.SafetyStop(self.stop_reason)
            if phase_gate is not None and phase_gate.inflight is not None:
                try:phase_gate.check_live(observation['time'],time.monotonic(),public_health(observation['time']))
                except RuntimeError as exc:
                    self.stop_reason='stage_permission:'+str(exc)
                    atomic(output/'stage_permission_trace.json',phase_gate.archived())
                    if self._aux_execution_in_progress:raise a.SafetyStop(self.stop_reason)
            return observation

        def step(self, model, data):
            wall()
            if budget.counts['physics_steps'] >= budget.limits['physics_steps']:
                raise RuntimeError('budget_exhausted:physics_steps')
            prior_rows = len(self.rows)
            try:
                super().step(model, data)
            finally:
                if len(self.rows) > prior_rows:
                    budget.charge('physics_steps')
                    pairs = {tuple(sorted((int(c.geom1),int(c.geom2)))) for c in data.contact if c.dist <= 0}
                    for contact in data.contact:
                        if contact.dist<=0:
                            self.contact_pair_rows.append([len(self.rows),float(data.time),int(contact.geom1),
                                                           int(contact.geom2),float(contact.dist)])
                    if self.block_events is not None:
                        for ga,gb in pairs:
                            category = a.category(self.geo, ga, gb)
                            if category is not None:
                                self.block_events['contacts'][category] = 1
                                if (ga,gb) not in self.start_pairs:
                                    self.block_events['new_contacts'][category] = 1
                    if len(self.rows) % 40 == 0:
                        self.capture_now()

        def execute_logged(self,commands,block,kind):
            start_rows=len(self.rows)
            start_time=float(self.d.time)
            def archive(completed):
                # mj_getState only: do not repair contacts or advance physics on interruption.
                np.savez_compressed(self.folder/f'block_{block:03d}_exit.npz',integration_state=a.snap(self.m,self.d))
                np.savez_compressed(self.folder/'private_contact_pairs.npz',
                    contact_rows=np.asarray(self.contact_pair_rows,dtype=np.float64).reshape(-1,5),
                    columns=np.array(['physical_step','absolute_time_s','geom1_id','geom2_id','distance_m']))
                if not completed:
                    audit={'block':block,'kind':kind,'start_time':start_time,'end_time':float(self.d.time),
                           'actual_horizon_seconds':float(self.d.time)-start_time,
                           'physics_steps':len(self.rows)-start_rows,'private_evaluator_only':True,
                           'software_interruption':True,'partial_prefix_only':True,
                           'labels_complete_original_implementation':False,
                           'completed_320ms_window':len(self.rows)-start_rows==320,
                           'stop_reason':self.stop_reason,
                           'current_integration_state':f'episode/block_{block:03d}_exit.npz',
                           'target_contacts':self.block_events['contacts'].tolist() if self.block_events else None,
                           'target_new_contacts':self.block_events['new_contacts'].tolist() if self.block_events else None,
                           'error':traceback.format_exc()}
                    private_rows.append(audit)
                    atomic(output/'private_block_events.json',private_rows)
                    atomic(self.folder/f'private_partial_block_{block:03d}.json',audit)
            def archive_failure(error):
                try:
                    atomic(self.folder/f'prefix_archive_error_{block:03d}.json',{'error':repr(error)})
                except Exception:
                    pass
            return execute_with_archive(lambda:self._execute_logged_impl(commands,block,kind),archive,archive_failure)

        def _execute_logged_impl(self, commands, block, kind):
            # No pose/target truth is returned into PublicMemory.
            budget.charge('action_chunks')
            before_state = a.public_motion(self.m, self.d, self.ix).copy()
            before_time = float(self.d.time)
            row_start = len(self.rows)
            initial_pairs = {tuple(sorted((int(c.geom1),int(c.geom2)))) for c in self.d.contact if c.dist <= 0}
            def any_grasp(pairs):
                return {g for g in self.geo['objects'] if all(tuple(sorted((g,f))) in pairs for f in self.geo['fingers'])
                        and not any(tuple(sorted((g,f))) in pairs for f in [self.geo['floor'],self.geo['bottom']])}
            start_grasp = any_grasp(initial_pairs)
            self.block_events = {'contacts': np.zeros(9,np.uint8), 'new_contacts': np.zeros(9,np.uint8)}
            self.original_outcome_return=None
            self._aux_execution_in_progress=True
            try:
                ok = super().execute(commands, block)
            finally:
                self._aux_execution_in_progress=False
            self.capture_now()  # Refresh genuine terminal rendering after original mj_forward.
            terminal_pairs = {tuple(sorted((int(c.geom1),int(c.geom2)))) for c in self.d.contact if c.dist <= 0}
            terminal_grasp = any_grasp(terminal_pairs)
            audit = {'block':block,'kind':kind,'start_time':before_time,'end_time':float(self.d.time),
                     'actual_horizon_seconds':float(self.d.time)-before_time,'physics_steps':len(self.rows)-row_start,
                     'target_contacts':self.block_events['contacts'].tolist(),
                     'target_new_contacts':self.block_events['new_contacts'].tolist(),
                     'terminal_any_grasp':bool(terminal_grasp),'grip_loss':bool(start_grasp-terminal_grasp),
                     'completed_320ms_window':len(self.rows)-row_start==320,
                     'private_evaluator_only':True,'stop_reason':self.stop_reason}
            if self.original_outcome_return is not None:
                truth=self.original_outcome_return
                outcome_file=self.folder/f'private_original_outcome_{block:03d}.npz'
                np.savez_compressed(outcome_file,**truth)
                audit['original_physical_outcome_npz']=str(outcome_file.relative_to(output))
                audit['original_physical_outcome_sha256']=sha(outcome_file)
                audit['labels_complete_original_implementation']=True
                for key in ('target_motion_delta','target_contacts','target_new_contacts',
                            'target_unlocalized_contacts','target_retention'):
                    audit[key]=truth[key].tolist()
                if not np.array_equal(self.block_events['contacts'],truth['target_contacts']):
                    raise RuntimeError('side contact audit differs from original labels')
                if not np.array_equal(self.block_events['new_contacts'],truth['target_new_contacts']):
                    raise RuntimeError('side new-contact audit differs from original labels')
            else:
                audit['labels_complete_original_implementation']=False
                audit['partial_prefix_only']=True
            np.savez_compressed(self.folder/'private_contact_pairs.npz',
                                contact_rows=np.asarray(self.contact_pair_rows,dtype=np.float64).reshape(-1,5),
                                columns=np.array(['physical_step','absolute_time_s','geom1_id','geom2_id','distance_m']))
            if self.stop_reason:
                audit['stop_trigger_contacts']=[{'geom1_id':ga,'geom2_id':gb,
                     'geom1_name':self.m.geom(ga).name,'geom2_name':self.m.geom(gb).name,
                     'category':a.category(self.geo,ga,gb)} for ga,gb in sorted(terminal_pairs)]
                audit['stop_trigger_displacements']=[{'geom_id':g,'geom_name':self.m.geom(g).name,
                     'reference_position':self.initial_xyz[g].tolist(),
                     'current_position':self.d.xpos[int(self.m.geom_bodyid[g])].tolist(),
                     'displacement_m':float(np.linalg.norm(self.d.xpos[int(self.m.geom_bodyid[g])]-self.initial_xyz[g])),
                     'audit_target_id':self.audit_target} for g in self.objects]
                audit['stop_kind']='control_target_rejected_before_block' if self.stop_reason=='control_target_interception' else 'public_observation_guard_after_executed_command' if self.stop_reason.startswith('public_gripper_guard:') else 'physical_protection_after_executed_step'
                audit['trigger_step_executed']=self.stop_reason!='control_target_interception'
            private_rows.append(audit)
            atomic(output / 'private_block_events.json',private_rows)
            self.block_events = None
            return ok and self.stop_reason is None, before_state, before_time, len(self.rows)-row_start

    try:
        status('loading_complete_recent_VLA_binding')
        import torch
        torch.set_num_threads(2)
        if shared is not None and 'VLA' in shared:
            if shared['VLA_binding']!=plan['VLA_binding']:
                raise RuntimeError('suite attempted to change frozen VLA bundle')
            vla,pre,post,binding_record=shared['VLA']
        else:
            vla, pre, post, binding_record = load_vla(plan['VLA_binding'])
            if shared is not None:
                shared['VLA']=(vla,pre,post,binding_record);shared['VLA_binding']=plan['VLA_binding']
        vbefore = snapshot(vla)
        atomic(output / 'VLA_parameters_before.json',vbefore)
        atomic(output / 'VLA_binding_verified.json',binding_record)
        wm = None
        if plan['mode'] == 'fusion':
            status('loading_frozen_WM_and_Jev')
            wm_binding={key:plan[key] for key in ('WM_score_module','WM_weight','source_WM_snapshot')}
            if shared is not None and 'WM' in shared:
                if shared['WM_binding']!=wm_binding:
                    raise RuntimeError('suite attempted to change frozen WM bundle')
                wm=shared['WM']
            else:
                wm_module = load_module('frozen_online_WM',plan['WM_score_module'])
                wm = wm_module.EntityMotionWM().to('mps')
                weight = torch.load(plan['WM_weight'],map_location='cpu',weights_only=True)
                wm.load_state_dict(weight['state_dict'],strict=True)
                wm.eval().requires_grad_(False)
                if shared is not None:
                    shared['WM']=wm;shared['WM_binding']=wm_binding
            wbefore=snapshot(wm)
            if wbefore != json.loads(Path(plan['source_WM_snapshot']).read_text()):
                raise RuntimeError('WM snapshot differs')
            atomic(output / 'WM_parameters_before.json',wbefore)
            for directory in ('ipc','public_images'):
                (ipc_root/directory).mkdir(parents=True,exist_ok=True)
            (output/'public_images').mkdir(exist_ok=True);(output/'public_inputs').mkdir()
            jev_binding={key:plan[key] for key in ('Jev_bundle','Jev_python','prior_BF16_snapshot')}
            if shared is not None and 'Jev_service' in shared:
                if shared['Jev_binding']!=jev_binding:
                    raise RuntimeError('suite attempted to change frozen Jev bundle')
                service=shared['Jev_service']
            else:
                with open(ipc_root/'jev_service.log','ab') as log:
                    service=subprocess.Popen([plan['Jev_python'],'-u',str(Path(__file__).parent/'jev_online_service.py'),
                                              '--plan',shared.get('service_plan_path',str(plan_path)) if shared is not None else str(plan_path),
                                              '--output',str(ipc_root)],cwd=Path(__file__).parent,
                                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                if shared is not None:
                    shared['Jev_service']=service;shared['Jev_binding']=jev_binding
            atomic(output/'Jev_service_launch.json',{'pid':service.pid})
        initial=np.load(plan['initial_state_file'])[plan.get('initial_state_key','integration_state')]
        run=LiveRun('episode',initial=initial)
        run.capture(0)
        current=run.capture_now()
        if plan['mode']=='fusion':
            if plan.get('initial_history_file'):
                z=np.load(plan['initial_history_file'])
                required={'history_images','history_state','history_times_s','history_front','history_wrist'}
                if not required<=set(z.files):
                    raise RuntimeError('recorded history cache missing required public arrays')
                history=[]
                for i in range(3):
                    history.append({'time':float(z['history_times_s'][i]),'small':z['history_images'][i].copy(),
                                    'state':z['history_state'][i].copy(),
                                    'full':[z['history_front'][i].copy(),z['history_wrist'][i].copy()]})
                history_arrays(history)
                if abs(history[-1]['time']-current['time'])>1e-9 or not np.array_equal(history[-1]['state'],current['state']):
                    raise RuntimeError('cached history does not end at restored public state')
                if not all(np.array_equal(history[-1]['full'][i],current['full'][i]) for i in range(2)):
                    raise RuntimeError('cached final RGB differs from current rendering')
                run.history=history
            else:
                if not plan.get('WM_allows_missing_history',False):
                    raise RuntimeError('genuine initial WM history unavailable and missing-frame mode not registered')
                atomic(output/'initial_history_status.json',{'valid':[0,0,1],
                       'invalid_slots':'zero_arrays_not_repeated_images','bootstrap_physics_steps':0,
                       'original_start_preserved':True})
        anchor=controller.association.front
        if anchor is None:raise RuntimeError('no_initial_public_track_for_phase_binding')
        import hashlib
        track_token=hashlib.sha256(json.dumps({'initial_public_front_bbox':anchor['bbox'],
            'identity_is_visual_track_not_private_ID':True},sort_keys=True).encode()).hexdigest()
        phase_gate=PhaseGate(plan['task'],track_token,max_events=plan['maximum_online_stage_events'],wall_ttl=plan['phase_permission_wall_ttl_s'])
        atomic(output/'stage_permission_trace.json',phase_gate.archived())
        block=len(run.blocks)
        memory=PublicMemory()
        sampling_seed=plan['sampling_seed']
        replan_previous=None
        same_replan_count=0
        no_progress_blocks=0
        ipc_index=shared.get('ipc_index',0) if shared is not None else 0
        stop_reason='decision_budget_window_completed'
        for local_index in range(plan['maximum_decisions']):
            wall()
            if budget.counts['physics_steps'] >= plan['budgets']['physics_steps']:
                break
            budget.charge('decisions')
            decision_id=f'{plan.get("episode_name","episode")}_decision_{local_index:03d}'
            status('online_VLA_proposal',decision_id=decision_id)
            observation=run.history[-1]
            budget.charge('VLA_calls')
            vla.reset()
            torch.manual_seed(sampling_seed)
            state=observation['state'][:7].copy(); state[6]=np.clip(state[6]/.08,0,1)
            frame={'observation.state':torch.from_numpy(state),
                   'observation.images.global':torch.from_numpy(observation['full'][0]).permute(2,0,1).float()/255,
                   'observation.images.wrist':torch.from_numpy(observation['full'][1]).permute(2,0,1).float()/255,
                   'task':plan['task']}
            inference_start=time.perf_counter()
            with torch.inference_mode():
                from prediction_capture import predict_with_capture
                raw50,network50,preprocessed_state=predict_with_capture(vla,pre,post,frame)
            torch.mps.synchronize()
            vla_seconds=time.perf_counter()-inference_start
            if not np.isfinite(raw50).all():
                raise RuntimeError('nonfinite VLA output')
            physical=raw50[:8].copy(); physical[:,6]=np.clip(physical[:,6],0,1)*.08
            intervention=load_module('registered_intervention',Path(__file__).parent/'action_intervention.py')
            if plan['action_correction']=='explicit_arm_limit_projection':
                from limit_projection import project_arm_targets
                corrected,correction_record=project_arm_targets(physical,run.m.actuator_ctrlrange[run.ix.aids])
            else:
                corrected,correction_record=intervention.apply_intervention(physical,observation['state'],plan['action_correction'])
            arm_projected_before_gripper=corrected.copy()
            actuator_names=[run.m.actuator(i).name for i in run.ix.aids]
            raw_boundary=diagnose_candidates({'raw_mapped_before_intervention':physical},
                                             run.m.actuator_ctrlrange[run.ix.aids],actuator_names)
            candidates,boundary,gripper_record,hold_record=transactions.preview(raw50,corrected,observation,
                'episode/prediction_%03d.npz'%local_index,run.m.actuator_ctrlrange[run.ix.aids],actuator_names,decision_id)
            corrected=candidates[0]['commands'] if candidates else corrected
            atomic(run.folder/f'candidate_transactions_{local_index:03d}.json',
                {'boundary':boundary,'bindings':[transactions.public_binding(c) for c in candidates],
                 'unselected_outcomes':'unknown_no_physical_execution'})
            np.savez_compressed(run.folder/f'prediction_{local_index:03d}.npz',raw50=raw50,network_output50_normalized=network50,postprocessed50=raw50,preprocessed_state=preprocessed_state,source_observation_state=state,mapped_physical=physical,
                                corrected_physical=corrected,arm_projected_before_gripper=arm_projected_before_gripper,legal_commands=np.stack([c['commands'] for c in candidates]) if candidates else np.empty((0,8,7)),
                                sampling_seed=sampling_seed)
            atomic(run.folder/f'boundary_{local_index:03d}.json',{'before_intervention':raw_boundary,'after_intervention':boundary})
            row={'decision_id':decision_id,'observed_time':observation['time'],'sampling_seed':sampling_seed,
                 'VLA_seconds':vla_seconds,'correction':correction_record,'boundary':boundary,
                 'raw_before_intervention_boundary':raw_boundary,'executed':False}
            row['preview_only_gripper_continuation']=gripper_record
            row['preview_only_terminal_hold']=hold_record
            row['candidate_bindings']=[transactions.public_binding(c) for c in candidates]
            # Only live observation/protection vetoes stop the system globally.
            if controller.public_stop or hold_controller.stop_reason:
                row['stop_reason']='public_gripper_guard:'+str(controller.public_stop or hold_controller.stop_reason)
                decisions.append(row);stop_reason=row['stop_reason'];break
            if not candidates:
                row['stop_reason']='no_legal_candidates' ; decisions.append(row); stop_reason=row['stop_reason'];break
            main_candidate=candidates[0]
            if main_candidate['name'] not in ('vla_raw','terminal_hold'):
                row['stop_reason']='phase_main_candidate_unavailable';decisions.append(row);stop_reason=row['stop_reason'];break
            phase_gate.prepare_candidate(main_candidate,decision_id+':phase_event')
            desired_phase=phase_of(main_candidate)
            desired_binding=phase_gate.binding(desired_phase,main_candidate['owners'])
            event_reason=phase_gate.needs_event(desired_binding,float(observation['time']),time.monotonic(),public_health(observation['time']))
            atomic(output/'stage_permission_trace.json',phase_gate.archived())
            if event_reason=='composite_permission_ended':
                row['stop_reason']='composite_permission_ended';decisions.append(row);stop_reason=row['stop_reason'];break
            if event_reason=='public_evidence_invalid':
                row['stop_reason']='stage_permission_public_evidence_invalid';decisions.append(row);stop_reason=row['stop_reason'];break
            row['proposed_control_phase']=desired_phase
            if replan_previous is not None:
                change=recovery_change(replan_previous,candidates)
                row['replan_result']=change
                if not change['changed']:
                    budget.charge('ineffective_replans');same_replan_count+=1
                else:
                    same_replan_count=0
                replan_previous=None
                if same_replan_count>=plan['maximum_ineffective_replans']:
                    row['stop_reason']='replanning_no_change';decisions.append(row);stop_reason=row['stop_reason'];break
            if plan['mode']=='pure_vla':
                if 'vla_raw' not in [c['name'] for c in candidates]:
                    row['stop_reason']='raw_VLA_candidate_rejected';decisions.append(row);stop_reason=row['stop_reason'];break
                choice='vla_raw'
            else:
                images,states,valid,times=history_arrays(run.history)
                if not np.array_equal(states[-1],a.public_motion(run.m,run.d,run.ix)):
                    raise RuntimeError('history terminal encoder state is stale')
                forecasts=[]
                for candidate in candidates:
                    status('online_WM_forecast',decision_id=decision_id,candidate=candidate['name'])
                    budget.charge('WM_forecasts')
                    forecast_start=time.perf_counter()
                    with torch.inference_mode():
                        out=wm(*[torch.from_numpy(x).unsqueeze(0).to('mps') for x in (images,states,valid,candidate['commands'])])
                    if not all(bool(torch.isfinite(x).all()) for x in out.values()):
                        raise RuntimeError('nonfinite WM output')
                    probs=out['events'].sigmoid().detach().cpu().numpy()[0]
                    if np.any(probs[9:18]>probs[:9]+1e-6) or np.any(probs[18:27]>probs[:9]+1e-6):
                        raise RuntimeError('WM contact semantic hierarchy violated')
                    candidate['forecast_seconds']=time.perf_counter()-forecast_start
                    forecasts.append({'name':candidate['name'],'candidate_index':candidate['candidate_index'],
                                      'command_sha256':candidate['command_sha256'],'actions':candidate['commands'].tolist(),
                                      'physical_contacts':dict(zip(CONTACT_TYPES,probs[:9].tolist())),
                                      'new_contacts':dict(zip(CONTACT_TYPES,probs[9:18].tolist())),
                                      'unlocalized_contacts':dict(zip(CONTACT_TYPES,probs[18:27].tolist())),
                                      'any_grasp_estimate':float(probs[27]),'grip_loss_estimate':float(probs[28]),
                                      'motion_delta':out['motion'].detach().cpu().numpy()[0].tolist()})
                phase_gate.confirm_current_WM()
                if plan.get('wm_motion_postprocess')=='joint_fk':
                    from wm_motion_adapter import adapt_forecasts
                    raw_forecasts=copy.deepcopy(forecasts)
                    forecasts,motion_archive=adapt_forecasts(forecasts,states[-1],relation_calibration)
                    atomic(output/'private_WM_motion_postprocess'/f'{decision_id}.json',
                           {'raw_forecasts':raw_forecasts,'adapter':motion_archive,'private_truth_inputs':False})
                public={'decision_id':decision_id,'task':plan['task'],'motor_state':states[-1].tolist(),
                        'history_valid':valid.tolist(),'short_memory':memory.public(),'candidates':forecasts,
                        'image_layout':'Rows -80,-40,0ms; global,wrist. valid=1 genuine RGB; valid=0 zero missing slot, not a captured frame.',
                        'scope':'Frozen online simulation research. Forecast next320ms only; no hardware release or task-truth memory.'}
                if plan.get('interface_revision', 'baseline')=='consistent_v1':
                    public['decision_context']={
                        'interface_revision':'consistent_v1',
                        'stage_control_authority':candidates[0]['control_stage'],
                        'candidate_control_bindings':[transactions.public_binding(c) for c in candidates],
                        'remaining_replan_budget':max(0,budget.limits['replans']-budget.counts['replans']),
                        'presentation_revision':plan.get('presentation_revision','shared_full_hold_v1'),
                        'consecutive_hold_count':no_progress_blocks,
                        'remaining_consecutive_hold_budget':max(0,plan['maximum_consecutive_hold_blocks']-no_progress_blocks),
                        'remaining_observation_budget':max(0,budget.limits['observations']-budget.counts['observations']),
                        'history_source':'captured_rgb_and_joint_encoders',
                        'history_valid_mask':valid.tolist(),
                        'history_capture_times_seconds':[float(t) for t,v in zip(times,valid) if v],
                    }
                check_public(public)
                np.savez_compressed(output/'public_inputs'/f'{decision_id}.npz',history_images=images,history_state=states,
                                    history_valid=valid,history_times_s=times,actions=np.stack([c['commands'] for c in candidates]))
                from PIL import Image
                picture=output/'public_images'/f'{decision_id}.png'
                Image.fromarray(np.concatenate([np.concatenate(list(h),axis=1) for h in images],axis=0)).save(picture)
                row['public']=public
                if event_reason is None:
                    choice=main_candidate['name']
                    row['control_decision_source']='valid_Jev_phase_permission'
                    row['permission_id']=phase_gate.permit['id']
                else:
                    if phase_gate.event_count>=plan['maximum_online_stage_events']:
                        row['stop_reason']='phase_event_budget_exhausted';decisions.append(row);stop_reason=row['stop_reason'];break
                    event={'event_id':decision_id+':phase_event','reason':event_reason,'binding':desired_binding,
                           'public_time_s':float(observation['time']),'main_action_name':main_candidate['name'],
                           'observed_authority':'terminal_hold' if hold_controller.phase=='holding' else 'policy_with_guarded_gripper',
                           'observed_auxiliary_state':{'gripper':controller.phase,'terminal_hold':hold_controller.phase},
                           'wait_action_name':next((c['name'] for c in candidates if c['name'] in ('command_hold','pose_hold')),None)}
                    if desired_phase=='grasp_lift_with_internal_handoff':
                        from composite_permission import CONTRACT
                        event['explicit_composite_contract']=copy.deepcopy(CONTRACT)
                        event['wait_action_name']=None
                    if desired_phase=='terminal_hold' and hold_controller.phase=='holding':event['wait_action_name']='terminal_hold'
                    phase_gate.event(event['event_id'],desired_binding,float(observation['time']),time.monotonic(),event_reason)
                    atomic(output/'stage_permission_trace.json',phase_gate.archived())
                    budget.charge('Jev_calls',2)
                    status('online_Jev_phase_event',decision_id=decision_id)
                    service_picture=ipc_root/'public_images'/f'{decision_id}.png'
                    if service_picture!=picture:service_picture.write_bytes(picture.read_bytes())
                    atomic(ipc_root/f'ipc/job_{ipc_index:03d}.json',{'public':public,'image':str(service_picture.relative_to(ipc_root)),
                        'image_sha256':sha(service_picture),'mode':'phase_event','phase_event':event,
                        'task_relative_observation':observation.get('task_relative_observation')})
                    answerfile=ipc_root/f'ipc/answer_{ipc_index:03d}.json'
                    deadline=time.monotonic()+plan['Jev_response_timeout_seconds']
                    paused_start=time.monotonic();physical_before=float(run.d.time)
                    while not answerfile.exists():
                        wall()
                        if service.poll() is not None:
                            phase_gate.revoke('Jev_service_terminated',physical_before,time.monotonic());raise RuntimeError('Jev_service_terminated')
                        if time.monotonic()>deadline:
                            phase_gate.revoke('Jev_response_timeout',physical_before,time.monotonic());raise RuntimeError('Jev_response_timeout')
                        time.sleep(.1)
                    answer=json.loads(answerfile.read_text());ipc_index+=1
                    if answer['decision_id']!=decision_id or answer['real_Jev_calls']!=2:
                        phase_gate.revoke('invalid_Jev_response',physical_before,time.monotonic());raise RuntimeError('Jev IPC response mismatch')
                    if float(run.d.time)!=physical_before:raise RuntimeError('physics_advanced_during_supervision_wait')
                    row['simulation_pause']={'physical_time_s':physical_before,'wall_s':time.monotonic()-paused_start,'simulation_only_not_hardware_stop':True}
                    row['answers']=answer['answers'];row['supervision_event']=event
                    last_target_answer=answer['answers']['target'];memory.target_proposal(last_target_answer,decision_id)
                    decision=answer['answers']['supervision']['choice']
                    memory.prior_decisions[-1]['choice']='phase_decision:'+decision
                    row['control_decision_source']='Jev_phase_event'
                    health=public_health(observation['time'])
                    if decision=='approve_phase' and last_target_answer['choice']!='red':health=dict(health,target_history_valid=False)
                    granted=phase_gate.decide(event['event_id'],decision,desired_binding,float(observation['time']),time.monotonic(),health)
                    if decision=='approve_phase':
                        if not granted:
                            row['stop_reason']='phase_approval_inapplicable';decisions.append(row);stop_reason=row['stop_reason'];break
                        choice=main_candidate['name'];row['permission_id']=phase_gate.permit['id']
                    elif decision=='wait_observe':
                        if event['wait_action_name'] is None:
                            row['stop_reason']='Jev_wait_has_no_legal_control';decisions.append(row);stop_reason=row['stop_reason'];break
                        wait=lookup_candidate(candidates,event['wait_action_name'])
                        if budget.counts['observations']>=budget.limits['observations']:
                            row['stop_reason']='observation_budget_exhausted';decisions.append(row);stop_reason=row['stop_reason'];break
                        budget.charge('observations')
                        wb=phase_gate.binding(phase_of(wait),wait['owners'])
                        if not phase_gate.authorize_wait(event['event_id'],wb,wait['action_id'],wait['command_sha256'],float(observation['time']),time.monotonic(),health):
                            row['stop_reason']='Jev_wait_permission_invalid';decisions.append(row);stop_reason=row['stop_reason'];break
                        choice=wait['name'];row['permission_id']=phase_gate.permit['id'];row['bounded_observation_requested']=True
                    elif decision=='replan':choice='replan'
                    elif decision=='terminate':choice='reject'
                    else:raise RuntimeError('invalid_native_phase_decision')
                    atomic(output/'stage_permission_trace.json',phase_gate.archived())
            row['choice']=choice
            if choice=='reject':
                row['stop_reason']='Jev_reject';decisions.append(row);stop_reason=row['stop_reason'];break
            if choice=='replan':
                budget.charge('replans')
                replan_previous=candidates
                sampling_seed=plan['sampling_seed']+budget.counts['replans']
                row['recovery_receipt']={'kind':'replan','next_sampling_seed':sampling_seed,'physics_steps':0,
                                         'not_task_progress':True,'candidate_change_pending':True,
                                         'not_independent_motion_planner':True,
                                         'original_rejection_context':row.get('answers'),
                                         'constraint_handling':'same_task_and_public_state; prior_jev_decision_retained; VLA_has_no_extra_constraint_input'}
                run.capture_now()  # Actual observation update even though this choice performs no motion.
                decisions.append(row);atomic(output/'decision_trace.json',decisions);continue
            if choice=='observe':
                budget.charge('observations')
                if 'command_hold' not in [c['name'] for c in candidates]:
                    row['stop_reason']='observation_has_no_legal_hold';decisions.append(row);stop_reason=row['stop_reason'];break
                selected=lookup_candidate(candidates,'command_hold')
                row['recovery_receipt']={'kind':'observe','real_physics_hold':True,'not_active_viewpoint_motion':True}
            else:
                selected=lookup_candidate(candidates,choice)
            commands=selected['commands']
            row['selected_candidate_index']=selected['candidate_index'];row['command_sha256']=selected['command_sha256']
            status('online_execute',decision_id=decision_id,choice=choice)
            permit_binding=phase_gate.binding(phase_of(selected),selected['owners'])
            row['permission_id']=phase_gate.begin(selected['action_id'],selected['command_sha256'],permit_binding,float(observation['time']),time.monotonic(),public_health(observation['time']))
            transactions.activate(selected)
            row['selected_binding']=transactions.public_binding(selected)
            start_rows=len(run.rows)
            try:
                ok,before,before_time,actual_steps=run.execute_logged(commands,block,'observe' if choice=='observe' else 'selected_action')
            finally:
                actual_steps=len(run.rows)-start_rows
                row['executed']=actual_steps>0;row['physics_steps']=actual_steps
                row['start_time']=observation['time'];row['end_time']=float(run.d.time)
                row['transaction_receipt']=transactions.settle(decision_id+':execution',selected['action_id'],commands,actual_steps,float(run.d.time))
                row['permission_receipt']=phase_gate.receipt(decision_id+':execution',selected['action_id'],selected['command_sha256'],int(actual_steps),float(run.d.time))
                phase_gate.check_real_cursor(controller.cursor)
                atomic(output/'stage_permission_trace.json',phase_gate.archived())
                atomic(output/'action_transaction_trace.json',transactions.events)
                atomic(output/'terminal_hold_trace.json',hold_controller.archived())
                atomic(output/'gripper_continuation_trace.json',controller.archived())
                if plan['mode']=='fusion' and private_rows:
                    atomic(output/'private_WM_alignment'/f'{decision_id}.json',{'forecast':forecasts[selected['candidate_index']],
                        'actual':private_rows[-1],'observation_time':observation['time'],
                        'evaluation_only':True,'not_branch_final_label':True,
                        'unobserved_suffix':'unknown' if actual_steps<320 else None})
            row['executed']=actual_steps>0;row['physics_steps']=actual_steps;row['start_time']=before_time;row['end_time']=float(run.d.time)
            row['private_block_event_index']=len(private_rows)-1
            # Align this selected forecast ONLY with its own actually executed 320ms block.
            if plan['mode']=='fusion':
                atomic(output/'private_WM_alignment'/f'{decision_id}.json',{'forecast':forecasts[selected['candidate_index']],
                            'actual':private_rows[-1],'observation_time':observation['time'],
                            'evaluation_only':True,'not_branch_final_label':True})
            memory.receipt(decision_id,selected['name'],before,run.history[-1]['state'],before_time,
                           float(run.d.time),actual_steps,selected['command_sha256'])
            no_progress_blocks=no_progress_blocks+1 if selected['name'] in ('pose_hold','command_hold') and selected['control_stage']!='terminal_hold' else 0
            block+=1
            sampling_seed=plan['sampling_seed']
            if not ok:
                row['stop_reason']=run.stop_reason;stop_reason=run.stop_reason
            elif no_progress_blocks>=plan['maximum_consecutive_hold_blocks']:
                row['stop_reason']='hold_stall_budget';stop_reason=row['stop_reason']
            decisions.append(row);atomic(output/'decision_trace.json',decisions)
            if row.get('stop_reason'):
                break
        phase_gate.revoke('finite_simulation_attempt_ended',float(run.d.time),time.monotonic())
        atomic(output/'stage_permission_trace.json',phase_gate.archived())
        result,trace=run.finish()
        result['absolute_start_time_s']=float(run.initial_state[0]);result['absolute_end_time_s']=float(run.d.time)
        result['local_duration_s']=float(run.d.time)-float(run.initial_state[0]);result.pop('simulated_duration_seconds',None)
        result['stop_reason']=stop_reason if run.stop_reason is None else run.stop_reason
        result['control_mode']=plan['mode'];result['action_correction']=plan['action_correction']
        result['pure_unassisted_VLA']=plan['mode']=='pure_vla' and plan['action_correction']=='none' and controller is None
        result['gripper_execution_aid']=plan.get('gripper_execution_aid')
        result['terminal_hold_aid']=plan.get('terminal_hold_aid')
        if hold_controller is not None:
            atomic(output/'terminal_hold_trace.json',hold_controller.archived())
        result['intervention_is_external_execution_assistance']=plan['action_correction']!='none'
        result['system_completion']=memory.public()['system_completion']
        result['task_truth_evaluator_only']=True
        result['development_task_outcome']={
            'independent_acceptance_rate':None,
            'strict_minimum_grasp_reached':bool(result['strict_L2_80ms']),
            'terminated_by_stall':result['stop_reason']=='hold_stall_budget',
            'counts_as_system_task_attempt':True,
            'full_sort_task_completion_established':False,
            'reason':'development_red_grasp_scope; no full sorting evaluator',
        }
        atomic(output/'result.json',result);atomic(output/'decision_trace.json',decisions)
        atomic(output/'public_memory_final.json',memory.public())
        vafter=snapshot(vla);atomic(output/'VLA_parameters_after.json',vafter)
        if vafter!=vbefore:
            raise RuntimeError('VLA parameters changed')
        if wm is not None:
            wafter=snapshot(wm);atomic(output/'WM_parameters_after.json',wafter)
            if wbefore!=wafter:
                raise RuntimeError('WM parameters changed')
        verify_plan(plan)
        atomic(output/'report.json',{'stage':'bounded_online_complete','result':result,'counts':budget.counts,
                                    'model_parameters_unchanged':True,'optimizer_updates':0,'hardware_enabled':False,
                                    'Jev_completed_calls':sum(2 for row in decisions if 'answers' in row),
                                    'budget_counts_are_requests_or_reservations':True,
                                    'wall_seconds':time.monotonic()-start_wall,'autonomous_acceptance_promoted':False})
        status('bounded_online_complete')
    except Exception:
        if phase_gate is not None:
            phase_gate.revoke('software_prediction_or_service_failure',float(controller.last_observation_time or 0),time.monotonic())
            atomic(output/'stage_permission_trace.json',phase_gate.archived())
        if controller is not None:
            controller.stop('software_interruption',float(controller.last_observation_time or 0))
            atomic(output/'gripper_continuation_trace.json',controller.archived())
        stage=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else None
        pending=locals().get('row')
        atomic(output/'failure.json',{'error':traceback.format_exc(),'counts':budget.counts,'hardware_enabled':False,
                                     'last_stage':stage,'pending_decision':pending})
        atomic(output/'decision_trace.json',decisions)
        if run is not None and not (run.folder/'trajectory.npz').exists():
            try:
                run.finish()
            except Exception:
                atomic(output/'secondary_archive_error.json',{'error':traceback.format_exc()})
        raise
    finally:
        if service is not None and shared is None:
            atomic(output/'ipc/stop.json',{'requested':True})
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.terminate()
                try:
                    service.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    service.kill();service.wait(timeout=10)
            atomic(output/'Jev_service_exit.json',{'exit_code':service.returncode,
                   'clean_parameter_snapshot_available':(output/'Jev_parameters_after.json').exists()})


if __name__=='__main__':
    main()
