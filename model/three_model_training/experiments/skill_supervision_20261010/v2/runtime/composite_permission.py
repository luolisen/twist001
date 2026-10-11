"""Uninstalled, simulation-only permission prototype. No model/physics imports."""
import copy,json,hashlib

CONTRACT={
    'revision':'composite_grasp_lift_v1_pending_confirmation',
    'phase':'grasp_lift_with_internal_handoff',
    'phase_name_is_not_grasp_confirmation':True,
    'maximum_blocks':16,'maximum_physics_s':5.120,'maximum_wall_s':180,
    'maximum_pre_handoff_blocks':8,'plan_commands':50,
    'allowed_sequence':['VLA_arm_plus_cached_plan','VLA_arm_plus_finite_last_plan_target_tail','VLA_seven_channels_nonopening'],
    'handoff_requires_actual_exhaustion_receipt':True,
    'no_clock_or_private_truth_trigger':True,'maximum_plan_activations':1,
    'wait_observe_available_in_this_phase':False,
    'native_choices':['approve_phase','replan','terminate'],
    'terminal_hold_requires_separate_permission':True,
    'existing_guards_and_public_handoff_nonopening_rule_unchanged':True,
    'remaining_VLA_blocks':'16 minus actually consumed composite blocks; not another 16',
    'maximum_granted_composite_instances_per_task':1,
    'expiry_exit':'revoke and end this branch; no compound renewal in the proposed trial',
}

def contract_digest():
    return hashlib.sha256(json.dumps(CONTRACT,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class CompositePermission:
    def __init__(self,binding):
        if set(binding)!={'task_sha256','target_track_token','plan_sha256','plan_length','event_id','contract_sha256'} or binding['plan_length']!=50 or binding['contract_sha256']!=contract_digest():
            raise ValueError('unsupported_explicit_plan_binding')
        self.binding=copy.deepcopy(binding);self.status='awaiting_Jev';self.cursor=0
        self.blocks=0;self.pre_handoff_blocks=0;self.segment='cached';self.inflight=None
        self.receipts={};self.exhaustion_receipt=None;self.events=[]
    def log(self,kind,**kw):self.events.append({'kind':kind,**copy.deepcopy(kw)})
    def revoke(self,reason):
        self.status='revoked';self.log('revoked',reason=reason)
    def decide(self,event_id,choice,t,wall):
        if self.status!='awaiting_Jev' or event_id!=self.binding['event_id']:raise ValueError('stale_or_repeated_permission_decision')
        if choice not in CONTRACT['native_choices']:
            self.revoke('invalid_or_unexecutable_decision');raise ValueError('no_unverified_wait_or_default_VLA')
        if choice!='approve_phase':self.status='not_approved';self.log('not_approved',choice=choice);return False
        self.status='approved';self.expires_physics=t+CONTRACT['maximum_physics_s'];self.expires_wall=wall+CONTRACT['maximum_wall_s']
        self.log('permission_granted',binding=self.binding,contract=CONTRACT,issued_time_s=t)
        return True
    def check_current(self,t,wall,health,phase=CONTRACT['phase'],reserve_block=False):
        bad=None
        if self.status!='approved':bad='permission_not_active'
        elif phase!=CONTRACT['phase']:bad='new_task_phase_requires_new_permission'
        elif not (health.get('input_valid') is True and health.get('target_history_valid') is True and health.get('association_conflict') is False and health.get('WM_valid') is True and not health.get('protection')):bad='public_prediction_or_original_protection_invalid'
        elif self.blocks>=16 or t+(.320 if reserve_block else 0)>self.expires_physics+1e-9 or wall>=self.expires_wall:bad='finite_permission_expired'
        if bad:self.revoke(bad);raise RuntimeError(bad)
    def begin(self,proposal,t,wall,health):
        self.check_current(t,wall,health,reserve_block=True)
        if self.inflight is not None:self.revoke('overlapping_execution');raise RuntimeError('overlapping_execution')
        if proposal['binding']!=self.binding or proposal['cursor_before']!=self.cursor:
            self.revoke('task_target_plan_or_cursor_binding_changed');raise RuntimeError('binding_changed')
        mode=proposal['mode'];take=proposal.get('take',0);tail=proposal.get('tail',0)
        owners=proposal['owners']
        ok=False
        if mode=='consume_active_plan':
            ok=(self.segment=='cached' and self.cursor<50 and self.pre_handoff_blocks<8
                and take==min(8,50-self.cursor) and tail==8-take and owners==['VLA']*6+['gripper_plan'])
        elif mode=='bounded_exhaustion_handoff':
            ok=(self.segment=='cached' and self.cursor==50 and self.exhaustion_receipt is not None
                and self.pre_handoff_blocks<8 and proposal.get('existing_handoff_rule_passed') is True
                and take==0 and owners==['VLA']*6+['last_registered_plan_target'])
        elif mode=='public_nonopening_fresh_policy_handoff':
            ok=(self.cursor==50 and self.exhaustion_receipt is not None and proposal.get('existing_handoff_rule_passed') is True
                and take==0 and owners==['VLA']*7)
        if not ok:
            self.revoke('unapproved_control_source_or_transition');raise RuntimeError('unapproved_control_source_or_transition')
        self.inflight={'proposal':copy.deepcopy(proposal),'start_time_s':t}
        self.log('execution_begin',action_id=proposal['action_id'],command_sha256=proposal['command_sha256'],mode=mode,cursor=self.cursor,
            per_command_clamp_sources=(['cached_plan']*take+['last_registered_plan_target']*tail if mode=='consume_active_plan'
                                       else ['last_registered_plan_target' if mode=='bounded_exhaustion_handoff' else 'VLA']*8))
    def receipt(self,receipt_id,action_id,digest,steps,end):
        facts=(action_id,digest,steps,end)
        if receipt_id in self.receipts:
            if self.receipts[receipt_id]!=facts:raise RuntimeError('conflicting_duplicate_receipt')
            return False
        pending=self.inflight
        if pending is None:raise RuntimeError('receipt_without_actual_selection')
        p=pending['proposal']
        if action_id!=p['action_id'] or digest!=p['command_sha256'] or type(steps)!=int or not 0<=steps<=320 or abs(end-pending['start_time_s']-steps*.001)>1e-7:
            self.revoke('receipt_not_bound_to_executed_prefix');raise RuntimeError('invalid_receipt')
        before=self.cursor
        if steps:
            self.blocks+=1
            if p['mode']=='consume_active_plan':
                self.pre_handoff_blocks+=1;self.cursor+=min(steps//40,p['take'])
                if self.cursor==50:self.exhaustion_receipt=receipt_id
            elif p['mode']=='bounded_exhaustion_handoff':self.pre_handoff_blocks+=1
            elif p['mode']=='public_nonopening_fresh_policy_handoff':self.segment='VLA_handoff'
        self.receipts[receipt_id]=facts;self.inflight=None
        self.log('actual_receipt',receipt_id=receipt_id,physics_steps=steps,cursor_before=before,cursor_after=self.cursor,
            complete_commands=steps//40,partial_command_steps=steps%40,blocks_used=self.blocks)
        if 0<steps<320:self.revoke('partial_execution_ends_branch_no_implicit_resume')
        return True
    def archived(self):
        return {'status':self.status,'binding':copy.deepcopy(self.binding),'cursor':self.cursor,'blocks':self.blocks,
                'segment':self.segment,'exhaustion_receipt':self.exhaustion_receipt,'events':copy.deepcopy(self.events),
                'installed_online':False,'not_new_robot_capability':True}
