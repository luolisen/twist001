"""Approved simulation scope. The old gate still owns pre-plan and terminal phases."""
import copy
import hashlib
from stage_permission import PhaseGate as OriginalGate, phase_of as original_phase, PHASE_BLOCKS
from composite_permission import CompositePermission, CONTRACT, contract_digest

PHASE=CONTRACT['phase']
PHASE_BLOCKS[PHASE]=CONTRACT['maximum_blocks']

def phase_of(candidate):
    old=original_phase(candidate)
    return PHASE if old in ('gripper_continuation','policy_handoff') else old

class PhaseGate(OriginalGate):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.compound=None; self.compound_granted=False; self.candidate=None
        self.next_event_id=None; self.WM_verified=False

    def prepare_candidate(self,candidate,event_id):
        self.candidate=candidate;self.next_event_id=event_id;self.WM_verified=False

    def confirm_current_WM(self):
        self.WM_verified=True

    def binding(self,phase,owners):
        if phase!=PHASE:return super().binding(phase,owners)
        g=self.candidate['g']
        if g.plan is None or len(g.plan)!=50 or g.started_count!=1:
            raise RuntimeError('unregistered_or_restarted_composite_plan')
        digest=hashlib.sha256(g.plan.tobytes()).hexdigest()
        event_id=(self.compound.binding['event_id'] if self.compound and self.compound.status=='approved'
                  else self.next_event_id)
        return {'task_sha256':self.task,'target_track_token':self.track,'phase':PHASE,
                'owners':CONTRACT['allowed_sequence'], 'plan_sha256':digest,'plan_length':len(g.plan),
                'event_id':event_id,'contract_sha256':contract_digest()}

    def kernel_binding(self,binding):
        return {k:v for k,v in binding.items() if k not in ('phase','owners')}

    def needs_event(self,binding,t,wall,health):
        if binding['phase']!=PHASE:
            if self.compound and self.compound.status=='approved':self.compound.revoke('actual_task_phase_changed')
            return super().needs_event(binding,t,wall,health)
        if not self.healthy(health):
            self.revoke('public_input_association_or_protection',t,wall);return 'public_evidence_invalid'
        if self.compound_granted and (not self.compound or self.compound.status!='approved'):
            self.revoke('composite_not_renewable',t,wall);return 'composite_permission_ended'
        if self.compound and self.compound.status=='approved':
            if self.kernel_binding(binding)!=self.compound.binding:
                self.revoke('task_target_plan_binding_changed',t,wall);return 'composite_permission_ended'
            try:self.compound.check_current(t,wall,dict(health,WM_valid=True),reserve_block=True)
            except RuntimeError:
                self.revoke('finite_composite_permission_ended',t,wall);return 'composite_permission_ended'
        return super().needs_event(binding,t,wall,health)

    def event(self,event_id,binding,t,wall,reason):
        if binding['phase']==PHASE:
            if self.compound_granted:raise RuntimeError('second_composite_grant_forbidden')
            self.compound=CompositePermission(self.kernel_binding(binding))
        return super().event(event_id,binding,t,wall,reason)

    def decide(self,event_id,choice,binding,t,wall,health):
        if binding['phase']==PHASE and choice not in CONTRACT['native_choices']:
            self.revoke('unexecutable_composite_decision',t,wall)
            raise RuntimeError('no_unverified_wait_or_default_VLA')
        granted=super().decide(event_id,choice,binding,t,wall,health)
        if binding['phase']==PHASE:
            if granted:
                self.compound.decide(event_id,choice,t,wall);self.compound_granted=True
            elif self.compound.status=='awaiting_Jev':
                if choice=='approve_phase':self.compound.revoke('approval_inapplicable')
                else:self.compound.decide(event_id,choice,t,wall)
        return granted

    def begin(self,action_id,digest,binding,t,wall,health):
        if binding['phase']==PHASE:
            if self.compound is None or self.compound.status!='approved':
                raise RuntimeError('composite_permission_not_active')
            c=self.candidate;record=c['gripper_record'];mode=record.get('mode')
            if c['name']!='vla_raw' or c['action_id']!=action_id or c['command_sha256']!=digest:
                self.revoke('selected_action_binding_changed',t,wall);raise RuntimeError('composite_selection_mismatch')
            owners=(['VLA']*6+['last_registered_plan_target'] if mode=='bounded_exhaustion_handoff' else c['owners'])
            proposal={'binding':self.kernel_binding(binding),'cursor_before':record['cursor_before'],
                'mode':mode,'take':record.get('planned_consumption',0),
                'tail':record.get('finite_tail_handoff_commands',0),'owners':owners,
                'existing_handoff_rule_passed':record.get('terminate') is None and record.get('release_authorized') is False,
                'action_id':action_id,'command_sha256':digest}
            self.compound.begin(proposal,t,wall,dict(health,WM_valid=self.WM_verified))
        return super().begin(action_id,digest,binding,t,wall,health)

    def check_live(self,t,wall,health):
        if self.inflight and self.inflight['binding']['phase']==PHASE:
            self.compound.check_current(t,wall,dict(health,WM_valid=self.WM_verified))
        return super().check_live(t,wall,health)

    def receipt(self,receipt_id,action_id,digest,steps,end):
        compound_receipt=self.inflight and self.inflight['binding']['phase']==PHASE
        result=super().receipt(receipt_id,action_id,digest,steps,end)
        if compound_receipt:self.compound.receipt(receipt_id,action_id,digest,steps,end)
        return result

    def check_real_cursor(self,cursor):
        if self.compound and self.compound.status=='approved' and cursor!=self.compound.cursor:
            raise RuntimeError('actual_gripper_receipt_and_permission_cursor_mismatch')

    def revoke(self,reason,t,wall):
        if self.compound and self.compound.status=='approved':self.compound.revoke(reason)
        return super().revoke(reason,t,wall)

    def archived(self):
        record=super().archived()
        record['composite']=self.compound.archived() if self.compound else None
        if record['composite'] is not None:record['composite']['installed_online']=True
        record['composite_granted_once']=self.compound_granted
        record['approved_composite_contract']=copy.deepcopy(CONTRACT)
        return record
