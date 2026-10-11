"""Finite skill leases. Public facts and executed prefixes, never task truth."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib, json
import numpy as np
from evidence_interfaces import valid_public_evidence,SUPPORT_KIND

@dataclass(frozen=True)
class SkillSpec:
    name: str
    generator: str
    owners: tuple
    max_blocks: int
    max_physics_s: float
    max_wall_s: float = 180.
    alternative_owners: tuple = ()

SPECS = {
    'Pick': SkillSpec('Pick','frozen_pick_v1',('VLA',)*7,34,10.88,180.,(('VLA',)*6+('gripper_plan',),)),
    'Hold': SkillSpec('Hold','terminal_hold_v1',('terminal_hold',)*7,6,1.92),
    'Transport': SkillSpec('Transport','bounded_ik_transport_v2',('transport_trajectory',)*6+('retained_clamp',),20,6.4),
    'Place': SkillSpec('Place','bounded_place_v2',('place_trajectory',)*6+('place_release_gate',),18,5.76),
    'Recovery': SkillSpec('Recovery','no_motion_recovery_v2',('none',)*7,0,0.),
}
EDGES={None:{'Pick'},'Pick':{'Hold','Recovery'},'Hold':{'Transport','Recovery'},
       'Transport':{'Place','Hold','Recovery'},'Place':{'Hold','Recovery'},'Recovery':set()}
def digest(commands):
    a=np.asarray(commands,dtype=np.float32)
    if a.shape!=(8,7) or not np.isfinite(a).all():raise ValueError('illegal_final_seven_channel_commands')
    return hashlib.sha256(a.tobytes()).hexdigest()

class SkillRouter:
    def __init__(self,task_id,target_token,mode='online',validators=()):
        if mode not in ('online','local_saved_state_diagnostic'):raise ValueError('unknown_mode')
        self.task=task_id;self.target=target_token;self.mode=mode;self.validators=set(validators)
        self.active=None;self.lease=None;self.pending=None;self.inflight=None;self.receipts={};self.events=[];self.version=0
        self.generators={};self.serial=0
    def register(self,key,generator):
        if key in self.generators:raise ValueError('generator_already_registered')
        self.generators[key]=generator
    def log(self,kind,**kw):self.events.append({'kind':kind,**deepcopy(kw)})
    def health(self,c):
        return c.get('input_valid') is True and c.get('target_valid') is True and not c.get('target_conflict') and not c.get('protection')
    def evidence_valid(self,c,key):
        e=c.get(key,{})
        if key=='support_evidence':
            return valid_public_evidence(e,SUPPORT_KIND,self.validators,c.get('observation_time_s'),self.target)
        return (e.get('established') is True and e.get('validator_id') in self.validators
                and e.get('source')=='public_observation_and_execution'
                and e.get('observation_time_s')==c.get('observation_time_s')
                and e.get('target')==self.target)
    def request(self,skill,c,t,wall):
        if self.inflight or self.pending:raise RuntimeError('unsettled_control_or_request')
        if skill not in EDGES[self.active]:raise ValueError('illegal_skill_edge')
        if not self.health(c) or c.get('observation_time_s')!=t:raise RuntimeError('public_health_invalid_or_stale')
        if skill=='Hold' and not self.hold_available(c):raise RuntimeError('no_verified_hold_controller_authority')
        if skill=='Transport' and not self.evidence_valid(c,'carry_evidence'):raise RuntimeError('public_carry_handoff_not_established')
        if skill=='Place' and (not self.evidence_valid(c,'carry_evidence') or not c.get('registered_worksite_reached')):raise RuntimeError('place_entry_not_established')
        spec=SPECS[skill]
        if spec.generator not in self.generators:raise RuntimeError('generator_not_installed')
        self.serial+=1
        self.pending={'id':f'skill_event_{self.serial:03d}','task':self.task,'target':self.target,
                      'previous_skill':self.active,'skill':skill,'version':self.version,'time_s':t,
                      'spec':spec.__dict__,'public_context':deepcopy(c)}
        # Switching requires a new decision; the old skill cannot run while awaiting it.
        self.revoke('new_skill_request');self.log('skill_request',request=self.pending)
        return deepcopy(self.pending)
    def decide(self,event_id,choice,c,t,wall,issuer='Jev_native'):
        if self.pending is None or self.pending['id']!=event_id or self.pending['time_s']!=t:raise RuntimeError('stale_decision')
        if issuer!='Jev_native':raise ValueError('online_permission_requires_native_Jev')
        if choice not in ('approve_phase','wait_observe','replan','terminate'):raise ValueError('invalid_native_choice')
        request=self.pending;self.pending=None
        if choice!='approve_phase' or not self.health(c) or c.get('observation_time_s')!=t or c.get('task',self.task)!=self.task or c.get('target',self.target)!=self.target:
            self.log('permission_not_granted',choice=choice,event_id=event_id);self.revoke('Jev_'+choice);return False
        spec=SPECS[request['skill']]
        # Recheck evidence against the current context, not the old request.
        if request['skill']=='Transport' and not self.evidence_valid(c,'carry_evidence'):raise RuntimeError('handoff_evidence_expired')
        if request['skill']=='Hold' and not self.hold_available(c):raise RuntimeError('hold_authority_expired')
        if request['skill']=='Place' and (not self.evidence_valid(c,'carry_evidence') or not c.get('registered_worksite_reached')):raise RuntimeError('place_entry_expired')
        self.active=request['skill'];self.version+=1
        self.lease={'event_id':event_id,'skill':self.active,'version':self.version,'owners':list(spec.owners),
                    'alternative_owners':[list(x) for x in spec.alternative_owners],
                    'task':self.task,'target':self.target,'issued_s':t,'expires_s':t+spec.max_physics_s,
                    'expires_wall':wall+spec.max_wall_s,'max_blocks':spec.max_blocks,'used_blocks':0,'issuer':issuer}
        self.log('skill_authorized',lease=self.lease);return True
    def hold_available(self,c):
        # Proposal is not yet executed authority; it can support a new authorization
        # only with its exact transaction-bound command identity.
        return (c.get('terminal_controller_active') is True or
                (c.get('terminal_takeover_ready') is True and bool(c.get('terminal_proposal_sha256'))))
    def nominate_local_transport(self,c,t,wall,source_state):
        if self.mode!='local_saved_state_diagnostic' or not source_state or not self.health(c):raise RuntimeError('no_diagnostic_nomination')
        if self.inflight or self.lease:raise RuntimeError('control_overlap')
        spec=SPECS['Transport'];self.active='Transport';self.version+=1
        self.lease={'event_id':'explicit_saved_state_diagnostic','skill':'Transport','version':self.version,
                    'owners':list(spec.owners),'task':self.task,'target':self.target,'issued_s':t,'expires_s':t+spec.max_physics_s,
                    'expires_wall':wall+spec.max_wall_s,'max_blocks':spec.max_blocks,'used_blocks':0,
                    'issuer':'user_authorized_local_assumption','source_state':source_state}
        self.log('offline_carry_assumption_not_online_authorization',lease=self.lease)
    def revoke(self,reason):
        if self.lease:self.log('skill_lease_revoked',reason=reason,lease=self.lease)
        self.lease=None
    def check(self,c,t,wall,reserve=False):
        p=self.lease
        if (not p or not self.health(c) or c.get('observation_time_s')!=t or c.get('task',self.task)!=self.task or c.get('target',self.target)!=self.target
            or (self.mode=='online' and self.active in ('Transport','Place') and not self.evidence_valid(c,'carry_evidence'))
            or t+(.32 if reserve else 0)>p['expires_s']+1e-9 or wall>=p['expires_wall']
            or (reserve and p['used_blocks']>=p['max_blocks'])):
            self.revoke('current_state_binding_or_finite_lease_invalid');raise RuntimeError('skill_execution_denied')
    def begin(self,action_id,commands,owners,c,t,wall,wm_binding):
        self.check(c,t,wall,True)
        if self.inflight:raise RuntimeError('two_control_sources')
        h=digest(commands)
        if not np.isfinite(c.get('last_clamp_target_m',float('nan'))):
            self.revoke('invalid_clamp_context');raise RuntimeError('invalid_clamp_context')
        allowed=[self.lease['owners']]+self.lease.get('alternative_owners',[])
        if list(owners) not in allowed:
            self.revoke('unauthorized_channel_owner');raise RuntimeError('unauthorized_channel_owner')
        if self.mode=='online' and (not wm_binding or wm_binding.get('command_sha256')!=h or wm_binding.get('observation_time_s')!=t or wm_binding.get('legal_forecast') is not True):
            self.revoke('WM_binding_invalid');raise RuntimeError('WM_not_bound_to_actual_commands')
        # Pick's original guards handle its opening/closure plan; only carried-object
        # skills require the clamp to be retained by the outer router.
        if self.active in ('Hold','Transport') and np.any(np.asarray(commands)[:,6]>c['last_clamp_target_m']+1e-9):raise RuntimeError('unapproved_release')
        if self.active=='Place' and np.any(np.asarray(commands)[:,6]>c['last_clamp_target_m']+1e-9) and not self.evidence_valid(c,'support_evidence'):raise RuntimeError('public_pre_release_support_not_established_no_release')
        if self.active=='Hold' and not c.get('terminal_controller_active') and c.get('terminal_proposal_sha256')!=h:raise RuntimeError('terminal_takeover_proposal_changed')
        self.inflight={'id':action_id,'sha256':h,'start_s':t,'lease_version':self.version,'owners':list(owners)}
        self.log('selected_commands_begin',**self.inflight)
    def receipt(self,rid,action_id,commands,steps,end):
        h=digest(commands);facts=(action_id,h,steps,end)
        if rid in self.receipts:
            if self.receipts[rid]!=facts:raise RuntimeError('conflicting_repeat_receipt')
            return False
        f=self.inflight
        if not f or f['id']!=action_id or f['sha256']!=h or type(steps)!=int or not 0<=steps<=320 or abs(end-f['start_s']-steps*.001)>1e-7:raise RuntimeError('wrong_execution_prefix')
        if self.lease and steps:self.lease['used_blocks']+=1
        self.receipts[rid]=facts;self.inflight=None;self.log('actual_prefix_receipt',id=rid,steps=steps,end_s=end,complete_commands=steps//40,partial_command_steps=steps%40)
        if 0<steps<320:self.revoke('partial_prefix_no_implicit_replay')
        return True
    def cancel(self,reason):self.pending=None;self.revoke(reason);self.log('cancel_without_clamp_release',reason=reason)

def public_handoff_evidence(state,guard,hold,receipt):
    s=np.asarray(state)
    valid=bool(s.shape==(21,) and np.isfinite(s).all())
    return {'observed':{'valid_encoders':valid,
                       'target_association_reconfirmed':bool(guard.get('front_track_confirmed') or guard.get('wrist_track_confirmed')),
                       'target_conflict':bool(guard.get('target_correspondence_ambiguous')),
                       'measured_gap_m':float(s[6]) if valid else None,'last_applied_clamp_target_m':float(s[20]) if valid else None,
                       'terminal_controller_active':hold.phase=='holding','actual_receipt':deepcopy(receipt)},
            'forecast':{},'unknown':{'actual_load_retention':None,'target_depth':None,'pre_release_support':None,'post_release_placement':None},
            'carry_evidence':{'established':False,'validator_id':None,'source':'public_observation_and_execution',
                              'reason':'closure_response_and_terminal_hold_are_not_validated_carry_confirmation'}}
