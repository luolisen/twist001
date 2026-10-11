"""Finite, revocable simulation phase permissions. No model or private truth input."""
import copy,hashlib,json,math

PHASE_BLOCKS={'policy_pre_plan':16,'gripper_continuation':8,'policy_handoff':16,'terminal_hold':6}
def phase_of(c):
    if c['consume_hold']:return 'terminal_hold'
    if c['g'].phase=='active':return 'gripper_continuation'
    if c['g'].phase in ('exhausted','handed_off'):return 'policy_handoff'
    if c['g'].phase=='idle':return 'policy_pre_plan'
    raise ValueError('unregistered_auxiliary_phase')
def task_digest(task):return hashlib.sha256(task.encode()).hexdigest()

class PhaseGate:
    def __init__(self,task,track_token,max_events=5,wall_ttl=180):
        self.task=task_digest(task);self.track=track_token;self.max_events=max_events
        self.wall_ttl=wall_ttl;self.permit=None;self.events=[];self.event_count=0
        self.inflight=None;self.receipts={};self.serial=0;self.pending_event=None;self.event_ids=set();self.last_decision=None
    def log(self,kind,**kw):self.events.append({'kind':kind,**copy.deepcopy(kw)})
    def revoke(self,reason,t,wall):
        if self.permit is not None:
            self.log('permission_revoked',reason=reason,permission_id=self.permit['id'],time_s=t,wall=wall)
        self.permit=None
    def binding(self,phase,owners):
        if phase not in PHASE_BLOCKS:raise ValueError('unregistered_permission_phase')
        if phase=='terminal_hold' and owners!=['terminal_hold']*7:raise ValueError('terminal_ownership_mismatch')
        return {'task_sha256':self.task,'target_track_token':self.track,'phase':phase,'owners':list(owners)}
    def healthy(self,health):
        return (health.get('input_valid') is True and health.get('association_conflict') is False
                and health.get('target_history_valid') is True and not health.get('protection'))
    def needs_event(self,binding,t,wall,health):
        if not self.healthy(health):
            self.revoke('public_input_association_or_protection',t,wall);return 'public_evidence_invalid'
        p=self.permit
        if p is None:return 'task_start_or_permission_revoked'
        if p['binding']!=binding:self.revoke('phase_or_control_binding_changed',t,wall);return 'phase_or_control_transition'
        if p['used_blocks']>=p['max_blocks'] or t+.320>p['expires_physics_s']+1e-9 or wall>=p['expires_wall']:
            self.revoke('finite_permission_expired',t,wall);return 'permission_expired'
        return None
    def event(self,event_id,binding,t,wall,reason):
        if self.event_count>=self.max_events:raise RuntimeError('phase_event_budget_exhausted')
        if event_id in self.event_ids or self.pending_event is not None:raise RuntimeError('repeated_or_overlapping_supervision_event')
        self.event_ids.add(event_id);self.pending_event={'id':event_id,'binding':copy.deepcopy(binding),'time_s':t}
        self.event_count+=1;self.log('supervision_event',event_id=event_id,binding=binding,time_s=t,wall=wall,reason=reason)
    def decide(self,event_id,choice,binding,t,wall,health):
        if self.pending_event!={'id':event_id,'binding':binding,'time_s':t}:raise ValueError('decision_not_bound_to_current_event_observation')
        if binding.get('task_sha256')!=self.task or binding.get('target_track_token')!=self.track:
            raise ValueError('permission_task_or_target_binding_mismatch')
        if choice not in ('approve_phase','wait_observe','replan','terminate'):raise ValueError('invalid_phase_choice')
        self.pending_event=None;self.last_decision=(event_id,choice)
        if choice!='approve_phase':
            self.revoke('Jev_'+str(choice),t,wall);self.log('permission_not_granted',event_id=event_id,choice=choice,time_s=t);return False
        if not self.healthy(health):
            self.revoke('approval_cannot_override_invalid_public_evidence_or_protection',t,wall)
            self.log('approval_inapplicable',event_id=event_id);return False
        self.serial+=1;limit=PHASE_BLOCKS[binding['phase']]
        self.permit={'id':f'permit_{self.serial:03d}','event_id':event_id,'binding':copy.deepcopy(binding),
                     'max_blocks':limit,'used_blocks':0,'issued_physics_s':t,'expires_physics_s':t+limit*.320,
                     'issued_wall':wall,'expires_wall':wall+self.wall_ttl}
        self.log('permission_granted',**self.permit);return True
    def authorize_wait(self,event_id,binding,action_id,digest,t,wall,health):
        if self.last_decision!=(event_id,'wait_observe') or not self.healthy(health):return False
        self.serial+=1
        self.permit={'id':f'permit_{self.serial:03d}','event_id':event_id,'binding':copy.deepcopy(binding),
                     'max_blocks':1,'used_blocks':0,'issued_physics_s':t,'expires_physics_s':t+.320,
                     'issued_wall':wall,'expires_wall':wall+self.wall_ttl,'purpose':'one_Jev_requested_observation',
                     'only_action_id':action_id,'only_digest':digest}
        self.log('one_wait_control_permission',**self.permit);return True
    def begin(self,action_id,digest,binding,t,wall,health):
        reason=self.needs_event(binding,t,wall,health)
        if reason:raise RuntimeError('execution_permission_denied:'+reason)
        if self.inflight is not None:raise RuntimeError('permission_execution_overlap')
        if 'only_digest' in self.permit and (action_id!=self.permit['only_action_id'] or digest!=self.permit['only_digest']):
            raise RuntimeError('supervised_wait_cannot_authorize_VLA')
        self.inflight={'action_id':action_id,'digest':digest,'binding':copy.deepcopy(binding),
                       'permission_id':self.permit['id'],'start_time_s':t}
        self.log('permission_execution_begin',**self.inflight)
        return self.permit['id']
    def check_live(self,t,wall,health):
        p=self.permit
        reason=None
        if not self.healthy(health):reason='public_evidence_or_protection_changed'
        elif p is None:reason='permission_missing'
        elif t>p['expires_physics_s']+1e-9 or wall>=p['expires_wall']:reason='permission_expired_during_execution'
        if reason:self.revoke(reason,t,wall);raise RuntimeError(reason)
    def receipt(self,receipt_id,action_id,digest,steps,end):
        facts=(action_id,digest,int(steps),float(end))
        if receipt_id in self.receipts:
            if self.receipts[receipt_id]!=facts:raise RuntimeError('conflicting_permission_receipt')
            return {'duplicate':True,'physics_steps':steps}
        f=self.inflight
        if f is None or action_id!=f['action_id'] or digest!=f['digest'] or type(steps)!=int or not 0<=steps<=320:
            raise RuntimeError('permission_receipt_not_bound_to_execution')
        if abs(end-f['start_time_s']-steps*.001)>1e-7:raise RuntimeError('permission_receipt_time_mismatch')
        if self.permit is not None and self.permit['id']==f['permission_id'] and steps:
            self.permit['used_blocks']+=1
        self.receipts[receipt_id]=facts;self.log('permission_execution_receipt',**f,receipt_id=receipt_id,
              physics_steps=steps,end_time_s=end,full_commands=steps//40,partial_command_steps=steps%40)
        self.inflight=None
        return {'duplicate':False,'permission_id':f['permission_id'],'physics_steps':steps}
    def archived(self):
        return {'permit':copy.deepcopy(self.permit),'event_count':self.event_count,'maximum_events':self.max_events,
                'events':copy.deepcopy(self.events),'inflight':copy.deepcopy(self.inflight),'private_inputs':False}
