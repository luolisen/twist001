"""Candidate-local action state; observations and protection stay on the live controllers."""
from copy import deepcopy
import numpy as np
from fusion_core import command_digest
from candidate_boundary import diagnose_candidates

GRIP_FIELDS = ('phase','plan','cursor','active_started','exhausted_at','last_command','pending','started_count')
HOLD_FIELDS = ('phase','started','reference','previous_control','cursor')

def fields(obj, names):
    return {k:deepcopy(getattr(obj,k)) for k in names}

def install(obj, values):
    for k,v in values.items():setattr(obj,k,deepcopy(v))

class ActionTransaction:
    def __init__(self, gripper, hold, offer_smoothed=True):
        self.g=gripper; self.h=hold; self.version=0; self.current=None
        self.receipts={}; self.events=[]
        self.offer_smoothed=bool(offer_smoothed)

    def preview(self, raw50, projected, observation, source, limits, actuator_names, decision_id):
        """All candidates share the same observed version; only the selected action can activate."""
        if self.current is not None:raise RuntimeError('unsettled_execution_transaction')
        g,h=deepcopy(self.g),deepcopy(self.h)
        gn,hn=len(g.events),len(h.events)
        main=np.asarray(projected,dtype=np.float32).copy();gr=None
        if h.phase!='holding':main,gr=g.propose(raw50,main,observation,source)
        main,hr=h.propose(main,observation,gr)
        candidates=[];filtered=[]
        # A public/phase veto is local to the proposal. Actual observation guards remain global.
        if gr and gr.get('terminate') or hr.get('terminate'):
            filtered.append({'name':'vla_raw','reason':(gr or {}).get('terminate') or hr.get('terminate')})
        else:
            candidates.append({'name':'terminal_hold' if h.phase=='holding' else 'vla_raw',
                'commands':main,'g':g,'h':h,'g_events':g.events[gn:],'h_events':h.events[hn:],
                'gripper_record':gr,'hold_record':hr,
                'owners':['terminal_hold']*7 if h.phase=='holding' else ['VLA']*6+['gripper_plan' if g.pending and g.pending['take'] else 'VLA'],
                'consume_gripper':bool(g.pending and g.pending['take']), 'consume_hold':h.phase=='holding'})
        # Once terminal ownership is active, ordinary holds cannot replace it.
        if h.phase!='holding':
            if candidates and self.offer_smoothed:
                smooth=main.copy();prev=np.asarray(observation['state'][:6],dtype=np.float32).copy()
                for j in range(8):
                    prev=(prev+np.clip(main[j,:6]-prev,-.02,.02)).astype(np.float32);smooth[j,:6]=prev
                c=deepcopy(candidates[0]);c.update(name='vla_smoothed',commands=smooth)
                c['owners']=['VLA_arm_slew']*6+c['owners'][6:];candidates.append(c)
            elif candidates:
                filtered.append({'name':'vla_smoothed','reason':'excluded_registered_action_set_ablation'})
            # Holds preserve the *applied* gap target, never substitute measured gap.
            for name,arm in [('pose_hold',observation['state'][:6]),('command_hold',observation['state'][14:20])]:
                cmd=np.repeat(np.r_[arm,observation['state'][20]].astype(np.float32)[None],8,axis=0)
                candidates.append({'name':name,'commands':cmd,'g':deepcopy(self.g),'h':deepcopy(self.h),
                    'g_events':[],'h_events':[],'gripper_record':None,'hold_record':None,
                    'owners':[name]*6+['last_applied_clamp_target'], 'consume_gripper':False,'consume_hold':False})
        result=[];hashes={}
        for c in candidates:
            diag=diagnose_candidates({c['name']:c['commands']},limits,actuator_names)
            if c['name'] not in diag['retained_candidate_names']:
                filtered.append({'name':c['name'],'reason':'numeric_or_limit_rejection','diagnostic':diag});continue
            digest=command_digest(c['commands'])
            if digest in hashes:
                filtered.append({'name':c['name'],'reason':'identical_commands_alias','canonical':hashes[digest]});continue
            hashes[digest]=c['name'];c.update(command_sha256=digest,candidate_index=len(result),
                action_id=decision_id+':'+c['name'],state_version=self.version+1,
                observation_time_s=float(observation['time']),control_stage='terminal_hold' if c['consume_hold'] else 'policy_with_guarded_gripper')
            result.append(c)
            c['invalidates_closure_history']=bool(c['h'].phase=='waiting' and np.any(c['commands'][:,6]>float(observation['state'][6])))
        return result,{'retained_candidate_names':[c['name'] for c in result],'filtered':filtered,
                       'state_version':self.version+1,'observation_time_s':float(observation['time'])},gr,hr

    def public_binding(self,c):
        return {k:c[k] for k in ('action_id','state_version','observation_time_s','command_sha256','owners','control_stage','consume_gripper','consume_hold')}

    def activate(self,c):
        if self.current is not None or c['state_version']!=self.version+1:raise RuntimeError('stale_or_overlapping_candidate')
        if self.g.public_stop or self.h.stop_reason or abs(self.g.last_observation_time-c['observation_time_s'])>1e-9:
            raise RuntimeError('live_protection_or_observation_changed_before_selection')
        if command_digest(c['commands'])!=c['command_sha256']:raise RuntimeError('candidate_commands_changed')
        self.current={'c':c,'g_before':fields(self.g,GRIP_FIELDS),'h_before':fields(self.h,HOLD_FIELDS)}
        install(self.g,fields(c['g'],GRIP_FIELDS));install(self.h,fields(c['h'],HOLD_FIELDS))
        self.g.events.extend(deepcopy(c['g_events']));self.h.events.extend(deepcopy(c['h_events']))
        # Effective opening control invalidates closure history; unselected predictions do not.
        if c['invalidates_closure_history']:
            self.h.invalidate_response('selected_effective_control_incompatible',c['observation_time_s'])
        self.events.append({'kind':'selected_action_activated_not_yet_consumed',**self.public_binding(c)})

    def settle(self,receipt_id,action_id,commands,steps,end_time):
        digest=command_digest(commands)
        facts=(action_id,digest,int(steps),float(end_time))
        if receipt_id in self.receipts:
            if self.receipts[receipt_id]['facts']!=facts:raise RuntimeError('conflicting_duplicate_receipt')
            return dict(self.receipts[receipt_id]['receipt'],duplicate=True)
        if self.current is None:raise RuntimeError('receipt_without_selected_action')
        c=self.current['c']
        if action_id!=c['action_id'] or digest!=c['command_sha256'] or not 0<=steps<=320:
            raise RuntimeError('receipt_action_or_prefix_mismatch')
        grip=None
        if steps==0:
            # Keep live observations, validity/expiry, and protection reasons; roll back action fields only.
            install(self.g,self.current['g_before']);install(self.h,self.current['h_before'])
        else:
            if c['consume_gripper']:grip=self.g.commit(steps,end_time)
            elif self.g.pending is not None:self.g.pending=None
            if c['consume_hold']:self.h.commit(steps,end_time)
        receipt={**self.public_binding(c),'receipt_id':receipt_id,'physics_steps':int(steps),
                 'fully_executed_commands':int(steps)//40,'partial_command_steps':int(steps)%40,
                 'end_time_s':float(end_time),'gripper_receipt':grip,'duplicate':False,
                 'partial_prefix_terminal_no_implicit_replay':bool(steps%40)}
        self.receipts[receipt_id]={'facts':facts,'receipt':receipt};self.events.append({'kind':'selected_prefix_committed',**receipt})
        self.current=None;self.version+=1;return receipt
