"""Public executed-closure response supports a bounded stabilization attempt, not grasp truth."""
import numpy as np

class TerminalHold:
    def __init__(self,rules):
        self.r=rules;self.phase='waiting';self.observations=[];self.events=[];self.previous_t=None
        self.support_since=None;self.latest_ready=False;self.started=None;self.reference=None;self.previous_control=None
        self.cursor=0;self.stop_reason=None;self.cancelled=False
        self.response_samples=[];self.open_witness=None;self.contraction_witness=False
        self.previous_gap=None;self.last_reliable_target=None
    def invalidate_response(self,reason,t):
        self.response_samples=[];self.support_since=None;self.latest_ready=False
        self.open_witness=None;self.contraction_witness=False
        self.events.append({'kind':'closure_history_invalidated','reason':reason,'time_s':float(t)})
    def observe(self,o,gripper,guard):
        t=float(o['time']);s=np.asarray(o['state']);record={'time_s':t,'phase':self.phase}
        delta=None if self.previous_t is None else t-self.previous_t
        duplicate=delta is not None and abs(delta)<1e-9
        if delta is not None and delta < -1e-9:self.stop_reason='hold_nonmonotonic_public_time'
        if not guard['input_valid'] or guard.get('terminate'):
            self.stop_reason=guard.get('terminate') or 'hold_input_invalid'
        timely=delta is None or -1e-9<=delta<=self.r['maximum_response_observation_gap_s']+1e-9
        ambiguous=guard.get('target_correspondence_ambiguous',False)
        reliable=bool(not ambiguous and (guard.get('front_track_confirmed') or guard.get('wrist_track_confirmed')))
        if reliable:self.last_reliable_target=t
        stale=self.last_reliable_target is None or t-self.last_reliable_target>self.r['target_history_max_unconfirmed_s']+1e-9
        if self.stop_reason or not timely or ambiguous or stale:
            self.invalidate_response('invalid_input_time_or_target_correspondence',t)
        wrist=gripper.association.wrist if guard.get('wrist_track_confirmed') else None
        box=self.r['wrist_bbox']
        visual=bool(wrist and not ambiguous and wrist['bbox'][0]>=box[0] and wrist['bbox'][1]>=box[1] and wrist['bbox'][2]<=box[2] and wrist['bbox'][3]<=box[3])
        gap=float(s[6]);applied=float(s[20]);margin=gap-applied;eps=self.r['numeric_gap_tolerance_m']
        closing_side=margin>eps
        # A fresh executed opening/equalized request cannot support continuing clamp-response evidence.
        if not closing_side:
            self.invalidate_response('applied_control_no_longer_on_closing_side',t)
        # Re-establish an actual response origin after a revoked/opening interval;
        # renewed closure may start from a partly closed aperture, not only fully open.
        if reliable and self.open_witness is None:
            self.open_witness={'time_s':t,'gap_m':gap}
        if (not duplicate and timely and reliable and closing_side and self.open_witness is not None
            and self.previous_gap is not None and gap < self.previous_gap-eps
            and gap < self.open_witness['gap_m']-eps):
            self.contraction_witness=True
        gap_supported=self.r['gap_min_m']-eps<=gap<=self.r['gap_max_m']+eps
        eligible=float(np.sqrt(np.mean((np.asarray(self.r['eligible_q_templates'])-s[:6])**2,axis=1)).min())
        earlier=float(np.sqrt(np.mean((np.asarray(self.r['earlier_q_templates'])-s[:6])**2,axis=1)).min())
        stage=eligible<=self.r['q_tolerance_rad'] and eligible<=earlier
        context=bool(visual and closing_side and gap_supported and self.contraction_witness and timely and not stale and not self.stop_reason)
        sample={'time_s':t,'gap_m':gap,'applied_gap_m':applied,'supported_context':context}
        if duplicate:
            if self.response_samples and abs(self.response_samples[-1]['time_s']-t)<1e-9:self.response_samples[-1]=sample
        else:self.response_samples.append(sample)
        self.response_samples=[v for v in self.response_samples if t-v['time_s']<=self.r['response_total_age_max_s']+1e-9]
        window=[v for v in self.response_samples if t-v['time_s']<=self.r['association_history_s']+1e-9]
        history=t-window[0]['time_s'] if window else 0.
        consecutive=all(b['time_s']-a['time_s']<=self.r['maximum_response_observation_gap_s']+1e-9 for a,b in zip(window,window[1:]))
        variation=float(np.ptp([v['gap_m'] for v in window])) if window else None
        response=bool(context and history>=self.r['association_history_s']-1e-9 and consecutive
                      and all(v['supported_context'] for v in window)
                      and variation<=self.r['response_window_gap_range_max_m']+eps)
        ready=bool(response and stage and gripper.phase=='handed_off')
        self.latest_ready=ready
        if self.phase=='holding' and t-self.started>self.r['maximum_hold_s']+1e-9:self.stop_reason='finite_terminal_hold_duration_exhausted'
        self.previous_t=t;self.previous_gap=gap
        record.update(input_valid=guard['input_valid'],tracked_wrist_supported=visual,
            executed_target_on_closing_side=bool(closing_side),executed_contraction_witness=bool(self.contraction_witness),
            fresh_closure_response_supported=response,gap_supported=bool(gap_supported),gap_m=gap,
            applied_command_gap_m=applied,margin_m=margin,response_window_gap_range_m=variation,
            response_window_observed_span_s=history,response_intervals_timely=consecutive,
            eligible_q_distance_rad=eligible,earlier_q_distance_rad=earlier,eligible_phase=bool(stage),
            gripper_phase=gripper.phase,ready_to_attempt=ready,actual_grasp_or_height_confirmation=None,terminate=self.stop_reason)
        self.observations.append(record);return record
    def propose(self,commands,o,control_decision=None):
        t=float(o['time']);s=np.asarray(o['state']);out=np.asarray(commands).copy()
        record={'time_s':t,'phase_before':self.phase,'action_source':'old_policy_and_gripper_continuation','triggered':False}
        if self.stop_reason:record['terminate']=self.stop_reason;return out,record
        if self.phase=='waiting':
            # Respect existing gripper authority before takeover; ignored archival VLA predictions after takeover are not releases.
            current_valid=bool(control_decision is not None and not control_decision.get('terminate') and
                np.isfinite(out[:,6]).all() and np.all(out[:,6]<=float(s[6])))
            record['current_control_compatible']=current_valid
            if not current_valid:
                self.invalidate_response('current_effective_control_decision_incompatible',t)
                record['trigger_veto']='current_control_incompatible_or_existing_guard_stop'
            elif self.latest_ready:
                self.start(s,t,'public_executed_closure_response_and_TRAIN_stage');record['triggered']=True
        if self.phase=='holding':
            out=self.commands(len(commands));record.update(action_source='isolated_terminal_hold',cursor=self.cursor,
                arm_reference_rad=self.reference[:6].tolist(),grip_reference_m=float(self.reference[6]),pending_VLA_prefix='discarded_not_queued',
                not_completion_confirmation=True,phase_after=self.phase)
        return out,record
    def start(self,state,t,source):
        if self.phase!='waiting':raise RuntimeError('terminal_hold_repeated_start')
        s=np.asarray(state);self.previous_control=s[14:21].astype(np.float32).copy()
        self.reference=self.previous_control.copy();self.reference[:6]=s[:6].astype(np.float32)
        self.started=float(t);self.phase='holding';self.cursor=0
        self.events.append({'kind':'hold_started','time_s':float(t),'source':source,'arm_start_control':self.previous_control[:6].tolist(),
                            'arm_measured_reference':self.reference[:6].tolist(),'grip_keeps_last_applied_target_m':float(self.reference[6]),
                            'measured_gap_not_substituted_m':float(s[6]),'phase_is_attempt_not_task_completion':True})
    def commands(self,count=8):
        out=np.repeat(self.reference[None],count,axis=0)
        for j in range(count):
            alpha=np.float32(min(1.,(self.cursor+j+1)/self.r['transition_commands']))
            out[j,:6]=self.previous_control[:6]+alpha*(self.reference[:6]-self.previous_control[:6])
        return out.astype(np.float32)
    def commit(self,steps,t):
        if self.phase=='holding':
            self.cursor+=int(steps)//40
            self.events.append({'kind':'hold_consumed','time_s':float(t),'steps':int(steps),'cursor':self.cursor,'partial_command_steps':int(steps)%40})
    def cancel(self,reason,t):
        self.stop_reason=reason;self.phase='terminated';self.events.append({'kind':'cancel_without_release_or_stale_queue','reason':reason,'time_s':float(t)})
    def archived(self):
        return {'phase':self.phase,'started':self.started,'consumed_commands':self.cursor,'reference':None if self.reference is None else self.reference.tolist(),
                'events':self.events,'observations':self.observations,'stop_reason':self.stop_reason,'release_executed':False,'completion_estimator':None}
