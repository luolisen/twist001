"""Finite gripper execution aid. Inputs: RGB, encoders, time and current prediction only."""
import hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import label
from public_association import PublicAssociation, ReadinessEvidence, bbox_match

def red_regions(rgb):
    im = Image.fromarray(np.asarray(rgb)).resize((128,128), Image.Resampling.BILINEAR).convert('HSV')
    h = np.asarray(im,dtype=np.float32)
    hue,s,v=h[:,:,0]*360/255,h[:,:,1]/255,h[:,:,2]/255
    mask=((hue<=15)|(hue>=345))&(s>=.45)&(v>=.25)
    ids,n=label(mask,structure=np.ones((3,3),dtype=np.uint8))
    result=[]
    for i in range(1,n+1):
        yy,xx=np.where(ids==i)
        if len(xx)<8:continue
        result.append({'area':len(xx),'fraction':len(xx)/16384,
                       'center':[float(xx.mean()/127),float(yy.mean()/127)],
                       'bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]})
    return sorted(result,key=lambda c:(-c['area'],c['bbox']))

def q_distance(q,templates):
    return float(np.sqrt(np.mean((np.asarray(templates)-np.asarray(q)[:6])**2,axis=1)).min())

class Continuation:
    def __init__(self,rules):
        self.rules=rules;self.phase='idle';self.plan=None;self.cursor=0
        self.front_anchor=None;self.wrist_anchor=None;self.last_reliable=None
        self.last_observation_time=None;self.start_ready_time=None;self.active_started=None
        self.exhausted_at=None;self.last_command=None;self.pending=None
        self.events=[];self.observations=[];self.public_stop=None;self.started_count=0
        self.association=PublicAssociation(rules);self.evidence=ReadinessEvidence(rules)
        self.evidence_ready=False

    def stop(self,reason,t):
        if self.public_stop is None:
            self.public_stop=reason;self.events.append({'kind':'terminate_without_release','reason':reason,'time_s':t,
                                                       'phase':self.phase,'consumed_commands':self.cursor})
        return reason

    def observe(self,observation):
        t=float(observation['time']);state=np.asarray(observation['state']);full=observation['full']
        valid=(state.shape==(21,) and np.isfinite(state).all() and len(full)==2 and
               all(np.asarray(im).dtype==np.uint8 and np.asarray(im).shape==shape and np.any(im)
                   for im,shape in zip(full,[(256,256,3),(512,512,3)])))
        if self.last_observation_time is not None:
            delta=t-self.last_observation_time
            valid=valid and -.000000001<=delta<=.040000001
        item={'time_s':t,'input_valid':bool(valid),'phase':self.phase,'cursor':self.cursor}
        if not valid:
            self.stop('public_input_invalid_or_not_timely',t);item['terminate']=self.public_stop
            self.observations.append(item);return item
        self.last_observation_time=t
        front,wrist=[red_regions(im) for im in full]
        had_front=self.association.front is not None
        front_match,wrist_match,ambiguous,reconfirmed=self.association.update(front,wrist,state[:6],t)
        if front_match is not None:
            self.front_anchor=front_match['center']
            if not had_front:self.events.append({'kind':'public_target_track_initialized','time_s':t,'front':front_match,
                'identity_is_visual_track_proposal_not_private_object_ID':True})
        if reconfirmed:self.events.append({'kind':'public_front_reconfirmed_original_bbox','time_s':t})
        if wrist_match is not None:self.wrist_anchor=wrist_match['center']
        close_distance=q_distance(state,self.rules['close_q_templates'])
        compatible=q_distance(state,self.rules['close_lift_q_templates'])<=self.rules['q_tolerance_rad']
        if self.phase=='idle':
            reliable=not ambiguous and (front_match is not None or wrist_match is not None)
            if reliable:self.last_reliable=t
            area_ok=bool(wrist_match and self.rules['wrist_fraction_min']<=wrist_match['fraction']<=self.rules['wrist_fraction_max'])
            box_ok=bool(wrist_match and bbox_match(wrist_match,{'bbox':self.rules['wrist_ready_bbox']},0))
            if ambiguous or close_distance>self.rules['q_tolerance_rad'] or (wrist_match and not (area_ok and box_ok)):
                evidence_kind='contradiction'
            elif wrist_match is not None and area_ok and box_ok:evidence_kind='positive'
            else:evidence_kind='uncertain'
            ev=self.evidence.update(t,evidence_kind);self.evidence_ready=ev['ready']
            item.update(readiness_evidence=ev,visual_readiness_area_pass=area_ok,visual_readiness_bbox_pass=box_ok)
        else:
            # The second camera corroborates an established track, never initiates
            # or substitutes a different red object. Fresh frames alone do not renew.
            reliable=not ambiguous and compatible and (front_match is not None or wrist_match is not None)
            if wrist_match is not None:self.wrist_anchor=wrist_match['center']
            if reliable:self.last_reliable=t
            if ambiguous:self.stop('ambiguous_target_correspondence',t)
            if not compatible:self.stop('actual_arm_left_registered_close_lift_envelope',t)
            if self.last_reliable is None or t-self.last_reliable>self.rules['maximum_unconfirmed_seconds']+1e-9:
                self.stop('public_target_unconfirmed_budget_exhausted',t)
        item.update(front_track_confirmed=front_match is not None,wrist_track_confirmed=wrist_match is not None,
                    target_correspondence_ambiguous=ambiguous,last_reliable_target_time_s=self.last_reliable,
                    unconfirmed_duration_s=None if self.last_reliable is None else t-self.last_reliable,
                    start_ready_duration_s=None,
                    close_q_RMSE_rad=close_distance,arm_compatible=compatible,
                    current_gap_m=float(state[6]),front_components=front,wrist_components=wrist,
                    target_grasp_estimate=None,terminate=self.public_stop)
        self.observations.append(item)
        return item

    def propose(self,prediction,commands,observation,source_id):
        t=float(observation['time']);state=np.asarray(observation['state'])
        record={'phase_before':self.phase,'time_s':t,'prediction_source':source_id,'cursor_before':self.cursor,
                'not_grasp_confirmation':True,'terminate':self.public_stop}
        result=np.asarray(commands).copy()
        if self.public_stop:return result,record
        gap=np.clip(np.asarray(prediction)[:,6],0,1).astype(np.float32)*np.float32(.08)
        if not np.isfinite(gap).all() or len(gap)<8:
            record['terminate']=self.stop('invalid_gripper_prediction',t);return result,record
        if self.phase=='idle':
            ready=self.evidence_ready and abs(t-self.evidence.previous_t)<=1e-9
            open_enough=state[6]>=self.rules['open_gap_min_m']
            closure=float(np.mean(gap[:8])-np.mean(gap[-8:]))>=self.rules['predicted_closure_min_m']
            record.update(start_conditions={'finite_window_effective_ready_evidence':bool(ready),
                                            'measured_gap_open_enough':bool(open_enough),
                                            'current_prediction_contains_closure':bool(closure)})
            if ready and open_enough and closure:
                self.plan=gap.copy();self.cursor=0;self.phase='active';self.active_started=t;self.started_count+=1
                self.events.append({'kind':'plan_started','time_s':t,'source':source_id,'length':len(gap),
                                    'gap_plan_sha256':hashlib.sha256(gap.tobytes()).hexdigest(),
                                    'gap_targets_m':gap.tolist()})
        if self.phase=='active':
            remain=len(self.plan)-self.cursor;take=min(8,remain)
            result[:take,6]=self.plan[self.cursor:self.cursor+take]
            if take<8:result[take:,6]=self.plan[-1]  # same registered block, finite tail handoff
            self.pending={'take':take,'tail_hold':8-take,'time_s':t}
            record.update(mode='consume_active_plan',cursor_before=self.cursor,planned_consumption=take,
                          finite_tail_handoff_commands=8-take,plan_length=len(self.plan))
        elif self.phase in ('exhausted','handed_off'):
            nonopening=bool(np.all(gap[:8]<=np.float32(state[6])))
            if nonopening:
                result[:,6]=gap[:8];self.phase='handed_off'
                record.update(mode='public_nonopening_fresh_policy_handoff',release_authorized=False)
            elif self.phase=='handed_off':
                record['terminate']=self.stop('fresh_policy_requests_release_without_verified_deposit',t)
            elif t-self.exhausted_at<self.rules['maximum_handoff_seconds']-1e-9:
                result[:,6]=self.plan[-1];record.update(mode='bounded_exhaustion_handoff',release_authorized=False)
            else:
                record['terminate']=self.stop('finite_plan_handoff_budget_exhausted',t)
            self.pending={'take':0,'tail_hold':0,'time_s':t}
        else:
            self.pending={'take':0,'tail_hold':0,'time_s':t};record['mode']='original_rolling_gripper'
        record.update(phase_after_proposal=self.phase,actual_gap_targets_m=result[:,6].tolist(),
                      arm_targets_unchanged=bool(np.array_equal(result[:,:6],commands[:,:6])))
        return result,record

    def commit(self,physical_steps,actual_end_time):
        # Only fully executed 40ms commands consume the cached plan; an interrupted
        # partial command is recorded but never silently replayed in another run.
        complete=int(physical_steps)//40
        before=self.cursor
        if self.phase=='active' and self.pending is not None:
            self.cursor+=min(complete,self.pending['take'])
            if self.cursor==len(self.plan):
                self.phase='exhausted';self.exhausted_at=float(actual_end_time)-(complete-self.pending['take'])*.040
                self.events.append({'kind':'plan_exhausted','time_s':float(actual_end_time),'cursor':self.cursor})
        receipt={'kind':'execution_consumed','time_s':float(actual_end_time),'actual_steps':int(physical_steps),
                 'fully_executed_commands':complete,'partial_command_steps':int(physical_steps)%40,
                 'cursor_before':before,'cursor_after':self.cursor,'phase':self.phase}
        self.events.append(receipt);self.pending=None
        return receipt

    def archived(self):
        return {'phase':self.phase,'cursor':self.cursor,'plan_started_count':self.started_count,
                'last_reliable_target_time_s':self.last_reliable,'public_guard_stop':self.public_stop,
                'events':self.events,'observations':self.observations,
                'release_policy':'grasp-only trial; no automatic release without verified deposit; end episode rather than reopen',
                'private_truth_used':False}
