"""Public-only evidence layering of an existing R/FK choice request.

This module never ranks, filters, executes, commits, or scores an action.
Legacy state, action payloads, risks and choice instructions remain unchanged.
"""
import copy,json,math

def _number(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
        raise ValueError('nonfinite or invalid public numeric input')
    return v

def evidence_request(request):
    out=copy.deepcopy(request);s=json.loads(out['state'])
    if 'public_stage_evidence' in s:
        raise ValueError('evidence already attached; do not double-generate')
    if out['questions']['selection']['type']!='choice':
        raise ValueError('this adapter only supports the existing choice interface')
    motor=s['motor_state']
    if len(motor)!=21:raise ValueError('unsupported encoder layout')
    for v in motor:_number(v)
    mask=s['RGB_history']['mask']
    if any(v not in (0,1) for v in mask):raise ValueError('invalid RGB validity mask')
    past=s['past_only']; receipts=past['executions']
    relation=s['task_relative_observation'];proposal=s['RGB_history']['target_proposal']
    authority=s['stage']['control_authority'];remaining=s['remaining'];holds=s['hold_count']
    for v in [holds,*remaining.values()]:
        if type(v)!=int or v<0:raise ValueError('invalid finite decision budget')
    usable=(bool(relation.get('valid')) and bool(mask[-1]) and proposal['choice']=='red'
            and relation.get('target_region') is not None)
    unknown={
        'target_depth':None,'full_grasp_pose':None,'full_path_clearance':None,
        'actual_future_contact':None,'future_completion':None,
        'projection_change_error_calibration':None,
        'meaning':'unconfirmed, not disproved; missing depth does not negate the 2D forecast',
    }
    # This layer does not turn a gate, encoder contraction or execution receipt
    # into contact, grasp, or task completion. No prior-to-current visual trend
    # is manufactured from a single frame or command receipt.
    observed={
        'source':'current valid RGB/encoders and past executed receipts only',
        'RGB_valid_slots':sum(mask),'RGB_capture_times_s':s['RGB_history']['times_s'],
        'target_visual_proposal':proposal['choice'],'target_identity_confirmed':relation.get('target_identity_verified',False),
        'relation_usable':usable,'relation_source_status':relation.get('target_relation_status'),
        'current_q_rad':motor[:6],'current_gap_m':motor[6],
        'executed_receipts_present':bool(receipts),
        'executed_receipts_reference':'past_only.executions; actual physical steps and times retained there',
        'confirmed_task_progress':None,'measured_target_approach_change':None,
        'control_authority':authority,
    }
    forecast={};cost={}
    for name,text in out['questions']['selection']['criteria'].items():
        try:c=json.loads(text)
        except json.JSONDecodeError:c=None
        if isinstance(c,dict) and 'WM' in c:
            if authority=='terminal_hold' and name!='terminal_hold':
                raise ValueError('ordinary motion incompatible with terminal control authority')
            f=c['task_relative_forecast'];h=_number(f['horizon_s'])
            if h<=0:raise ValueError('invalid forecast horizon')
            delta=c['WM']['EE_delta_world_mm']
            if c['WM'].get('EE_motion_source')!='static_FK_of_current_public_encoder_plus_WM_predicted_joint_delta':
                raise ValueError('unsupported EE forecast source')
            if len(delta)!=3:raise ValueError('invalid derived EE vector')
            for v in delta:_number(v)
            projection=f.get('predicted_ee_image_projection',{})
            valid=usable and projection.get('valid') is True
            current=predicted=shrink=None
            if valid:
                current=_number(f['current_image_center_offset_norm_pixels'])
                predicted=_number(f['predicted_image_center_offset_norm_pixels'])
                if min(current,predicted)<0:raise ValueError('negative image distance')
                shrink=current-predicted
            forecast[name]={
                'action_index':c['index'],
                'source':'WM joint response + static FK + current fixed-camera projection',
                'horizon_s':h,'EE_delta_world_mm':delta,
                'relation_valid':bool(valid),'current_offset_px':current,'predicted_offset_px':predicted,
                'predicted_offset_decrease_px':shrink,
                'status':'forecast relative to current target region, not measured progress or future target motion',
            }
            terminal=authority=='terminal_hold'
            # Matches the frozen runner's actual front-stage counter rule.
            counts_as_front_hold=name in ('pose_hold','command_hold') and not terminal
            cost[name]={
                'physical_time_s_if_full_execution':h,
                'front_consecutive_hold_increment_if_executed':1 if counts_as_front_hold else 0,
                'counter_resets_if_executed':not counts_as_front_hold,
                'observation_allowance_consumed':0,
                'status':'conditional on actual execution; partial/aborted receipts remain authoritative',
            }
        elif name=='replan':
            cost[name]={'physical_time_s_by_this_choice':0,'replan_allowance_if_attempted':1,'no_motion_receipt_by_choice':True}
        elif name=='reject':
            cost[name]={'physical_time_s_by_this_choice':0,'ends_finite_run_without_release':True}
        elif name=='observe':
            # Exact existing alias contract, independently of iteration order.
            known='Execute the identical command_hold index, digest, commands and WM forecast above; obtain normal RGB/encoders, no additional viewpoint or assured information.'
            if text!=known or authority=='terminal_hold':
                raise ValueError('unsupported observation alias/control ownership')
            hold=json.loads(out['questions']['selection']['criteria']['command_hold'])
            h=_number(hold['task_relative_forecast']['horizon_s'])
            forecast[name]={'same_forecast_as':'command_hold','not_extra_viewpoint_or_assured_information':True}
            cost[name]={'physical_time_s_if_full_execution':h,
                        'front_consecutive_hold_increment_if_executed':1,
                        'observation_allowance_consumed':1,
                        'actual_control_source':'command_hold',
                        'status':'observation budget charged on accepted choice; hold counter only after actual executed control'}
        else:
            raise ValueError('unsupported executable alias; must bind its actual control and budget semantics')
    evidence={
        'observed':observed,'forecast':forecast,'unknown':unknown,
        'execution_cost':{'current_hold_count':holds,'remaining':copy.deepcopy(remaining),'actions':cost,
                          'meaning':'time and finite-budget cost, not collision risk or a ranking'},
    }
    s['public_stage_evidence']=evidence
    out['state']=json.dumps(s,separators=(',',':'),allow_nan=False)
    return out
