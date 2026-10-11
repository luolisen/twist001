"""Native choice about a bounded phase, built after current R/FK evidence."""
import copy,json
from public_stage_evidence import evidence_request
from composite_stage_gate import PHASE_BLOCKS

def stage_request(selection,event):
    req=evidence_request(selection);s=json.loads(req['state'])
    # Retain all finalized action forecasts as context, not executable choice options.
    contexts={name:json.loads(text) for name,text in selection['questions']['selection']['criteria'].items()
              if name not in ('observe','replan','reject')}
    forecasts=s['public_stage_evidence'].pop('forecast')
    costs=s['public_stage_evidence']['execution_cost'].pop('actions')
    e=event['binding'];limit=PHASE_BLOCKS[e['phase']]
    s['supervision_event']=copy.deepcopy(event)
    explicit_contract=s['supervision_event'].pop('explicit_composite_contract',None)
    explicit_binding=copy.deepcopy(s['supervision_event']['binding']) if explicit_contract is not None else None
    if explicit_binding is not None:s['supervision_event']['binding'].pop('owners')
    s['supervision_event'].update(maximum_phase_blocks=limit,maximum_phase_physics_s=limit*.320,
                                maximum_phase_wall_s=180,
                                approval_is_revocable=True,
                                future_blocks_require_new_VLA_and_WM_and_public_checks=True,
                                initial_forecast_not_a_prediction_of_entire_phase=True,
                                simulation_clock_paused_while_request_pending=True)
    # Proposed auxiliary activation is not an already executed transition.
    s['public_stage_evidence']['observed']['control_authority']=event['observed_authority']
    s['public_stage_evidence']['observed']['auxiliary_state']=event['observed_auxiliary_state']
    s['stage']['control_authority']=event['observed_authority']
    choices={'approve_phase':'Permit the proposed bounded autonomous phase: VLA supplies stage-internal actions, WM forecasts every final command, original protections and public evidence can revoke permission. Only the declared control sources are permitted; no release or task completion is authorized.',
             'wait_observe':'Do not grant phase permission now. Request one bounded observation via the provided legal hold control and its forecast, consuming original observation/hold budgets; no automatic continuation or release.',
             'replan':'Do not grant phase permission now. Request one existing bounded replan using new VLA output on the actual current public observation; no motion by this decision.',
             'terminate':'Reject or end this finite simulation without executing the proposed phase or releasing the gripper.'}
    choices['approve_phase']=json.dumps({'decision_semantics':choices['approve_phase'],
        'proposed_main_action':event['main_action_name'],
        'final_commands_WM_risks_and_R':contexts[event['main_action_name']],
        'forecast_evidence':forecasts[event['main_action_name']],
        'execution_cost':costs[event['main_action_name']]},separators=(',',':'),allow_nan=False)
    if explicit_contract is not None:
        approval=json.loads(choices['approve_phase'])
        approval['explicit_composite_contract']=explicit_contract
        approval['explicit_permission_binding']=explicit_binding
        choices['approve_phase']=json.dumps(approval,separators=(',',':'),allow_nan=False)
    wait=event.get('wait_action_name')
    if wait is None:choices.pop('wait_observe')
    else:choices['wait_observe']=json.dumps({'decision_semantics':choices['wait_observe'],
        'bounded_wait_action':wait,'final_commands_WM_risks_and_R':contexts[wait],
        'forecast_evidence':forecasts[wait],'execution_cost':costs[wait]},separators=(',',':'),allow_nan=False)
    if s['remaining']['replan']<=0:choices.pop('replan')
    shared_forecast_contract=s.pop('forecast_contract')
    shared_motor_units=s.pop('motor_units')
    req['state']=json.dumps(s,separators=(',',':'),allow_nan=False)
    req['questions']={'supervision':{'type':'choice','criteria':choices,
        'instructions':'Decide whether to permit the stated finite autonomous phase using current public observations, short-horizon WM forecasts, unknowns and execution costs. This is phase permission, not ranking raw versus hold actions. Approval is neither future success nor a safety certification; uncertainty remains explicit. Do not infer value from action names or assume approval is correct. Retain actual veto: wait, replan or terminate do not approve VLA execution. Existing physical protections and terminal control ownership remain independent and cannot be overridden.' + '\nShared forecast contract for all choices: ' + json.dumps(shared_forecast_contract,separators=(',',':'),allow_nan=False) + '\nShared encoder units for all choices: ' + shared_motor_units}}
    return req
