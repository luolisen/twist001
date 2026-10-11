"""Native Jev choice request; no new model output protocol or default approval."""
import copy,json

def skill_request(base_request,event,final_action_forecast,public_evidence):
    req=copy.deepcopy(base_request)
    state=json.loads(req['state'])
    if not isinstance(state,dict):raise ValueError('base_state_must_be_structured')
    # Retain task, actual encoders, history, units and existing unknown/risk context.
    # Only the evidence layer and proposed supervision event are updated.
    state.update(public_stage_evidence=public_evidence,observed_skill=event['previous_skill'],pending_event=event['id'])
    req['state']=json.dumps(state,separators=(',',':'),allow_nan=False)
    # Complete contract and action evidence live in a native choice branch, preserving
    # the service state-length limit. No unverified carry/deposit becomes approval.
    approve={'proposed_skill':event['skill'],'finite_revocable_contract':event,
             'final_action_and_WM_evidence':final_action_forecast,
             'not_a_task_success_or_safety_certificate':True}
    criteria={'approve_phase':json.dumps(approve,separators=(',',':'),allow_nan=False),
              'replan':'No motion permission. Request bounded registered replanning; no implicit release.',
              'terminate':'No motion permission. End this finite simulation; no implicit release.'}
    # There is no validated loaded wait control in this first v2 interface.
    req['questions']={'supervision':{'type':'choice','criteria':criteria,
        'instructions':'Authorize only the declared registered skill, finite duration and seven-channel owners using public observations and current WM forecasts. Unknown carry or deposit cannot be turned into success. Decline if the contract cannot be supported. Existing execution guards remain independent; rejection or invalid output never executes a fallback policy.'}}
    return req
