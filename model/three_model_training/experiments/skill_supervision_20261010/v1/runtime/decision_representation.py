"""One compact physical/stage presentation; exact matrices stay in WM/archives."""
import copy, json
import numpy as np
from fusion_core import check_public, compact, command_digest, CONTACT_TYPES

def dumps(x): return json.dumps(compact(x), separators=(',', ':'))

def revised_selection(public, target):
    check_public(public)
    memory=public['short_memory']; context=public['decision_context']
    receipts=memory['executed_commands']
    gaps=[x for x in memory['verified_progress'] if x['kind']=='encoder_gap_change']
    # No inferred contact/pose/height label is created from encoder contraction.
    facts=copy.deepcopy(memory['verified_progress'])
    state={'task':public['task'], 'stage':{'target_proposed':target['choice'],
        'proposal_only':True,'grasp_geometry':'unknown','grasp_contact':'unknown',
        'task_completion':None,'control_authority':context['stage_control_authority'],
        'encoder_contraction_observed':bool(gaps and gaps[-1]['after_m']<gaps[-1]['before_m']),
        'meaning':'Encoder contraction is measured motion, not alignment, grasp or a closure-phase confirmation.'},
        'motor_state':public['motor_state'], 'motor_units':'q6 rad,gap m,velocity6 rad/s,gap_velocity m/s,applied_target6 rad,applied_gap m',
        'RGB_history':{'mask':public['history_valid'],'times_s':context.get('history_capture_times_seconds'),
        'layout':public['image_layout'],'target_proposal':target},
        'past_only':{'decisions':memory['prior_decisions'],'executions':receipts,'observed_facts':facts},
        'remaining':{'consecutive_hold':context['remaining_consecutive_hold_budget'],
            'observation':context['remaining_observation_budget'],'replan':context['remaining_replan_budget']},
        'hold_count':context['consecutive_hold_count'],
        'forecast_contract':{'horizon_s':.320,'contact_order':list(CONTACT_TYPES),
            'contact_arrays':'all current-window/persistent-inclusive, newly-starting, unlocalized probabilities; estimates not calibrated safety guarantees',
            'motion':'WM joint delta6 rad,gap delta mm,world EE delta XYZ mm; estimate, not measured motion',
            'command_summary':'first/last/range of actual 8 absolute targets, joints rad,gap mm; summary is not a reversible matrix or a substitute for executing exact archived commands',
            'task_progress':'EE motion alone does not establish motion toward target, alignment, grasp or final success'}}
    binding={i:b for i,b in enumerate(context['candidate_control_bindings'])}
    choices={}
    for c in public['candidates']:
        a=np.asarray(c['actions'],dtype=np.float32)
        assert command_digest(a)==c['command_sha256']
        scale=np.array([1]*6+[1000.]); values=a.astype(float)*scale
        md=c['motion_delta']
        b=binding[c['candidate_index']]
        item={'index':c['candidate_index'],'action_sha256':c['command_sha256'],
            'owners':{'arm':b['owners'][0] if len(set(b['owners'][:6]))==1 else b['owners'][:6], 'gripper':b['owners'][6]},
            'plan_consumption':[b['consume_gripper'],b['consume_hold']],
            'targets':{'first':values[0].tolist(),'last':values[-1].tolist(),
                'range':[values.min(0).tolist(),values.max(0).tolist()]},
            'WM':{'contacts':[c['physical_contacts'][k] for k in CONTACT_TYPES],
                'new':[c['new_contacts'][k] for k in CONTACT_TYPES],
                'unlocalized':[c['unlocalized_contacts'][k] for k in CONTACT_TYPES],
                'joint_delta_rad':md[:6], 'gap_delta_mm':md[6]*1000.,
                'EE_delta_world_mm':[v*1000. for v in md[7:10]],
                'grasp_estimate':c['any_grasp_estimate'],'grip_loss_estimate':c['grip_loss_estimate']}}
        choices[c['name']]=dumps(item)
    if 'command_hold' in choices and context['remaining_observation_budget']>0 and context['remaining_consecutive_hold_budget']>0:
        choices['observe']='Execute the identical command_hold index, digest, commands and WM forecast above; obtain normal RGB/encoders, no additional viewpoint or assured information.'
    if context['remaining_replan_budget']>0:
        choices['replan']='Recompute VLA on current real observation using the next registered seed; no motion or consumption by this decision; limited remaining budget.'
    choices['reject']='Stop this finite run without releasing or executing this block.'
    return {'model':'NeoHorse-Jev-4B','state':dumps(state), 'questions':{'selection':{'type':'choice','criteria':choices,
        'instructions':'Choose among legal candidates for the current task and horizon. Consider predicted motion, risks and uncertainty together. Existing support or necessary finger contact is not automatically forbidden; new/unlocalized contact remains risk information. Grasp and loss estimates are unverified. A small encoder contraction does not confirm a grasp phase. A hold has its own physical motion forecast and consumes finite stall budget; it is not task completion. Any choice may be justified by available evidence; never assume a particular candidate is the correct answer. Respect control ownership; terminal_hold is sole motion when it owns control.'}}}
