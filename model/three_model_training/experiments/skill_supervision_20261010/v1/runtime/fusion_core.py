"""Public-only contracts and finite recovery; imports no model or simulator."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from candidate_boundary import diagnose_candidates

CONTACT_TYPES = ('object_finger', 'object_nonfinger', 'object_object',
                 'object_tray_wall', 'object_tray_bottom', 'object_floor',
                 'robot_tray', 'robot_floor', 'robot_self')
TASK = 'pick up one red cube and keep it suspended; do not move any other cube'
RECOVERY_CHOICES = ('observe', 'replan', 'reject')


def compact(value):
    """Original Jev five-significant-digit text presentation; archive stays exact."""
    if isinstance(value,float):
        if not math.isfinite(value):
            raise ValueError('nonfinite public numeric value')
        return float(format(value,'.5g'))
    if isinstance(value,list):
        return [compact(v) for v in value]
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items()}
    return value


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8388608), b''):
            h.update(part)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def command_digest(commands):
    data = np.asarray(commands, dtype=np.float32)
    return hashlib.sha256(data.tobytes(order='C')).hexdigest()


def filtered_candidates(raw, state, previous_control, limits, actuator_names):
    """Keep the actual float32 matrix identical across WM, Jev and execution."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (8, 7):
        raise ValueError('VLA prefix must be 8x7')
    smooth, prev = [], np.asarray(state[:7], dtype=np.float32).copy()
    for command in raw:
        prev = prev + np.clip(command - prev, -np.array([.02] * 6 + [.002]),
                              np.array([.02] * 6 + [.002]))
        smooth.append(prev.copy())
    sources = {'vla_raw': raw, 'vla_smoothed': np.asarray(smooth, dtype=np.float32),
               'pose_hold': np.repeat(np.asarray(state[:7], dtype=np.float32)[None], 8, axis=0),
               'command_hold': np.repeat(np.asarray(previous_control, dtype=np.float32)[None], 8, axis=0)}
    diagnostic = diagnose_candidates(sources, limits, actuator_names)
    kept = [{ 'name': name, 'commands': sources[name], 'command_sha256': command_digest(sources[name])}
            for name in diagnostic['retained_candidate_names']]
    for index, entry in enumerate(kept):
        entry['candidate_index'] = index
    return kept, diagnostic


def lookup_candidate(candidates, choice):
    matches = [c for c in candidates if c['name'] == choice]
    if len(matches) != 1:
        raise ValueError('choice is not a currently legal candidate')
    candidate = matches[0]
    if candidates[candidate['candidate_index']] is not candidate:
        raise ValueError('candidate index mapping changed')
    if command_digest(candidate['commands']) != candidate['command_sha256']:
        raise ValueError('candidate commands changed after forecast')
    return candidate


def history_arrays(history):
    if not history:
        raise ValueError('at least one genuinely captured frame required')
    chosen = history[-3:]
    times = np.asarray([h['time'] for h in chosen], dtype=np.float64)
    if not np.allclose(times - times[-1], np.arange(-len(chosen)+1, 1)*.04, rtol=0, atol=1e-9):
        raise ValueError('WM history must have genuine 40ms intervals')
    missing=3-len(chosen)
    images=np.stack([np.zeros_like(chosen[-1]['small'])]*missing+[h['small'] for h in chosen])
    states=np.stack([np.zeros_like(chosen[-1]['state'])]*missing+[h['state'] for h in chosen])
    valid=np.asarray([0]*missing+[1]*len(chosen),dtype=np.uint8)
    # Nominal timestamps for invalid zero slots are explicitly not captured frames.
    slots=times[-1]+np.asarray([-.08,-.04,0])
    return images,states,valid,slots


@dataclass
class Budget:
    limits: dict
    counts: dict = field(default_factory=lambda: dict(decisions=0, VLA_calls=0, WM_forecasts=0,
                                                     Jev_calls=0, action_chunks=0, physics_steps=0,
                                                     observations=0, replans=0, ineffective_replans=0))
    parent: object = None

    def charge(self, key, amount=1):
        maximum = self.limits[key]
        if self.counts[key] + amount > maximum:
            raise RuntimeError('budget_exhausted:' + key)
        if self.parent is not None:
            self.parent.charge(key,amount)
        self.counts[key] += amount


@dataclass
class PublicMemory:
    """Only public execution facts; task truth belongs to a separate evaluator."""
    stage: str = 'target_search'
    prior_decisions: list = field(default_factory=list)
    executed_commands: list = field(default_factory=list)
    verified_progress: list = field(default_factory=list)
    closure_observed: bool = False

    def target_proposal(self, answer, decision_id):
        if answer['choice'] == 'red':
            # Visual model proposal is NOT a verified object identity or grasp.
            self.stage = 'grasp_attempt'
        self.prior_decisions.append({'decision_id': decision_id, 'RGB_target_proposal': answer['choice'],
                                     'proposal_only': True})
        self.prior_decisions = self.prior_decisions[-2:]

    def receipt(self, decision_id, candidate_name, before_state, after_state, start, end, steps, digest):
        before, after = np.asarray(before_state), np.asarray(after_state)
        self.executed_commands.append({'decision_id': decision_id, 'candidate': candidate_name,
                                       'command_sha256': digest, 'start_time': float(start),
                                       'end_time': float(end), 'physics_steps': int(steps)})
        self.executed_commands = self.executed_commands[-2:]
        if steps:
            self.verified_progress = [{'kind': 'control_block_executed', 'source': 'execution_receipt',
                                       'decision_id': decision_id, 'physics_steps': int(steps)}]
            self.closure_observed = bool(after[6] < before[6])
            self.stage = 'closure_observed' if self.closure_observed else 'grasp_attempt'
            self.verified_progress.append({'kind': 'encoder_gap_change', 'source': 'joint_encoders',
                                           'before_m': float(before[6]), 'after_m': float(after[6]),
                                           'not_grasp_confirmation': True})

    def public(self):
        return {'stage': self.stage, 'prior_decisions': self.prior_decisions,
                'executed_commands': self.executed_commands, 'verified_progress': self.verified_progress,
                'system_completion': {'value': None, 'reason': 'public_grasp_completion_estimator_not_validated'}}


def check_memory(memory):
    if set(memory) != {'stage', 'prior_decisions', 'executed_commands', 'verified_progress', 'system_completion'}:
        raise ValueError('unexpected memory keys; privileged evaluator data forbidden')
    if memory['stage'] not in {'target_search', 'grasp_attempt', 'closure_observed'}:
        raise ValueError('unsupported public stage')
    if memory['system_completion'] != {'value': None, 'reason': 'public_grasp_completion_estimator_not_validated'}:
        raise ValueError('task truth must not become public completion')
    for row in memory['prior_decisions']:
        allowed={'decision_id','RGB_target_proposal','proposal_only','choice'}
        if not set(row)<=allowed or row.get('proposal_only') is not True:
            raise ValueError('invalid prior decision provenance')
    for row in memory['executed_commands']:
        if set(row)!={'decision_id','candidate','command_sha256','start_time','end_time','physics_steps'}:
            raise ValueError('invalid public execution receipt')
    for p in memory['verified_progress']:
        allowed = {'control_block_executed': ({'kind', 'source', 'decision_id', 'physics_steps'}, 'execution_receipt'),
                   'encoder_gap_change': ({'kind', 'source', 'before_m', 'after_m', 'not_grasp_confirmation'}, 'joint_encoders')}
        if p.get('kind') not in allowed:
            raise ValueError('unsupported verified progress fact')
        keys, source = allowed[p['kind']]
        if set(p) != keys or p['source'] != source:
            raise ValueError('invalid progress provenance')
        if p['kind'] == 'encoder_gap_change' and p['not_grasp_confirmation'] is not True:
            raise ValueError('encoder motion is not grasp confirmation')


def check_public(public):
    expected = {'decision_id', 'task', 'motor_state', 'history_valid', 'short_memory',
                'candidates', 'image_layout', 'scope'}
    if set(public) not in (expected, expected | {'decision_context'}) or not isinstance(public['task'], str):
        raise ValueError('invalid public contract')
    if len(public['motor_state']) != 21 or public['history_valid'] not in ([0,0,1],[0,1,1],[1,1,1]):
        raise ValueError('invalid public state/history')
    check_memory(public['short_memory'])
    if 'decision_context' in public:
        check_decision_context(public['decision_context'], public['history_valid'])
    names = [c['name'] for c in public['candidates']]
    if not names or len(set(names)) != len(names) or not set(names) <= {'vla_raw', 'vla_smoothed', 'pose_hold', 'command_hold', 'terminal_hold'}:
        raise ValueError('invalid legal candidate names')
    keys = {'name', 'candidate_index', 'command_sha256', 'actions', 'physical_contacts', 'new_contacts',
            'unlocalized_contacts', 'any_grasp_estimate', 'grip_loss_estimate', 'motion_delta'}
    for index, candidate in enumerate(public['candidates']):
        if set(candidate) != keys or candidate['candidate_index'] != index:
            raise ValueError('invalid candidate mapping')
        commands = np.asarray(candidate['actions'], dtype=np.float32)
        if commands.shape != (8, 7) or not np.isfinite(commands).all():
            raise ValueError('invalid actions')
        if command_digest(commands) != candidate['command_sha256']:
            raise ValueError('public action digest differs from WM matrix')
        for field in ('physical_contacts', 'new_contacts', 'unlocalized_contacts'):
            if set(candidate[field]) != set(CONTACT_TYPES):
                raise ValueError('invalid contact schema')
    json.dumps(public, allow_nan=False)


def check_decision_context(context, history_valid):
    required = {'interface_revision', 'consecutive_hold_count', 'remaining_consecutive_hold_budget',
                'remaining_observation_budget', 'history_source', 'history_valid_mask'}
    optional = {'history_capture_times_seconds', 'presentation_revision','stage_control_authority','candidate_control_bindings','remaining_replan_budget'}
    if not required <= set(context) or not set(context) <= required | optional:
        raise ValueError('invalid decision context keys; evaluator facts forbidden')
    if context.get('presentation_revision', 'full_duplicate_v1') not in ('full_duplicate_v1','shared_full_hold_v1'):
        raise ValueError('unknown presentation revision')
    if context['interface_revision'] not in ('baseline', 'consistent_v1'):
        raise ValueError('unknown interface revision')
    for key in ('consecutive_hold_count', 'remaining_consecutive_hold_budget', 'remaining_observation_budget'):
        if type(context[key]) is not int or context[key] < 0:
            raise ValueError('decision budget must be actual nonnegative integer')
    if context['history_source'] != 'captured_rgb_and_joint_encoders' or context['history_valid_mask'] != history_valid:
        raise ValueError('history provenance mismatch')
    if 'candidate_control_bindings' in context:
        if context['stage_control_authority'] not in ('terminal_hold','policy_with_guarded_gripper'):
            raise ValueError('invalid control authority')
        keys={'action_id','state_version','observation_time_s','command_sha256','owners','control_stage','consume_gripper','consume_hold'}
        for b in context['candidate_control_bindings']:
            if set(b)!=keys or len(b['owners'])!=7 or b['state_version']<1 or not math.isfinite(b['observation_time_s']):
                raise ValueError('invalid public action binding')
            if not set(b['owners'])<={'VLA','VLA_arm_slew','gripper_plan','terminal_hold','pose_hold','command_hold','last_applied_clamp_target'}:
                raise ValueError('invalid channel owner')
        if type(context['remaining_replan_budget']) is not int or context['remaining_replan_budget']<0:
            raise ValueError('invalid replan budget')
    if 'history_capture_times_seconds' in context:
        times = context['history_capture_times_seconds']
        if len(times) != sum(history_valid) or not all(math.isfinite(t) for t in times):
            raise ValueError('only valid captured frame timestamps allowed')
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('capture times must advance')


def consistent_interface(public):
    return public.get('decision_context', {}).get('interface_revision') == 'consistent_v1'


def request_state(public):
    # Baseline requests stay byte-equivalent to the original presentation even when
    # the runner archives context for both conditions.
    state = {k: v for k, v in public.items() if k != 'candidates'
             and (k != 'decision_context' or consistent_interface(public))}
    if consistent_interface(public):
        def omit_trace_ids(value):
            if isinstance(value, dict): return {k: omit_trace_ids(v) for k,v in value.items() if k not in ('decision_id','command_sha256')}
            if isinstance(value, list): return [omit_trace_ids(v) for v in value]
            return value
        # IDs/hashes are provenance, not execution facts. Complete public is unchanged
        # in IPC/archive; retain every measured value, execution time, step and source.
        state = omit_trace_ids(state)
        context=state.get('decision_context',{})
        if 'candidate_control_bindings' in context:
            # Full transaction IDs/version/hash and 7-channel owners remain in IPC/archive.
            # Present the same authority compactly without repeating trace IDs or six identical arm owners.
            bindings=context.pop('candidate_control_bindings')
            context['channel_authority']=[{'candidate_index':i,
                'arm':b['owners'][0] if len(set(b['owners'][:6]))==1 else b['owners'][:6],
                'gap':b['owners'][6], 'consume_gripper_plan':b['consume_gripper'],
                'consume_terminal_hold':b['consume_hold']} for i,b in enumerate(bindings)]
        state['image_layout'] = 'Rows -80,-40,0ms: global,wrist; mask 1=captured RGB,0=missing zero slot'
        state['scope'] = 'Frozen online simulation; next320ms forecasts; no hardware release or task-truth memory'
    return state


def target_request(public):
    check_public(public)
    return {'model': 'NeoHorse-Jev-4B', 'state': json.dumps(compact(request_state(public)), separators=(',', ':')),
            'questions': {'target': {'type': 'choice', 'instructions': 'Use actual RGB history. Propose whether a red cube is visible; this is not verified identity, grasp or completion.',
                                     'criteria': {'red': 'Visible task-relevant red cube.', 'not_visible': 'No red target visible.', 'uncertain': 'Insufficient RGB evidence.'}}}}


def selection_request(public, target):
    check_public(public)
    state = request_state(public)
    state['native_RGB_target_proposal'] = {'choice': target['choice'], 'probabilities': target['probabilities'],
                                         'proposal_only': True, 'identity_or_progress_verified': False}
    choices={}
    for candidate in public['candidates']:
        presented=compact(candidate)
        # Jev must receive a round-trip identical float32 action payload, not rounded commands.
        presented['actions']=candidate['actions']
        if command_digest(np.asarray(presented['actions'],dtype=np.float32))!=candidate['command_sha256']:
            raise ValueError('Jev action presentation differs from forecast/execution matrix')
        if 'candidate_control_bindings' in public.get('decision_context',{}):
            runs=[]
            for row in presented['actions']:
                if runs and np.array_equal(np.asarray(runs[-1]['row'],dtype=np.float32),np.asarray(row,dtype=np.float32)):
                    runs[-1]['repeat']+=1
                else:runs.append({'repeat':1,'row':row})
            if len(runs)<8:
                expanded=np.asarray([r['row'] for r in runs for _ in range(r['repeat'])],dtype=np.float32)
                if command_digest(expanded)!=candidate['command_sha256']:raise ValueError('lossless row packing differs from execution')
                presented.pop('actions');presented['actions_exact_row_runs']=runs
                presented['actions_encoding']='Expand each exact row repeat times in order, one40ms instruction per repeat; shape8x7'
        choices[candidate['name']]=json.dumps(presented,separators=(',', ':'))
    if consistent_interface(public):
        shared_layout = public['decision_context'].get('presentation_revision') == 'shared_full_hold_v1'
        state['action_contract'] = {
            'actions': '8x7 float32 absolute [J1-J6 rad,gap m],40ms/row; horizon320ms. actions_exact_row_runs is lossless consecutive-row packing: expand repeat copies of row.',
            'state': '21=[q6 rad,gap m,qvel6 rad/s,gapvel m/s,ctrl6 rad,gap m]',
            'motion': '10=[dq6 rad,dgap m,dEE world XYZ m]',
            'contacts': 'physical may persist; new starts this block; unlocalized lacks reliable counterpart',
            'observe': 'identical command_hold name/index/hash/actions/predictions; every executed action captures RGB/encoders; no extra view or assured information'}
        context = public['decision_context']
        hold = next((c for c in public['candidates'] if c['name'] == 'command_hold'), None)
        if hold is not None:
            # Full same forecast/actions on both aliases: no model-opaque hash reference.
            payload = json.loads(choices['command_hold'])
            if shared_layout:
                state['shared_physical_actions'] = {'command_hold': payload}
                choices['command_hold'] = 'Hold: execute the complete command_hold actions and WM predictions in state.shared_physical_actions.command_hold; obtains the same new RGB/encoders.'
                if context['remaining_observation_budget'] > 0 and context['remaining_consecutive_hold_budget'] > 0:
                    choices['observe'] = 'Observe: execute exactly state.shared_physical_actions.command_hold, the same hold actions and WM predictions; no extra view or assured new information.'
            else:
                choices['command_hold'] = json.dumps(payload, separators=(',', ':'))
                if context['remaining_observation_budget'] > 0 and context['remaining_consecutive_hold_budget'] > 0:
                    observe = dict(payload)
                    observe['selection_intent'] = 'observe via identical command_hold'
                    choices['observe'] = json.dumps(observe, separators=(',', ':'))
        choices.update(replan='Re-run VLA once on the actual current observation using registered next sampling seed; compare candidates. No motion by this choice alone.',
                       reject='Stop this finite run without executing this block.')
    else:
        choices.update(observe='Execute a legal command_hold block, advance physics 320ms and obtain newly captured RGB/encoders.',
                       replan='Re-run VLA once on the actual current observation using registered next sampling seed; compare candidates. No motion by this choice alone.',
                       reject='Stop this finite run without executing this block.')
    if consistent_interface(public) and public['decision_context'].get('remaining_replan_budget',1)==0:
        choices.pop('replan',None)
    if public.get('decision_context',{}).get('stage_control_authority')=='terminal_hold':
        state['stage_contract']='Bounded terminal stabilization attempt; terminal_hold is sole allowed 7-channel motion. Continued physical hold is registered task action, not a front-stage stall. reject stops without release; replan cannot override control ownership.'
    instructions = (
        'Choose legal action or finite recovery. WM horizon320ms, not final success. '
        'Necessary finger-object or existing support contact may be allowed; distinguish persistent/new contacts. '
        'Limits or low new-contact scores do not certify safety. WM grasp/loss are estimates, not verified progress. '
        'Public progress is execution/encoders only. No complete; proposals or external gate cannot verify grasp. '
        'Advance task respecting forecasts; repeated holds are not success.'
    ) if consistent_interface(public) else (
        'Choose one legal action or finite recovery for the red-grasp task. WM predicts the NEXT 320ms, not final task success. '
        'Allow necessary finger contact and existing support; distinguish existing and new contact. '
        'No action is guaranteed safe merely because numerical limits pass. Public progress is execution/encoder evidence only. '
        'No complete option; never infer verified grasp from proposals or an external release gate. '
        'Choose actions that advance the task while respecting physical predictions; repeated hold is not task success.'
    )
    presented_state = compact(state)
    if 'shared_physical_actions' in state:
        # Existing WM display precision is unchanged; exact actions must survive
        # recursive state compaction. The full payload is model-visible shared state.
        shared = state['shared_physical_actions']['command_hold']
        presented_state['shared_physical_actions']['command_hold']['actions'] = shared['actions']
        if command_digest(presented_state['shared_physical_actions']['command_hold']['actions']) != shared['command_sha256']:
            raise ValueError('shared Jev matrix differs from forecast/execution')
    return {'model':'NeoHorse-Jev-4B','state':json.dumps(presented_state, separators=(',', ':')),
            'questions':{'selection':{'type':'choice','criteria':choices,'instructions':instructions}}}


def recovery_change(previous, current):
    old = {c['name']: c['command_sha256'] for c in previous}
    new = {c['name']: c['command_sha256'] for c in current}
    return {'changed': old != new, 'changed_names': sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name)),
            'definition': 'exact float32 candidate command payload or legal membership changed; not proof of improved feasibility'}


def execute_with_archive(operation, archive, archive_failure):
    """Always archive the current prefix; never replace an execution exception."""
    try:
        result=operation()
    except BaseException:
        try:
            archive(False)
        except Exception as error:
            archive_failure(error)
        raise
    else:
        archive(True)
        return result


def fatal_suite_error(error):
    return any(marker in str(error) for marker in ('budget_exhausted','maximum_wall_seconds_exhausted',
        'Jev_service_terminated','Jev_response_timeout','Jev IPC response mismatch'))
