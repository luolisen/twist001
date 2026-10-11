"""Eight selection-only saved-request comparisons, no simulator or VLA/WM call."""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from fusion_core import atomic, sha, selection_request


def prepare(input_root, original_plan, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'ipc').mkdir(); (output / 'public_images').mkdir()
    jobs = [(p, json.loads(p.read_text())) for p in sorted((input_root / 'ipc').glob('job_*.json'))]
    sequence = []
    for wanted in ('full_fusion_half_decision_000', 'full_fusion_half_decision_001'):
        position = next(i for i, (_, job) in enumerate(jobs) if job['public']['decision_id'] == wanted)
        source, job = jobs[position]
        answer_path = source.with_name(source.name.replace('job_', 'answer_'))
        fixed_target = json.loads(answer_path.read_text())['answers']['target']
        prior = [json.loads(p.with_name(p.name.replace('job_', 'answer_')).read_text())['answers']['selection']['choice']
                 for p, j in jobs[:position] if j['public']['decision_id'].startswith('full_fusion_half_')]
        consecutive = 0
        for choice in reversed(prior):
            if choice not in ('observe', 'pose_hold', 'command_hold'): break
            consecutive += 1
        context = {'interface_revision': 'consistent_v1', 'presentation_revision':'shared_full_hold_v1', 'consecutive_hold_count': consecutive,
                   'remaining_consecutive_hold_budget': original_plan['maximum_consecutive_hold_blocks'] - consecutive,
                   'remaining_observation_budget': original_plan['budgets']['observations'] - prior.count('observe'),
                   'history_source': 'captured_rgb_and_joint_encoders', 'history_valid_mask': job['public']['history_valid']}
        image_src = input_root / job['image']
        if sha(image_src) != job['image_sha256']: raise ValueError('saved image hash mismatch')
        image_dst = output / job['image']; shutil.copy2(image_src, image_dst)
        for ordinal, mode in enumerate(('baseline','consistent_v1','consistent_v1','baseline')):
            prepared = copy.deepcopy(job)
            if mode == 'consistent_v1': prepared['public']['decision_context'] = context
            prepared['mode'] = 'selection_only'; prepared['fixed_target'] = fixed_target
            prepared['comparison_metadata'] = {'case_id': wanted, 'interface': mode, 'within_case_ordinal': ordinal,
                 'ordering': 'ABBA', 'source_job_sha256': sha(source), 'source_answer_sha256': sha(answer_path),
                 'target_fixed_from_archive': True, 'physics_steps': 0,
                 'warmth_note': 'first service prediction is separately labelled; ABBA balances order, does not provide two cold starts'}
            request = selection_request(prepared['public'], fixed_target)
            prepared['comparison_metadata']['selection_request_sha256'] = __import__('hashlib').sha256(
                json.dumps(request,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
            sequence.append(prepared)
    atomic(output / 'prepared_saved_requests.json', {'jobs': sequence, 'Jev_selection_calls': len(sequence),
        'no_training_no_physics_no_model_change': True})
    return sequence


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-root', required=True);p.add_argument('--original-plan',required=True)
    p.add_argument('--service-plan',required=True);p.add_argument('--output',required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--maximum-wall-seconds',type=int,default=1800)
    p.add_argument('--skip-completed-count',type=int,choices=(0,1),default=0)
    p.add_argument('--completed-prefix-root');args=p.parse_args()
    output=Path(args.output); jobs=prepare(Path(args.input_root),json.loads(Path(args.original_plan).read_text()),output)
    if args.skip_completed_count:
        if not args.completed_prefix_root: raise ValueError('completed prefix evidence required')
        prior=Path(args.completed_prefix_root)
        trace=json.loads((prior/'Jev_request_trace.json').read_text())['records']
        if len(trace)!=1 or trace[0]['question']!='selection': raise ValueError('exactly one registered completed selection required')
        expected=selection_request(jobs[0]['public'],jobs[0]['fixed_target'])
        if trace[0]['request']!=expected: raise ValueError('completed baseline request differs')
        answer=prior/'ipc/answer_000.json'
        if not answer.exists() or json.loads(answer.read_text())['real_Jev_calls']!=1: raise ValueError('missing actual completed baseline receipt')
        archived={'skip_completed_count':1,'trace_path':str(prior/'Jev_request_trace.json'),
                  'trace_sha256':sha(prior/'Jev_request_trace.json'),'answer_path':str(answer),'answer_sha256':sha(answer),
                  'completed_original_sequence_ordinal':0,'selection':trace[0]['answer']['answers']['selection']}
        atomic(output/'completed_prefix_reference.json',archived)
        jobs=jobs[1:]
        for original_index,job in enumerate(jobs,start=1):
            job['comparison_metadata']['original_sequence_ordinal']=original_index
            job['comparison_metadata']['completed_prefix_reference']=archived
    if args.prepare_only: return
    service_plan=json.loads(Path(args.service_plan).read_text());service_plan['budgets']=dict(service_plan['budgets'],Jev_calls=len(jobs))
    atomic(output/'service_plan.json',service_plan)
    started=time.monotonic();process=None
    with (output/'service.stdout.log').open('w') as stdout,(output/'service.stderr.log').open('w') as stderr:
        process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('jev_online_service.py')),
                '--plan',str(output/'service_plan.json'),'--output',str(output)],stdout=stdout,stderr=stderr)
        try:
            def wait(path):
                while not path.exists():
                    if process.poll() is not None: raise RuntimeError('Jev service exited: '+str(process.returncode))
                    if time.monotonic()-started > args.maximum_wall_seconds: raise RuntimeError('saved comparison wall budget exhausted')
                    time.sleep(.2)
            wait(output/'jev_service_status.json')
            status=json.loads((output/'jev_service_status.json').read_text())
            if status['stage'] != 'ready': raise RuntimeError('Jev service not ready')
            for i,job in enumerate(jobs):
                atomic(output/f'ipc/job_{i:03d}.json',job);wait(output/f'ipc/answer_{i:03d}.json')
                answer=json.loads((output/f'ipc/answer_{i:03d}.json').read_text())
                if answer['real_Jev_calls'] != 1: raise RuntimeError('unexpected target or extra model call')
            atomic(output/'ipc/stop.json',{'reason':'registered remaining selections complete'})
            process.wait(timeout=180)
            if process.returncode: raise RuntimeError('Jev service failed on final snapshot')
            atomic(output/'comparison_result.json',{'stage':'complete','selection_calls_this_process':len(jobs),'selection_calls_total_including_archived_prefix':len(jobs)+args.skip_completed_count,'physics_steps':0,
                    'wall_seconds':time.monotonic()-started,'note':'development decision interface comparison; no physical outcome or success claim'})
        except BaseException as error:
            atomic(output/'comparison_result.json',{'stage':'interrupted','error':str(error),'physics_steps':0,
                    'wall_seconds':time.monotonic()-started})
            atomic(output/'ipc/stop.json',{'reason':'comparison interrupted'})
            if process.poll() is None:
                try:process.wait(timeout=180)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
            raise

if __name__ == '__main__':main()
