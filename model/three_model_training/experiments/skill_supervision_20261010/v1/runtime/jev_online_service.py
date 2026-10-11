"""Frozen BF16 Jev IPC; dynamic legal candidates and public-only progress."""
import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
from fusion_core import atomic, sha, target_request, selection_request
from model_binding import snapshot
from decision_representation import revised_selection
from task_relative import revised_request
from stage_request import stage_request


class StageProfiler:
    """Common synchronized instrumentation for both interface conditions.

    Uses actual callable boundaries confirmed in frozen vision.py, not estimated
    GPU substage durations. Fences affect timing; compare instrumented peers only.
    """
    def __init__(self, engine, vision, torch):
        self.torch = torch
        self.values = {}
        self.active = False
        self.handles = []
        self.original_encode = engine.model.encode
        profiler = self
        def encode(*args, **kwargs):
            if not profiler.active: return profiler.original_encode(*args, **kwargs)
            started = time.perf_counter()
            result = profiler.original_encode(*args, **kwargs)
            profiler.add('internal_text_token_encoding', time.perf_counter()-started)
            return result
        engine.model.encode = encode
        processor = vision.processor
        class ProcessorProxy:
            def __getattr__(self, name): return getattr(processor, name)
            def __call__(self, *args, **kwargs):
                started = time.perf_counter()
                result = processor(*args, **kwargs)
                profiler.add('image_preprocessing', time.perf_counter()-started)
                return result
        vision.processor = ProcessorProxy()
        self.instrument(vision.wrapper, 'multimodal_forward_inclusive')
        self.instrument(vision.wrapper.visual, 'vision_encoder_forward')
        if hasattr(vision.wrapper, 'language_model'):
            self.instrument(vision.wrapper.language_model, 'language_model_forward')
        self.instrument(engine.model.head, 'decision_head_forward')

    def add(self, key, value):
        if self.active: self.values[key] = self.values.get(key, 0.) + value

    def instrument(self, module, key):
        starts=[]
        def pre(module, arguments):
            if self.active:
                sync=time.perf_counter(); self.torch.mps.synchronize()
                self.add('profiling_boundary_sync',time.perf_counter()-sync)
                starts.append(time.perf_counter())
        def post(module, arguments, result):
            if self.active:
                self.torch.mps.synchronize()
                self.add(key,time.perf_counter()-starts.pop())
        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def start(self): self.values={};self.active=True
    def finish(self): self.active=False;return dict(self.values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    plan = json.loads(Path(args.plan).read_text())
    import torch
    from PIL import Image
    from neohorse_decision import DecisionEngine
    from neohorse_decision.vision import VisionDecisionEngine
    from neohorse_decision._vendor.schema import SystemOneRequest, to_record
    engine = DecisionEngine(plan['Jev_bundle'], device='mps')
    vision = VisionDecisionEngine(plan['Jev_bundle'], device='mps', text_engine=engine)
    engine.model.eval().requires_grad_(False)
    if not all(p.dtype == torch.bfloat16 for p in engine.model.multimodal.parameters()):
        raise RuntimeError('Jev multimodal dtype is not frozen BF16')
    if not all(p.dtype == torch.float32 for p in engine.model.head.parameters()):
        raise RuntimeError('Jev decision head dtype is not float32')
    before = snapshot(engine.model)
    if before != json.loads(Path(plan['prior_BF16_snapshot']).read_text()):
        raise RuntimeError('Jev parameters differ from frozen snapshot')
    atomic(root / 'Jev_parameters_before.json', before)
    profiler = StageProfiler(engine, vision, torch) if plan.get('profile_internal_stages', True) else None
    atomic(root / 'effective_service_configuration.json', {'pid': os.getpid(), 'plan_file': str(Path(args.plan).resolve()), 'plan_sha256': sha(args.plan), 'profile_internal_stages': bool(profiler), 'request_endpoint_sync': True, 'presentation': plan.get('dynamic_task_relation'), 'phase_evidence_integrated':True, 'Jev_role':'finite_phase_supervision', 'device': 'mps', 'threads': torch.get_num_threads()})
    calibration = json.loads(Path(plan['public_relation_calibration_file']).read_text()) if plan.get('dynamic_task_relation') else None
    rows, index = [], 0
    atomic(root / 'jev_service_status.json', {'stage': 'ready', 'requests_done': 0})
    while not (root / 'ipc/stop.json').exists():
        jobfile = root / f'ipc/job_{index:03d}.json'
        if not jobfile.exists():
            time.sleep(.1)
            continue
        job = json.loads(jobfile.read_text())
        imagepath = root / job['image']
        if imagepath.parent != root / 'public_images' or sha(imagepath) != job['image_sha256']:
            raise RuntimeError('IPC RGB path/hash mismatch')
        answers = {}
        image_started = time.perf_counter()
        with Image.open(imagepath) as opened:
            image = opened.copy()
        image_decode_seconds = time.perf_counter() - image_started
        kinds = ('supervision',) if job.get('mode') == 'supervision_only' else ('target', 'supervision') if job.get('mode') == 'phase_event' else ('selection',) if job.get('mode') == 'selection_only' else ('target', 'selection')
        if kinds in (('selection',),('supervision',)):
            answers['target'] = job['fixed_target']
        with image:
            for kind in kinds:
                if len(rows) >= plan['budgets']['Jev_calls']:
                    raise RuntimeError('Jev service call budget exhausted')
                build_started = time.perf_counter()
                request = target_request(job['public']) if kind == 'target' else selection_request(job['public'], answers['target'])
                if kind in ('selection','supervision') and calibration is not None:
                    if 'task_relative_observation' not in job:
                        raise RuntimeError('dynamic public relation missing from current observation')
                    request = revised_request(revised_selection(job['public'], answers['target']), job['public'], job['task_relative_observation'], calibration)
                    motion_mode=job.get('wm_motion_postprocess',plan.get('wm_motion_postprocess','direct'))
                    if motion_mode not in ('direct','joint_fk'):raise RuntimeError('unregistered_WM_motion_postprocess')
                    if motion_mode=='joint_fk':
                        from wm_motion_adapter import annotate_request
                        request=annotate_request(request)
                if kind == 'supervision':
                    request = stage_request(request, job['phase_event'])
                request_build_seconds = time.perf_counter() - build_started
                atomic(root / 'Jev_preencode' / f'{len(rows):03d}.json', {'decision_id':job['public']['decision_id'],'question':kind,'request':request,'image_sha256':job['image_sha256']})
                encoding_started = time.perf_counter()
                record, _ = to_record(SystemOneRequest(**request))
                encoded = engine.model.encode(engine.tokenizer, record, strict=True, max_state=engine.max_state, max_branch=engine.max_branch)
                if encoded['state_truncated'] or len(encoded['ids']) > engine.max_tokens:
                    raise RuntimeError('Jev input exceeds frozen context; no silent truncation')
                validation_seconds = time.perf_counter() - encoding_started
                torch.mps.synchronize()
                start = time.perf_counter()
                if profiler: profiler.start()
                with torch.inference_mode():
                    response = vision.predict(request, image)
                internals = profiler.finish() if profiler else {}
                predict_return = time.perf_counter()
                torch.mps.synchronize()
                sync_finished = time.perf_counter()
                answer = response['answers'][kind]
                probabilities = answer['probabilities']
                choices = request['questions'][kind]['criteria']
                if set(probabilities) != set(choices) or any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in probabilities.values()):
                    raise RuntimeError('Jev response option/probability mismatch')
                if abs(sum(probabilities.values()) - 1) >= 1e-4 or answer['choice'] != max(probabilities, key=probabilities.get):
                    raise RuntimeError('Jev response normalization/choice mismatch')
                answers[kind] = answer
                rows.append({'decision_id': job['public']['decision_id'], 'question': kind, 'request': request,
                             'answer': response, 'image_sha256': job['image_sha256'], 'seconds': sync_finished - start,
                             'request_wall_seconds_excluding_shared_image_decode': sync_finished - build_started,
                             'timings_seconds': {'image_decode': image_decode_seconds, 'request_build': request_build_seconds,
                                 'token_validation_encode': validation_seconds, 'vision_predict_call': predict_return - start,
                                 'mps_final_sync': sync_finished - predict_return,
                                 'internal_stages': internals,
                                 'vision_encoding_exclusive': internals.get('vision_encoder_forward'),
                                 'text_model_compute_exclusive': internals.get('language_model_forward')},
                             'timing_scope': 'request entrance/return MPS fences retained; internal module hooks active only if explicitly recorded',
                             'profile_internal_stages': bool(profiler), 'dynamic_task_relation': calibration is not None,
                             'text_input_tokens_validation': len(encoded['ids']),
                             'interface_revision': job['public'].get('decision_context', {}).get('interface_revision', 'baseline'),
                             'service_request_ordinal': len(rows), 'service_first_predict': len(rows) == 0,
                             'comparison_metadata': job.get('comparison_metadata')})
                atomic(root / 'Jev_request_trace.json', {'records': rows})
        atomic(root / f'ipc/answer_{index:03d}.json', {'decision_id': job['public']['decision_id'],
                                                   'answers': answers, 'real_Jev_calls': len(kinds)})
        atomic(root / 'jev_service_status.json', {'stage': 'ready', 'requests_done': len(rows)})
        index += 1
    after = snapshot(engine.model)
    atomic(root / 'Jev_parameters_after.json', after)
    if before != after:
        raise RuntimeError('Jev frozen parameters changed')
    atomic(root / 'jev_service_status.json', {'stage': 'complete', 'requests_done': len(rows), 'parameters_unchanged': True})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Also preserve initial-load exceptions; parent supervisor retains stderr.
        if '--output' in __import__('sys').argv:
            root = Path(__import__('sys').argv[__import__('sys').argv.index('--output') + 1])
            atomic(root / 'jev_service_status.json', {'stage': 'failed', 'error': traceback.format_exc()})
        raise
