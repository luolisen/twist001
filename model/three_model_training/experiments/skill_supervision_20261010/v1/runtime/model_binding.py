"""Register and verify a complete frozen SmolVLA bundle, not just its pathname."""
import json
from pathlib import Path

from fusion_core import sha

REQUIRED = ('config.json', 'model.safetensors', 'policy_preprocessor.json',
            'policy_postprocessor.json', 'policy_preprocessor_step_5_normalizer_processor.safetensors',
            'policy_postprocessor_step_0_unnormalizer_processor.safetensors')
VLM_SOURCE_SUFFIXES={'.json','.txt','.model','.jinja','.py'}
VLM_SOURCE_REQUIRED={'config.json','tokenizer_config.json','tokenizer.json'}


def vlm_source_files(directory):
    root=Path(directory)
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in VLM_SOURCE_SUFFIXES
                  and not any(part.startswith('.') for part in p.relative_to(root).parts))


def register_vlm_sources(binding):
    """Run on the actual model host BEFORE freezing the plan, never at inference."""
    root=Path(binding['VLM_source'])
    files=vlm_source_files(root)
    names={str(p.relative_to(root)) for p in files}
    if not VLM_SOURCE_REQUIRED<=names:
        raise ValueError('actual VLM/tokenizer source metadata incomplete')
    result=dict(binding)
    result['VLM_files_sha256']={str(p):sha(p) for p in files}
    result['VLM_source_inventory']={'root':str(root),'metadata_paths':sorted(names),
        'purpose':'lock actual local AutoConfig/AutoProcessor/AutoTokenizer sources and related metadata',
        'base_VLM_weights_loaded':False,'weight_files_excluded':'load_vlm_weights=False; checkpoint model.safetensors is separately pinned'}
    return result


def from_prior_plan(plan, checkpoint_label='recent_native'):
    checkpoint = Path(plan['checkpoints'][checkpoint_label])
    known = plan['frozen_files']
    paths = {str(checkpoint / name): known[str(checkpoint / name)] for name in REQUIRED}
    return {'checkpoint_path': str(checkpoint), 'files_sha256': paths,
            'VLM_files_sha256':{p:h for p,h in known.items() if p.startswith(plan['VLM_source']+'/')
                                and Path(p).suffix in VLM_SOURCE_SUFFIXES},
            'VLM_source': plan['VLM_source'], 'device': 'mps',
            'input_mapping': {'state': 'six_joint_radians_plus_measured_gap_m_divided_by_0.08',
                              'RGB': 'original_public_front_256_and_wrist_512_letterbox',
                              'task': 'registered_literal'},
            'output_mapping': {'joints': 'absolute_position_radians',
                               'gap': 'postprocessed_unit_interval_clipped_to_0_1_then_times_0.08m',
                               'prefix': 'first_8_of_50', 'command_dtype': 'float32'},
            'preprocessor_overrides': {'device_processor': {'device': 'mps'},
                                      'tokenizer_processor': {'tokenizer_name': plan['VLM_source']}},
            'research_binding_only': True, 'autonomous_acceptance_promoted': False}


def verify_binding(binding):
    root = Path(binding['checkpoint_path'])
    files = binding['files_sha256']
    for name in REQUIRED:
        if str(root / name) not in files:
            raise ValueError('incomplete checkpoint registration: ' + name)
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('checkpoint fingerprint differs: ' + path)
    vlm_root=Path(binding['VLM_source'])
    expected_overrides={'device_processor':{'device':binding['device']},
                        'tokenizer_processor':{'tokenizer_name':binding['VLM_source']}}
    if binding['preprocessor_overrides']!=expected_overrides:
        raise ValueError('actual preprocessor tokenizer/device override differs from pinned source')
    vlm_files=binding.get('VLM_files_sha256',{})
    actual={str(p) for p in vlm_source_files(vlm_root)}
    if not VLM_SOURCE_REQUIRED <= {str(Path(p).relative_to(vlm_root)) for p in vlm_files}:
        raise ValueError('VLM/tokenizer/processor source fingerprints missing')
    if actual!=set(vlm_files):
        raise ValueError('actual VLM source metadata inventory differs from registration')
    for path,digest in vlm_files.items():
        if sha(path)!=digest:
            raise ValueError('VLM/tokenizer/processor source fingerprint differs: '+path)
    config = json.loads((root / 'config.json').read_text())
    expected_shapes = {'observation.state': [7], 'observation.images.global': [3,256,256],
                       'observation.images.wrist': [3,512,512]}
    for feature, shape in expected_shapes.items():
        if config['input_features'][feature]['shape'] != shape:
            raise ValueError('VLA input feature changed: ' + feature)
    if config['output_features']['action']['shape'] != [7]:
        raise ValueError('VLA output feature changed')
    traces = {}
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        processor = json.loads((root / filename).read_text())
        for step in processor['steps']:
            if 'state_file' in step:
                state_path = str(root / step['state_file'])
                if state_path not in files:
                    raise ValueError('unregistered processor normalization state')
        traces[filename] = processor
    return {'files_verified': len(files), 'VLM_source_files_verified':len(vlm_files), 'processors': traces,
            'config': config, 'binding': binding}


def load_vla(binding):
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy
    verified = verify_binding(binding)
    path = Path(binding['checkpoint_path'])
    config = PreTrainedConfig.from_pretrained(path)
    config.device = binding['device']
    config.vlm_model_name = binding['VLM_source']
    config.load_vlm_weights = False
    model = SmolVLAPolicy.from_pretrained(path, config=config, strict=True).eval().requires_grad_(False)
    pre, post = make_pre_post_processors(config, path, preprocessor_overrides=binding['preprocessor_overrides'])
    return model, pre, post, verified


def snapshot(model):
    import hashlib
    import torch
    return {name: {'shape': list(p.shape), 'dtype': str(p.dtype),
                   'sha256': hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
            for name, p in model.named_parameters()}
