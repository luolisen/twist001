"""Offline archive checks only: no inference, MuJoCo or physical execution."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)

def main():
    manifest=read('source_provenance.json')
    for item in manifest['files']:
        p=ROOT/item['published']
        require(hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256'],f"published fingerprint: {p}")
        if p.suffix=='.py':
            ast.parse(p.read_text(),filename=str(p))
            require(item['source_sha256']==item['published_sha256'],f"source modified: {p}")
    v1=read('evidence/v1_result.json')
    require(v1['physics_steps']==10880 and v1['blocks']==34,'v1 execution scope')
    require(v1['original_8_320']['obtained'] and not v1['original_8_320']['task_completion'],'8.320 endpoint must remain failed')
    require(v1['extended_10_880']['task_completion'],'10.880 archived pass')
    require(not v1['supervision_benefit_established'],'no unsupported benefit claim')
    ff=read('evidence/feedforward_comparison.json')
    end=ff['comparison'][-1]
    require(ff['steps']==640 and ff['attempts']==1 and ff['all_steps_dual_finger_without_other_support'],'local retention scope')
    require(end['position_error_mm']<=2.5 and end['orientation_error_rad']<=.05,'original local tolerances')
    improvement=1-end['position_error_mm']/end['baseline_position_error_mm']
    require(.93<improvement<.95,'reported improvement')
    full=np.load(ROOT/'long_path/candidate_0_80mm_nominal.npz')['commands']
    prefix=np.load(ROOT/'long_path/static_legal_prefix_nominal.npz')['commands']
    proposed=np.load(ROOT/'long_path/proposed_70mm_nominal_136_commands.npz')['commands']
    require(full.shape==(160,7) and prefix.shape==(128,7) and proposed.shape==(136,7),'exact path shapes')
    require(full.dtype==np.float32 and np.array_equal(prefix,full[:128]),'prefix source binding')
    require(np.array_equal(proposed[:128],prefix) and np.all(proposed[128:]==prefix[-1]),'fixed nominal response tail')
    contract=read('evidence/frozen_feedforward_contract.json')
    limits=np.array(contract['mechanical_ranges']);control=np.array(contract['control_ranges'])
    require(np.isfinite(prefix).all() and np.all(prefix>=control[:,0]) and np.all(prefix<=control[:,1]),'control ranges')
    require(np.all(prefix[:,:6]>=limits[:,0]) and np.all(prefix[:,:6]<=limits[:,1]),'actual mechanical ranges')
    require(np.all(prefix[:,6]==np.float32(contract['clamp_target_m'])),'retained clamp')
    jumps=np.abs(np.diff(np.vstack([contract['initial_q_rad'],prefix[:,:6]]),axis=0))
    require(jumps.max()<=.02+1e-8,'command continuity including block boundaries')
    rows=read('long_path/command_audit.json');samples=read('long_path/static_sample_audit.json')
    require(all(not x['command_failures'] and not x['path_failures'] and not x['servo_proxy_geometry_failures'] for x in rows[:128]),'archived nominal/proxy checks')
    require(np.array_equal(np.array([r['nominal'] for r in rows],np.float32),full),'audited actual nominal matrix')
    require(len([s for s in samples if s['command_index']<128])==640,'prefix static coverage')
    require(rows[132]['nominal'][4]>limits[4,1] and rows[132]['command_failures'],'first blocked command')
    plan=read('long_path/next_local_simulation_registration.json')
    require(not plan['physical_execution_authorized'] and plan['maximum_physics_steps']==5440,'proposal is not execution approval')
    sys.path.insert(0,str(ROOT/'v2/runtime'))
    from public_association import camera_poses
    cal=read('long_path/public_calibration.json')
    pose=camera_poses({'cameras':[cal['ee_reference']]},prefix[-1,:6])[0]
    expected=read('long_path/path_result.json')['legal_prefix_actual_static_ee_world_m']
    require(np.linalg.norm(pose[:3,3]-expected)<1e-9,'published FK endpoint')
    cov=read('public_evidence/coverage_summary.json')
    require(cov['carry_verdict']=='unknown' and not cov['automatic_transport_enabled'] and not cov['automatic_release_enabled'],'unknown evidence must not grant control')
    print(json.dumps({'archive_integrity_passed':True,'published_files':len(manifest['files']),
        'legal_prefix_mm':70,'candidate80mm_rejected':True,'local_feedforward_error_mm':end['position_error_mm'],
        'new_physics_steps':0,'new_model_calls':0,'new_IK_calls':0},ensure_ascii=False))

if __name__=='__main__':main()
