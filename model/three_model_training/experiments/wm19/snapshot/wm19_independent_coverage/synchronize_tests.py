"""New preregistered physical fixtures with actual causal observation histories."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parent;V7=ROOT.parent/'wm_consistent_v7'
sys.path.insert(0,str(V7))
from physical_labels import np,mujoco,setup,sha,atomic,render,public_motion,snap,restore,segmentation_labels,geometry,physical_outcome,CONTACT_TYPES

def main():
    plan=json.loads((ROOT/'execution_plan.json').read_text());assert not plan['training_auto_start']
    base=json.loads((ROOT/'new_fixtures/coverage_manifest.json').read_text());records=[];counts=np.zeros(9,np.int64);hidden=np.zeros(9,np.int64)
    out=ROOT/'synchronized_tests';out.mkdir(exist_ok=False)
    private=ROOT/'physical_private_audit';private.mkdir(exist_ok=False)
    for i,row in enumerate(base['records']):
        path=ROOT/'new_fixtures'/row['cache'];assert sha(path)==row['sha256']
        with np.load(path) as a:baseline=a['integration_state'].copy();commands=a['actions'].copy()
        m,d,ix=setup(Path(row['scene']));restore(m,d,baseline);m.light_diffuse[0]=np.array([.5]*3)*np.random.default_rng(row['seed']).uniform(.85,1.15)
        renderer=mujoco.Renderer(m,360,640);history=[];states=[]
        # Observations come from forward simulation, never reverse reconstruction.
        for frame in range(3):
            mujoco.mj_forward(m,d);history.append(render(m,d,renderer)[1]);states.append(public_motion(m,d,ix))
            if frame<2:
                d.ctrl[ix.aids]=d.ctrl[ix.aids].copy()
                for _ in range(40):mujoco.mj_step(m,d)
        images=np.stack(history);motor=np.stack(states);ids=segmentation_labels(m,d,renderer);now=snap(m,d)
        truth=physical_outcome(m,d,ix,geometry(m),commands,renderer,ids);counts+=truth['target_contacts'];hidden+=truth['target_unlocalized_contacts']
        destination=out/f'{i:03d}.npz';np.savez_compressed(destination,history_images=images,history_state=motor,history_valid=np.ones(3,np.uint8),history_relative_times_s=np.array([-.08,-.04,0],np.float32),actions=commands,**truth)
        private_path=private/f'{i:03d}.npz';np.savez_compressed(private_path,integration_state=now)
        records.append(dict(row,private_audit_cache=str(private_path.relative_to(ROOT)),private_audit_sha256=sha(private_path),cache=str(destination.relative_to(ROOT)),sha256=sha(destination),source_cache=str(path),source_sha256=row['sha256'],public_history_generated_causally=True,physical_truth_after_history=True))
        renderer.close();atomic(ROOT/'test_status.json',dict(stage='wm13_causal_contact_coverage_synchronization',cases_done=i+1,cases_total=len(base['records']),compute_host='M2 Max',pid=os.getpid(),training_started=False))
        if (i+1)%12==0:print('synchronized cases',i+1,flush=True)
    critical=('object_nonfinger','object_object','object_tray_wall','robot_tray','robot_floor')
    coverage={k:dict(positives=int(counts[CONTACT_TYPES.index(k)]),negatives=int(len(records)-counts[CONTACT_TYPES.index(k)]),unlocalized_positive_labels=int(hidden[CONTACT_TYPES.index(k)])) for k in critical}
    missing=[k for k,v in coverage.items() if v['positives']<3 or v['negatives']<3]
    atomic(ROOT/'synchronized_test_manifest.json',dict(records=records,coverage=coverage,coverage_gate_passed=not missing,insufficient_classes=missing,history_validity_complete=True,no_resampling=True,fitting_allowed=False,construction_failures=base['failures'],execution_plan_sha256=sha(ROOT/'execution_plan.json'),scope='New fixed physical fixtures after 80 ms observed simulation. Their coverage and visibility are reported, not silently repaired or treated as full policy task success.',hardware_enabled=False,full_task_acceptance_passed=False))
    for n,h in plan['frozen_evidence_sha256'].items():assert sha(ROOT.parent/n)==h,n
    atomic(ROOT/'report.json',dict(stage='wm13_causal_contact_coverage_complete',data_preparation_only=True,test_recipe="frozen_wm11",new_synchronized_test_cases=len(records),new_test_coverage=coverage,new_test_coverage_gate_passed=not missing,insufficient_classes=missing,prototype_trained=False,training_started=False,old_evidence_unchanged=True,compute_host='M2 Max',hardware_enabled=False,full_task_acceptance_passed=False,next='Review actual synchronized coverage and unlocalized cases. Register a separate bounded candidate only after review of all required gates; no fit or WM scoring in this worker.'))

if __name__=='__main__':main()
