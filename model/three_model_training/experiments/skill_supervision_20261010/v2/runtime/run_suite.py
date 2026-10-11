"""Run six finite registered episodes with shared frozen model loading."""
import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

from fusion_core import Budget, atomic, sha, fatal_suite_error
from run_online import main as run_episode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.suite).resolve();plan=json.loads(path.read_text());root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    atomic(root/'suite_plan.json',plan)
    shared={'budget':Budget(plan['budgets']),'start_wall':time.monotonic(),
            'maximum_wall_seconds':plan['maximum_wall_seconds'],'service_root':str(root/'Jev_shared')}
    results=[]
    try:
        for episode in plan['episodes']:
            episode_path=Path(episode['plan']).resolve()
            if sha(episode_path)!=episode['plan_sha256']:
                raise RuntimeError('episode plan changed after suite registration')
            # The separate Jev service spans both fusion branches and uses the suite budget.
            config=json.loads(episode_path.read_text())
            if config['mode']=='fusion' and 'Jev_service' not in shared:
                from fusion_core import atomic as write
                service_plan=dict(config,budgets=dict(config['budgets'],Jev_calls=plan['budgets']['Jev_calls']))
                service_path=root/'shared_service_plan.json';write(service_path,service_plan)
                shared['service_plan_path']=str(service_path)
            output=root/episode['name']
            try:
                run_episode(episode_path,output,shared=shared)
                results.append({'name':episode['name'],'status':'complete','result':json.loads((output/'result.json').read_text())})
            except Exception:
                results.append({'name':episode['name'],'status':'software_interruption','error':traceback.format_exc()})
                # Keep remaining finite branches only for ordinary archived model/execution errors;
                # never continue a hard suite budget overrun or wall-time breach.
                error=results[-1]['error']
                if fatal_suite_error(error):
                    for skipped in plan['episodes'][len(results):]:
                        results.append({'name':skipped['name'],'status':'not_executed_after_fatal_suite_error',
                                        'reason':'shared_budget_deadline_or_Jev_IPC_state_invalid'})
                    atomic(root/'suite_results.json',results)
                    raise
            atomic(root/'suite_results.json',results)
            atomic(root/'suite_status.json',{'stage':'episode_finished','episodes_done':len(results),'counts':shared['budget'].counts})
            if 'VLA' in shared:
                import torch
                torch.mps.empty_cache()
        atomic(root/'suite_report.json',{'stage':'registered_suite_finished','results':results,
                                         'counts':shared['budget'].counts,'hardware_enabled':False,
                                         'optimizer_updates':0,'models_loaded_once':True,
                                         'wall_seconds':time.monotonic()-shared['start_wall']})
    except Exception:
        atomic(root/'suite_failure.json',{'error':traceback.format_exc(),'results':results,'counts':shared['budget'].counts})
        raise
    finally:
        if shared.get('Jev_service') is not None:
            service=shared['Jev_service'];service_root=Path(shared['service_root'])
            atomic(service_root/'ipc/stop.json',{'requested':True})
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.terminate()
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    service.kill();service.wait(timeout=10)
            atomic(root/'Jev_service_exit.json',{'exit_code':service.returncode,
                    'clean_parameter_snapshot_available':(service_root/'Jev_parameters_after.json').exists()})


if __name__=='__main__':
    main()
