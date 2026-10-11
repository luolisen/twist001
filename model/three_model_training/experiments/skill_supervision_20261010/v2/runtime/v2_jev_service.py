"""Actual BF16 native Jev, finite filesystem RPC, no simulation access."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import argparse,json,time,traceback,math
from pathlib import Path
from fusion_core import atomic
from model_binding import snapshot
from skill_model_bridge import NativeSkillJevBridge

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.load(open(a.plan));out=Path(a.output);start=time.monotonic();calls=0
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    from neohorse_decision import DecisionEngine
    from neohorse_decision.vision import VisionDecisionEngine
    from neohorse_decision._vendor.schema import SystemOneRequest,to_record
    engine=DecisionEngine(plan['Jev_bundle'],device='mps')
    vision=VisionDecisionEngine(plan['Jev_bundle'],device='mps',text_engine=engine)
    engine.model.eval().requires_grad_(False)
    before=snapshot(engine.model)
    if before!=json.load(open(plan['prior_BF16_snapshot'])):raise RuntimeError('Jev_frozen_binding_mismatch')
    atomic(out/'Jev_parameters_before.json',before)
    bridge=NativeSkillJevBridge(engine,vision,torch,SystemOneRequest,to_record,lambda:Image.open(out/'image.png').convert('RGB'))
    atomic(out/'service_ready.json',{'profile_internal_stages':False,'endpoint_sync':True,
        'device':'mps','native_state_max':engine.max_state,'native_branch_max':engine.max_branch,
        'initialization_seconds':time.monotonic()-start,'parameters_match_frozen':True})
    for index in range(plan['budgets']['Jev_calls']):
        f=out/f'job_{index:03d}.json'
        while not f.exists():
            if (out/'stop_service').exists() or time.monotonic()-start>=plan['budgets']['wall_seconds']:break
            time.sleep(.1)
        if not f.exists():break
        job=json.load(open(f));request=job['request'];record_path=out/f'Jev_{index:03d}'
        atomic(record_path.with_suffix('.request.json'),request)
        try:
            if job['question']=='supervision':
                response=bridge(request);rec=bridge.last_record;calls+=1
            else:
                tick=time.perf_counter();record,_=to_record(SystemOneRequest(**request))
                encoded=engine.model.encode(engine.tokenizer,record,strict=True,max_state=engine.max_state,max_branch=engine.max_branch)
                if encoded['state_truncated'] or len(encoded['ids'])>engine.max_tokens:raise RuntimeError('target_request_overflow_no_truncation')
                torch.mps.synchronize();calls+=1
                with torch.inference_mode():response=vision.predict(request,Image.open(out/'image.png').convert('RGB'))
                torch.mps.synchronize();answer=response['answers']['target'];scores=answer['probabilities']
                if set(scores)!=set(request['questions']['target']['criteria']) or any(not math.isfinite(float(v)) or not 0<=v<=1 for v in scores.values()) or abs(sum(scores.values())-1)>1e-4 or answer['choice']!=max(scores,key=scores.get):raise RuntimeError('target_return_invalid')
                rec={'request':request,'response':response,'wall_seconds':time.perf_counter()-tick,
                     'input_tokens':len(encoded['ids']),'internal_stage_profiling':False,'endpoint_sync':True}
            atomic(record_path.with_suffix('.record.json'),rec)
            atomic(out/f'reply_{index:03d}.json',{'ok':True,'response':response,'timing':rec['wall_seconds']})
        except Exception as e:
            atomic(out/f'reply_{index:03d}.json',{'ok':False,'error':repr(e),'traceback':traceback.format_exc()})
            break
    after=snapshot(engine.model);atomic(out/'Jev_parameters_after.json',after)
    atomic(out/'service_result.json',{'calls':calls,'parameters_unchanged':before==after,'wall_seconds':time.monotonic()-start})

if __name__=='__main__':main()
