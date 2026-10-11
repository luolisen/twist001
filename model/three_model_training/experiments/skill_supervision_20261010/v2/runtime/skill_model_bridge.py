"""Actual frozen-model invocation interfaces; constructed with registered instances.

No loading, model substitution or saved forecast fallback. These are not exercised
by the nominated local controller diagnostic, which stops before physical motion.
"""
import time,math
import numpy as np
from fusion_core import history_arrays,CONTACT_TYPES,command_digest
from wm_motion_adapter import adapt_motion

class FrozenWMBridge:
    def __init__(self,wm,torch,history_provider,calibration):
        self.model=wm;self.torch=torch;self.history=history_provider;self.calibration=calibration;self.calls=0
    def __call__(self,commands,observation):
        commands=np.asarray(commands,np.float32)
        if commands.shape!=(8,7) or not np.isfinite(commands).all():raise ValueError('invalid_final_commands')
        images,states,valid,times=history_arrays(self.history())
        if abs(times[-1]-observation['time'])>1e-9 or not np.array_equal(states[-1],observation['state']):raise RuntimeError('WM_history_not_current')
        torch=self.torch;self.calls+=1;torch.mps.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            out=self.model(*[torch.from_numpy(x).unsqueeze(0).to('mps') for x in (images,states,valid,commands)])
        if not all(bool(torch.isfinite(v).all()) for v in out.values()):raise RuntimeError('nonfinite_WM_output')
        probs=out['events'].sigmoid().detach().cpu().numpy()[0]
        raw=out['motion'].detach().cpu().numpy()[0]
        if probs.shape!=(29,) or raw.shape!=(10,):raise RuntimeError('WM_output_definition_mismatch')
        if np.any(probs[9:18]>probs[:9]+1e-6) or np.any(probs[18:27]>probs[:9]+1e-6):raise RuntimeError('WM_contact_hierarchy_invalid')
        motion,archive=adapt_motion(states[-1],raw,self.calibration)
        torch.mps.synchronize()
        return {'legal_forecast':True,'command_sha256':command_digest(commands),'actions':commands.tolist(),
                'observation_time_s':observation['time'],'motion_delta':motion,'raw_motion_and_FK_archive':archive,
                'physical_contacts':dict(zip(CONTACT_TYPES,probs[:9].tolist())),
                'new_contacts':dict(zip(CONTACT_TYPES,probs[9:18].tolist())),
                'unlocalized_contacts':dict(zip(CONTACT_TYPES,probs[18:27].tolist())),
                'any_grasp_estimate':float(probs[27]),'grip_loss_estimate':float(probs[28]),
                'wall_seconds':time.perf_counter()-start,'forecast_horizon_s':.32,'not_calibrated_safety':True}

class NativeSkillJevBridge:
    def __init__(self,engine,vision,torch,SystemOneRequest,to_record,image_provider):
        self.engine=engine;self.vision=vision;self.torch=torch;self.request_type=SystemOneRequest
        self.to_record=to_record;self.image=image_provider;self.calls=0;self.last_record=None
    def __call__(self,request):
        start=time.perf_counter();e=self.engine
        record,_=self.to_record(self.request_type(**request))
        encoded=e.model.encode(e.tokenizer,record,strict=True,max_state=e.max_state,max_branch=e.max_branch)
        if encoded['state_truncated'] or len(encoded['ids'])>e.max_tokens:raise RuntimeError('native_skill_request_overflow_no_truncation')
        self.torch.mps.synchronize();self.calls+=1
        with self.torch.inference_mode():answer=self.vision.predict(request,self.image())
        self.torch.mps.synchronize()
        a=answer['answers']['supervision'];scores=a['probabilities'];choices=request['questions']['supervision']['criteria']
        if set(scores)!=set(choices) or any(not math.isfinite(float(v)) or not 0<=v<=1 for v in scores.values()):raise RuntimeError('native_Jev_option_mismatch')
        if abs(sum(scores.values())-1)>=1e-4 or a['choice']!=max(scores,key=scores.get):raise RuntimeError('native_Jev_choice_mismatch')
        self.last_record={'request':request,'response':answer,'wall_seconds':time.perf_counter()-start,
                          'input_tokens':len(encoded['ids']),'internal_stage_profiling':False,'endpoint_sync':True}
        return answer
