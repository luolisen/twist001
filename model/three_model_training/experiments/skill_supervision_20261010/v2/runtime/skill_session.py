"""High-level integration with actual caller-supplied WM/Jev/execution services.

No model substitutes are installed here. Without current WM and native Jev the
online path remains closed. Local nominations use the separate diagnostic entry.
"""
import time
from skill_request import skill_request
from skill_execution import SkillExecution,wm_binding
from skill_router import digest

class SkillSession:
    def __init__(self,router,observe,wm_predict,jev_native,executor,archive,
                 max_physics_s=30.,max_wall_s=1800.,max_Jev=12):
        self.router=router;self.observe=observe;self.wm_predict=wm_predict
        self.jev=jev_native;self.executor=executor;self.archive=archive
        self.gate=SkillExecution(router);self.start_wall=time.monotonic();self.start_physics=None
        self.physical_cap=max_physics_s;self.wall_cap=max_wall_s;self.Jev_cap=max_Jev
        self.counts={'WM':0,'Jev':0,'executed_physics_steps':0}
    def bounded_observation(self):
        o,c=self.observe()
        if self.start_physics is None:self.start_physics=o['time']
        if time.monotonic()-self.start_wall>=self.wall_cap or o['time']-self.start_physics>=self.physical_cap:
            self.router.cancel('session_budget_exhausted');raise RuntimeError('session_budget_exhausted')
        return o,c
    def forecast(self,commands,o):
        if self.wm_predict is None:raise RuntimeError('real_WM_service_missing')
        self.counts['WM']+=1;f=self.wm_predict(commands,o)
        binding=wm_binding(commands,o['time'],f)
        self.archive('WM_on_final_commands',{'binding':binding,'forecast':f})
        if not binding['legal_forecast']:
            self.router.revoke('WM_illegal_forecast');raise RuntimeError('WM_illegal_forecast')
        return f,binding
    def authorize(self,skill,commands,base_request,public_evidence):
        o,c=self.bounded_observation();event=self.router.request(skill,c,o['time'],time.monotonic())
        if self.jev is None or self.counts['Jev']>=self.Jev_cap:
            self.router.cancel('native_Jev_missing_or_budget_exhausted');return False
        f,b=self.forecast(commands,o);request=skill_request(base_request,event,f,public_evidence)
        self.archive('actual_skill_request',request);self.counts['Jev']+=1
        try:answer=self.jev(request)
        except Exception:
            self.router.cancel('native_Jev_timeout_or_service_failure');raise
        self.archive('native_Jev_response',answer)
        # Caller must perform native token encoding and bounded service wait. No
        # fallback approval is generated for timeout/invalid answers.
        try:choice=answer['answers']['supervision']['choice']
        except (KeyError,TypeError):
            self.router.cancel('invalid_native_Jev_response');return False
        now,context=self.bounded_observation()
        if abs(now['time']-o['time'])>1e-9:
            self.router.cancel('physics_advanced_during_simulation_authorization');return False
        granted=self.router.decide(event['id'],choice,context,now['time'],time.monotonic())
        return granted
    def trajectory_block(self,generator,action_id):
        o,c=self.bounded_observation();commands=generator.preview()
        if o['time']+.32-self.start_physics>self.physical_cap+1e-9:
            self.router.cancel('whole_block_exceeds_session_budget');raise RuntimeError('whole_block_exceeds_session_budget')
        f,b=self.forecast(commands,o)
        self.gate.start_trajectory(generator,action_id,c,o['time'],time.monotonic(),b)
        receipt=self.executor(commands,action_id)
        self.counts['executed_physics_steps']+=receipt['steps']
        self.archive('actual_execution_receipt',receipt)
        if receipt.get('action_id')!=action_id or receipt.get('command_sha256')!=digest(commands):
            self.router.revoke('actual_control_receipt_mismatch');raise RuntimeError('actual_control_receipt_mismatch')
        return self.gate.settle_trajectory(generator,receipt['id'],action_id,commands,receipt['steps'],receipt['end_time_s'])
