"""Permission/forecast/execution/commit gate for interchangeable generators."""
from skill_router import digest

class SkillExecution:
    def __init__(self,router,transport_guard=None):
        self.router=router;self.transport_guard=transport_guard
    def start_trajectory(self,generator,action_id,context,t,wall,wm_binding):
        commands=generator.preview()
        if self.router.active=='Transport':
            if self.transport_guard is None:raise RuntimeError('independent_transport_guard_required')
            self.transport_guard.prepare_block(commands,t,wall,context)
        self.router.begin(action_id,commands,self.router.lease['owners'],context,t,wall,wm_binding)
        generator.activate(commands)
        return commands
    def settle_trajectory(self,generator,receipt_id,action_id,commands,steps,end):
        fresh=self.router.receipt(receipt_id,action_id,commands,steps,end)
        if fresh:
            # Record the actual executed prefix even when protection terminates it.
            generator_error=None
            try:generator.receipt(steps)
            except RuntimeError as e:generator_error=e
            if self.router.active=='Transport':
                if self.transport_guard is None:raise RuntimeError('independent_transport_guard_required')
                self.transport_guard.settle(steps)
            # Only complete commands advance the cursor. A partial command ends the
            # branch; the exception cannot cause implicit replay on a new preview.
            if generator_error is not None:raise generator_error
        return fresh
    def observe_executed_step(self,public_state,t,wall,geometry_check):
        if self.router.active=='Transport':
            if self.transport_guard is None:raise RuntimeError('independent_transport_guard_required')
            return self.transport_guard.step(public_state,t,wall,geometry_check)
    def start_pick(self,generator,candidate,context,t,wall,wm_binding):
        if self.router.active!='Pick':raise RuntimeError('pick_not_authorized')
        commands=candidate['commands']
        self.router.begin(candidate['action_id'],commands,candidate['owners'],context,t,wall,wm_binding)
        generator.activate(candidate)
        return commands
    def start_terminal_takeover(self,pick_generator,candidate,context,t,wall,wm_binding):
        if self.router.active!='Hold' or candidate['name']!='terminal_hold':raise RuntimeError('terminal_takeover_not_authorized')
        self.router.begin(candidate['action_id'],candidate['commands'],candidate['owners'],context,t,wall,wm_binding)
        pick_generator.activate(candidate)
        return candidate['commands']
    def settle_pick(self,generator,receipt_id,candidate,steps,end):
        if self.router.receipt(receipt_id,candidate['action_id'],candidate['commands'],steps,end):
            return generator.receipt(receipt_id,candidate['action_id'],candidate['commands'],steps,end)
        return None

def wm_binding(commands,time_s,forecast):
    """Attach only a freshly computed validated forecast, not a saved substitute."""
    h=digest(commands)
    if forecast.get('command_sha256')!=h or forecast.get('observation_time_s')!=time_s:raise RuntimeError('WM_return_not_bound_to_actual_command_and_observation')
    return {'command_sha256':h,'observation_time_s':time_s,
            'legal_forecast':forecast.get('legal_forecast') is True}
