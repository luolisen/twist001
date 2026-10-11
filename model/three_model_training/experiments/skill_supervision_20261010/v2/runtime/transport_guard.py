"""Independent Transport protection. No private task truth or future result input.

Only the registered saved-state research sequence is supported. Dynamic quantities
are measured, not assigned uncalibrated safety limits. General Transport is closed.
"""
import hashlib
from copy import deepcopy
import numpy as np
from gripper_continuation import red_regions

def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class GuardDenied(RuntimeError):
    pass

class TransportGuard:
    def __init__(self, contract, commands, router, rules, prior_public_guard):
        self.c=deepcopy(contract);self.commands=np.asarray(commands).copy();self.router=router
        self.rules=deepcopy(rules)
        self.association=deepcopy(prior_public_guard.association)
        self.last_observation_time=prior_public_guard.last_observation_time
        self.last_reliable=prior_public_guard.last_reliable
        self.last_step_time=None;self.steps=0;self.cursor=0;self.block=None
        self.lease_version=None;self.issued=None;self.wall_start=None
        self.reason=None;self.events=[];self.observations=[];self.telemetry=[]
        if (self.c['profile']!='fixed_saved_state_transport_research' or
                self.c['general_transport_enabled'] is not False or
                self.commands.dtype!=np.float32 or self.commands.shape!=(16,7) or
                not np.isfinite(self.commands).all()):
            raise GuardDenied('unsupported_contract_or_sequence')
        if hashlib.sha256(self.commands.tobytes()).hexdigest()!=self.c['plan_raw_float32_sha256']:
            raise GuardDenied('registered_plan_fingerprint_changed')
        if np.max(np.abs(self.commands[0,:6]-np.asarray(self.c['initial_q_rad'])))>self.c['joint_target_delta_rad']:
            raise GuardDenied('registered_first_target_discontinuous')
        if not np.all(self.commands[:,6]==np.float32(self.c['clamp_target_m'])):
            raise GuardDenied('clamp_target_changed')
        bounds=np.asarray(self.c['control_ranges'])
        if np.any(self.commands<bounds[:,0]) or np.any(self.commands>bounds[:,1]):
            raise GuardDenied('command_control_range')
        if np.max(np.abs(np.diff(self.commands[:,:6],axis=0)))>self.c['joint_target_delta_rad']:
            raise GuardDenied('registered_target_sequence_discontinuous')

    def deny(self, reason):
        if self.reason is None:
            self.reason=reason;self.events.append({'kind':'transport_revoked','reason':reason,'steps':self.steps})
            self.router.revoke(reason)
        raise GuardDenied(self.reason)

    def permission(self,t,wall):
        p=self.router.lease
        if self.reason:self.deny(self.reason)
        if (self.router.active!='Transport' or not p or p.get('skill')!='Transport' or
                p.get('task')!=self.router.task or p.get('target')!=self.router.target or
                p.get('version')!=self.router.version or p.get('owners')!=self.c['owners']):
            self.deny('transport_lease_or_owner_invalid')
        if self.router.mode!='local_saved_state_diagnostic' or p.get('issuer')!='user_authorized_local_assumption':
            self.deny('general_or_automatic_transport_not_validated')
        if not p.get('source_state') or p.get('source_state_sha256')!=self.c['source_state_sha256']:
            self.deny('transport_source_state_binding_invalid')
        if self.lease_version is None:
            self.lease_version=p['version'];self.issued=p['issued_s'];self.wall_start=wall
        if (p['version']!=self.lease_version or t<self.issued-1e-9 or
                t>min(p['expires_s'],self.issued+self.c['max_physics_s'])+1e-9 or
                wall>=p['expires_wall'] or wall-self.wall_start>=self.c['max_wall_s'] or
                p['used_blocks']>self.c['max_blocks']):
            self.deny('transport_permission_expired_or_changed')

    def observe(self,o,wall):
        t=float(o['time']);self.permission(t,wall)
        s=np.asarray(o['state']);images=o['full']
        valid=(s.shape==(21,) and np.isfinite(s).all() and len(images)==2 and
               all(np.asarray(im).dtype==np.uint8 and np.asarray(im).shape==shape and np.any(im)
                   for im,shape in zip(images,[(256,256,3),(512,512,3)])))
        delta=0 if self.last_observation_time is None else t-self.last_observation_time
        if not valid or not -1e-9<=delta<=self.c['observation_dt_s']+1e-9:
            self.deny('transport_public_input_invalid_or_stale')
        self.last_observation_time=t
        fronts,wrists=[red_regions(im) for im in images]
        f,w,conflict,reconfirmed=self.association.update(fronts,wrists,s[:6],t)
        if conflict:self.deny('transport_target_association_conflict')
        if f is not None or w is not None:self.last_reliable=t
        if self.last_reliable is None or t-self.last_reliable>self.rules['maximum_unconfirmed_seconds']+1e-9:
            self.deny('transport_target_history_expired')
        item={'time_s':t,'phase':'Transport','input_valid':True,
              'front_track_confirmed':f is not None,'wrist_track_confirmed':w is not None,
              'target_correspondence_ambiguous':False,'last_reliable_target_time_s':self.last_reliable,
              'terminate':None,'carry_confirmation':None,'pre_release_support':None,
              'close_lift_applicable':False,'reconfirmed':reconfirmed}
        self.observations.append(item);return item

    def prepare_block(self,commands,t,wall,context):
        self.permission(t,wall)
        self.router.check(context,t,wall,True)
        if self.block is not None:self.deny('unsettled_transport_block')
        if self.cursor>=len(self.commands) or t+.32>self.issued+self.c['max_physics_s']+1e-9:
            self.deny('transport_fixed_scope_exhausted')
        expected=self.commands[self.cursor:self.cursor+8]
        actual=np.asarray(commands)
        if actual.dtype!=np.float32 or not np.array_equal(actual,expected):self.deny('transport_command_identity_changed')
        self.block={'start':t,'steps':0,'sha256':hashlib.sha256(actual.tobytes()).hexdigest()}

    def step(self,public_state,t,wall,geometry_check):
        """Called after each original protected step, with real public encoders.

        geometry_check is a separate protection adapter returning only violations;
        no object pose/contact identity is forwarded to the policy or RGB observer.
        """
        self.permission(t,wall);s=np.asarray(public_state)
        if s.shape!=(21,) or not np.isfinite(s).all():self.deny('transport_encoder_invalid')
        if not 0<=s[6]<=.08:self.deny('transport_gripper_mechanical_gap_range')
        p=self.router.inflight
        if not self.block or not p or p['lease_version']!=self.lease_version or p['owners']!=self.c['owners'] or p['sha256']!=self.block['sha256']:
            self.deny('transport_inflight_control_binding_invalid')
        expected_time=self.block['start']+(self.block['steps']+1)*.001
        if abs(t-expected_time)>1e-7:self.deny('transport_physics_receipt_time_invalid')
        if self.steps>=self.c['max_steps'] or self.block['steps']>=320:self.deny('transport_step_budget')
        if self.last_observation_time is None or t-self.last_observation_time>self.c['observation_dt_s']+1e-9:
            self.deny('transport_public_observation_expired')
        limits=np.asarray(self.c['mechanical_ranges']);q=s[:6]
        if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):self.deny('transport_actual_mechanical_range')
        command=self.commands[self.cursor+self.block['steps']//40]
        if not np.array_equal(s[14:21].astype(np.float32),command):self.deny('transport_actual_command_mismatch')
        if geometry_check is None:self.deny('transport_geometry_protection_missing')
        violations=geometry_check(q)
        if violations:self.deny('transport_geometry:'+str(violations[0]))
        # No threshold is inferred from past observations or endpoint acceptance.
        self.telemetry.append({'time_s':t,'q_rad':q.tolist(),'qvel_rad_s':s[7:13].tolist(),
                               'tracking_error_rad':(command[:6]-q).tolist(),
                               'dynamic_safety_limits_validated':False})
        self.steps+=1;self.block['steps']+=1;self.last_step_time=t

    def settle(self,steps):
        if self.reason:
            # The executor retains the offending actual step. It is a terminal
            # prefix, not an implicit repeat of an uncommitted command.
            if self.block and steps in (self.block['steps'],self.block['steps']+1):
                self.cursor+=steps//40;self.block=None
            raise GuardDenied(self.reason)
        if not self.block or steps!=self.block['steps']:self.deny('transport_guard_receipt_mismatch')
        self.cursor+=steps//40;self.block=None
        if steps!=320:self.deny('transport_partial_prefix_no_resume')

class StageProtection:
    """V2 dispatch; the frozen continuation remains untouched and unreset."""
    def __init__(self,router,frozen_continuation,transport=None):
        self.router=router;self.frozen=frozen_continuation;self.transport=transport
    def observe(self,o,wall):
        if self.router.active in ('Pick','Hold'):
            return self.frozen.observe(o)
        if self.router.active=='Transport' and self.transport is not None:
            return self.transport.observe(o,wall)
        raise GuardDenied('no_registered_stage_protection')

def physical_launch_gate(contract_path,approval,source_state_path,plan_path,path_result):
    """Fail closed before simulator initialization. Caller supplies explicit approval.

    Software fixtures cannot authorize a physical run. The approval record is not
    generated by this module or its tests.
    """
    if not approval or approval.get('source')!='direct_user_confirmation' or approval.get('scope')!='one_saved_state_transport_simulation':
        raise GuardDenied('new_transport_scope_requires_user_confirmation')
    if approval.get('contract_sha256')!=file_sha(contract_path):raise GuardDenied('approval_contract_mismatch')
    import json
    c=json.loads(contract_path.read_text())
    if file_sha(source_state_path)!=c['source_state_sha256'] or file_sha(plan_path)!=c['plan_sha256']:
        raise GuardDenied('initial_state_or_plan_fingerprint_mismatch')
    if path_result.get('passed') is not True or path_result.get('contract_sha256')!=file_sha(contract_path):
        raise GuardDenied('independent_transport_path_not_checked')
    if path_result.get('physics_steps')!=0:raise GuardDenied('invalid_preflight_evidence')
    return c
