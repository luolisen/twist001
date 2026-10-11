"""Replaceable generators. No future object truth and no private success input."""
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from public_association import camera_poses
from evidence_interfaces import valid_public_evidence,SUPPORT_KIND

class PlanningBlocked(RuntimeError):
    def __init__(self,record):
        self.record=record
        super().__init__('bounded_IK_endpoint_not_reached:'+str(record))

class PickGenerator:
    binding='frozen_pick_v1'
    def __init__(self,transactions,limits,names):self.tx=transactions;self.limits=limits;self.names=names
    def preview(self,raw50,projected,observation,source,decision_id):
        cs,boundary,_,_=self.tx.preview(raw50,projected,observation,source,self.limits,self.names,decision_id)
        main=next((c for c in cs if c['name'] in ('vla_raw','terminal_hold')),None)
        if main is None:raise RuntimeError('frozen_main_action_unavailable')
        return main,boundary
    def activate(self,c):self.tx.activate(c)
    def receipt(self,*args):return self.tx.settle(*args)

class HoldGenerator:
    binding='terminal_hold_v1'
    def __init__(self,controller):self.controller=controller;self.owner=True
    def commands(self,count=8):
        if not self.owner or self.controller.phase!='holding' or self.controller.stop_reason:raise RuntimeError('hold_ownership_unavailable')
        return self.controller.commands(count)
    def receipt(self,steps,t):self.controller.commit(steps,t)
    def release_authority(self,router):
        if router.active!='Transport' or not router.lease or router.inflight:raise RuntimeError('transport_lease_required_before_handoff')
        self.owner=False
        return {'previous_owner':'terminal_hold','next_owner':'transport_trajectory','release_clamp':False}

class BoundedTrajectory:
    def __init__(self,commands):
        self.plan=np.asarray(commands,np.float32);self.cursor=0;self.pending=None
        if self.plan.ndim!=2 or self.plan.shape[1]!=7 or len(self.plan)%8 or not np.isfinite(self.plan).all():raise ValueError('invalid_plan')
    def preview(self):
        if self.pending is not None:raise RuntimeError('unsettled_generator')
        if self.cursor+8>len(self.plan):raise StopIteration
        return self.plan[self.cursor:self.cursor+8].copy()
    def activate(self,commands):
        if self.pending is not None or not np.array_equal(commands,self.preview()):raise RuntimeError('generator_binding_changed')
        self.pending=commands.copy()
    def receipt(self,steps):
        if self.pending is None or type(steps)!=int or not 0<=steps<=320:raise RuntimeError('invalid_generator_receipt')
        self.cursor+=steps//40;self.pending=None
        if steps%40:raise RuntimeError('partial_command_requires_end_no_resume')

def fk_pose(calibration,q):return camera_poses({'cameras':[calibration['ee_reference']]},q)[0]
def bounded_ik(calibration,q,xyz,rotation,limits,max_evaluations=120):
    # One seed only; public static mechanical chain, not MuJoCo rollouts.
    q=np.asarray(q,float);lo=limits[:6,0];hi=limits[:6,1]
    if np.any(q<lo) or np.any(q>hi):raise ValueError('initial_encoder_out_of_range')
    def residual(x):
        pose=fk_pose(calibration,x)
        return np.r_[(pose[:3,3]-xyz)*10,Rotation.from_matrix(rotation@pose[:3,:3].T).as_rotvec()*2]
    fit=least_squares(residual,q,bounds=(lo,hi),max_nfev=max_evaluations,xtol=1e-8,ftol=1e-8,gtol=1e-8)
    e=residual(fit.x);p=float(np.linalg.norm(e[:3])/10);a=float(np.linalg.norm(e[3:])/2)
    record={'evaluations':fit.nfev,'position_error_m':p,'orientation_error_rad':a,'one_seed':True,'public_static_FK':True,'private_future_used':False}
    if p>.0025 or a>.05:raise PlanningBlocked(record)
    return fit.x,record

def segment(q0,q1,g0,g1,count):
    t=np.arange(1,count+1,dtype=float)/count;s=t*t*(3-2*t)
    return np.c_[q0[None]+s[:,None]*(q1-q0)[None],g0+s*(g1-g0)].astype(np.float32)

def registered_transport_waypoint(worksite):
    contract=worksite.get('transport_waypoint',{})
    if (contract.get('kind')!='intermediate_EE_carry_waypoint'
        or contract.get('source')!='public_encoder_static_configuration'
        or not contract.get('source_binding')
        or contract.get('certifies_object_placement') is not False):
        raise RuntimeError('explicit_EE_carry_waypoint_required_object_center_not_EE_target')
    goal=np.asarray(contract.get('ee_world_m'),float)
    if goal.shape!=(3,) or not np.isfinite(goal).all():raise ValueError('invalid_registered_EE_waypoint')
    return goal

class TransportGenerator(BoundedTrajectory):
    binding='bounded_ik_transport_v2'
    def __init__(self,state,calibration,worksite,limits,count=160):
        s=np.asarray(state);pose=fk_pose(calibration,s[:6]);goal=registered_transport_waypoint(worksite)
        q,record=bounded_ik(calibration,s[:6],goal,pose[:3,:3],limits)
        self.record={'IK':record,'start_public_q':s[:6].tolist(),'start_ee_world_m':pose[:3,3].tolist(),
                     'goal_ee_world_m':goal.tolist(),'target_kind':'intermediate_EE_carry_waypoint',
                     'object_placement_certified':False,'target_object_position_not_used':True,
                     'keeps_last_applied_clamp_target':float(s[20]),'endpoint_not_path_safety_proof':True}
        super().__init__(segment(s[:6],q,float(s[20]),float(s[20]),count))

class PlaceGenerator:
    binding='bounded_place_v2'
    def __init__(self,state,calibration,worksite,limits):
        s=np.asarray(state);pose=fk_pose(calibration,s[:6]);goal=np.asarray(worksite['cell_center_world_m']).copy()
        # Use every finger-box corner in the EE frame, including actual clamp
        # separation and orientation. A local Z half-size alone is not world height.
        corners=np.asarray(worksite['finger_corners_in_ee_frame_m'],float)
        if corners.ndim!=2 or corners.shape[1]!=3 or not np.isfinite(corners).all():raise ValueError('finger_geometry_missing')
        low=float((pose[:3,:3]@corners.T)[2].min())
        goal[2]=worksite['support_top_world_m']-low
        # This constrains robot geometry only, not the carried-object offset or
        # deposited support. Validated pre-release support is required to open.
        q,record=bounded_ik(calibration,s[:6],goal,pose[:3,:3],limits)
        self.descend=BoundedTrajectory(segment(s[:6],q,float(s[20]),float(s[20]),80))
        self.state='descend';self.q=q;self.g=float(s[20]);self.limits=limits;self.record=dict(record,robot_geometry_endpoint_world_m=goal.tolist(),object_support_known=False);self.release=None;self.withdraw=None
    def release_preview(self,public_support_evidence,registered_validators=(),*,observation_time_s=None,target=None):
        if (self.state!='await_support' or not valid_public_evidence(public_support_evidence,SUPPORT_KIND,
                registered_validators,observation_time_s,target)):raise RuntimeError('pre_release_support_unknown_release_disabled')
        if self.release is None:self.release=BoundedTrajectory(segment(self.q,self.q,self.g,float(self.limits[6,1]),32))
        return self.release.preview()
    def finish_descent(self):
        if self.descend.cursor!=len(self.descend.plan):raise RuntimeError('descent_not_executed')
        self.state='await_support'
    def prepare_withdraw(self,public_state,calibration,worksite):
        if self.release is None or self.release.cursor!=len(self.release.plan):raise RuntimeError('no_actual_release_prefix')
        pose=fk_pose(calibration,np.asarray(public_state)[:6]);goal=pose[:3,3].copy();goal[2]+=2*worksite['finger_half_height_m']
        q,_=bounded_ik(calibration,np.asarray(public_state)[:6],goal,pose[:3,:3],self.limits)
        self.withdraw=BoundedTrajectory(segment(np.asarray(public_state)[:6],q,float(public_state[20]),float(public_state[20]),32));self.state='withdraw'

class RecoveryGenerator:
    binding='no_motion_recovery_v2'
    def commands(self):raise RuntimeError('recovery_interface_only_no_unverified_motion')
