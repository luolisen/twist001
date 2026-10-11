"""Reuse archived IK points; static planning only, no execution permission."""
import sys,json,hashlib,time,csv,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

R=Path(__file__).resolve().parent
S=R.parent
F=S/'v2_transport_feedforward_20261010'
T=S/'v2_taskspace_transport_20261010'
sys.path.insert(0,str(F/'runtime'))
sys.path.insert(0,str(S/'wm_consistent_v7'))
from skill_generators import fk_pose,segment
from gravity_feedforward import FixedStructureGravity
from transport_geometry import StaticTransportGeometry,overlaps
import physical_labels as labels

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x): (R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    start=time.monotonic()
    c=json.load(open(F/'contract.json'))
    registration=json.load(open(R/'registration.json'))
    reg=json.load(open(T/'registration.json'))
    prior=json.load(open(T/'planning_result.json'))
    cal=json.load(open(F/'sources/public_calibration.json'))
    source_paths=[F/'contract.json',F/'runtime/gravity_feedforward.py',F/'runtime/transport_geometry.py',
        F/'runtime/skill_generators.py',F/'sources/public_calibration.json',T/'registration.json',T/'planning_result.json',Path(reg['scene'])]
    before={str(p):sha(p) for p in source_paths}
    if before[str(Path(reg['scene']))]!=c['scene_sha256']:raise RuntimeError('scene_identity_mismatch')
    for name in ('registration.json','planning_result.json'):
        if sha(R/'sources'/name)!=sha(T/name):raise RuntimeError('archived_planning_source_differs:'+name)
    gravity=FixedStructureGravity(reg['scene'])
    if (gravity.method_sha256!=c['gravity_method_sha256'] or gravity.parameters_sha256!=c['gravity_parameters_sha256']):
        raise RuntimeError('frozen_gravity_identity_mismatch')
    geom=StaticTransportGeometry(c,reg['scene'],cal,labels)
    q0=np.asarray(c['initial_q_rad']);pose0=fk_pose(cal,q0)
    ctrl=np.asarray(c['control_ranges']);mech=np.asarray(c['mechanical_ranges']);clamp=np.float32(c['clamp_target_m'])
    points=[x for x in prior['continuation'] if x['hard_constraints_met'] and x['distance_along_route_m']<=.080000001]
    # Predeclared smoothstep timing, preserving original .02rad command limit.
    # Minimum one original block per5mm; longer intervals allocated by slope.
    commands=[];metadata=[];prev=q0.copy();timing=[]
    for si,row in enumerate(points):
        goal=np.asarray(row['q_rad']);count=max(8,8*math.ceil(1.5*np.max(np.abs(goal-prev))/.02/8))
        seq=segment(prev,goal,float(clamp),float(clamp),count)
        commands.extend(seq)
        metadata.extend([{'segment':si,'route_end_m':row['distance_along_route_m'],'fraction':(i+1)/count} for i in range(count)])
        timing.append({'route_end_m':row['distance_along_route_m'],'commands':count,'duration_s':count*.04})
        prev=goal
    cmd=np.asarray(commands,dtype=np.float32)
    np.savez_compressed(R/'candidate_0_80mm_nominal.npz',commands=cmd,initial_q=q0,route_endpoints=np.array([x['target_ee_world_m'] for x in points]))
    point_rows=[]
    for r in prior['continuation']:
        q=np.asarray(r['q_rad']);bad=np.where((q<mech[:,0])|(q>mech[:,1]))[0]
        point_rows.append({'route_mm':r['distance_along_route_m']*1000,'old_control_bounded_IK_pass':r['hard_constraints_met'],
            'position_error_mm':r['position_error_m']*1000,'rotation_error_rad':r['rotation_error_rad'],
            'q_rad':q.tolist(),'mechanical_violations':[{'joint':int(j+1),'q_rad':float(q[j]),'range_rad':mech[j].tolist()} for j in bad],
            'mechanically_legal':not len(bad)})
    rows=[];samples=[];failures=[];last_q=q0;last_servo=np.asarray(c['initial_servo_command'])[:6];first=None
    desired=np.asarray(points[-1]['target_ee_world_m'])-pose0[:3,3];direction=desired/np.linalg.norm(desired)
    max_jump=0.;max_servo_jump=0.;max_rot=0.;max_corridor=0.
    for i,u in enumerate(cmd):
        if time.monotonic()-start>registration['static_wall_cap_s']:raise TimeoutError('static_audit_wall_cap')
        max_jump=max(max_jump,float(np.max(np.abs(u[:6]-last_q))))
        servo=None;command_fail=[];proxy_geometry=[]
        if np.any(u<ctrl[:,0]) or np.any(u>ctrl[:,1]):command_fail.append('nominal_control_range')
        if np.any(u[:6]<mech[:,0]) or np.any(u[:6]>mech[:,1]):command_fail.append('nominal_mechanical_range')
        if not command_fail:
            # Static proxy only: hypothetical encoder equals nominal target.
            servo=u.astype(float);servo[:6]+=gravity.known_gravity(u[:6])/100;servo=servo.astype(np.float32)
            jump=float(np.max(np.abs(servo[:6]-last_servo)));max_servo_jump=max(max_servo_jump,jump)
            if jump>.02+1e-8:command_fail.append('proxy_servo_adjacent_delta')
            if np.any(servo<ctrl[:,0]) or np.any(servo>ctrl[:,1]):command_fail.append('proxy_servo_control_range')
            if np.any(servo[:6]<mech[:,0]) or np.any(servo[:6]>mech[:,1]):command_fail.append('proxy_servo_mechanical_range')
            else:
                proxy_geometry=sorted(set(geom(servo[:6])))
                command_fail.extend(['proxy_servo_geometry:'+k for k in proxy_geometry])
            last_servo=servo[:6]
        interval_fail=[]
        for fraction in np.linspace(0,1,5):
            q=last_q+(u[:6]-last_q)*fraction;pose=fk_pose(cal,q)
            rotation=float(Rotation.from_matrix(pose0[:3,:3]@pose[:3,:3].T).magnitude())
            rel=pose[:3,3]-pose0[:3,3];distance=float(rel@direction);corridor=float(np.linalg.norm(rel-distance*direction))
            max_rot=max(max_rot,rotation);max_corridor=max(max_corridor,corridor)
            kinds=[];details=[]
            if np.any(q<mech[:,0]) or np.any(q>mech[:,1]):kinds.append('sample_mechanical_range')
            else:
                kinds.extend(geom(q))
                if 'conditional_payload_registered_geometry_overlap' in kinds:
                    payload=geom.env@pose[:3,:3].T+pose[:3,3];box=np.stack([payload.min(0),payload.max(0)])
                    for g in sorted(geom.obstacles|geom.robot_obstacles):
                        ob=geom.aabb(g)
                        if (box[0,2]<=geom.d.geom_xpos[g,2] if ob is None else overlaps(box,ob)):
                            details.append(geom.m.geom(g).name)
            if corridor>.0025:kinds.append('static_2_5mm_corridor')
            row={'command_index':i,'fraction':float(fraction),'route_projection_mm':distance*1000,
                'ee_world_m':pose[:3,3].tolist(),'orientation_error_rad':rotation,'corridor_error_mm':corridor*1000,
                'failure_kinds':sorted(set(kinds)),'overlap_geometries':details}
            samples.append(row)
            if kinds:interval_fail.append(row)
        row={'command_index':i,'segment':metadata[i]['segment'],'segment_end_mm':metadata[i]['route_end_m']*1000,
            'nominal':u.tolist(),'servo_proxy':None if servo is None else servo.tolist(),
            'servo_proxy_geometry_failures':proxy_geometry,'command_failures':command_fail,'path_failures':interval_fail}
        rows.append(row)
        if command_fail or interval_fail:
            failures.append(row)
            if first is None:first=row
        last_q=u[:6]
    prefix_count=first['command_index'] if first else len(cmd)
    # Use complete blocks only; don't claim an unregistered partial block endpoint.
    prefix_count=(prefix_count//8)*8
    np.savez_compressed(R/'static_legal_prefix_nominal.npz',commands=cmd[:prefix_count],initial_q=q0)
    prefix_pose=fk_pose(cal,cmd[prefix_count-1,:6]) if prefix_count else pose0
    first_ik=prior['first_failed'];first_mech=next((r for r in point_rows if not r['mechanically_legal']),None)
    save('endpoint_audit.json',point_rows);save('command_audit.json',rows);save('static_sample_audit.json',samples)
    final={'scope':'static_only_no_physical_permission','physics_steps':0,'model_calls':{'VLA':0,'WM':0,'Jev':0},
        'IK_solver_calls':0,'reuses_previous_IK':True,'candidate_route_mm':points[-1]['distance_along_route_m']*1000,
        'candidate_nominal_commands':len(cmd),'candidate_blocks':len(cmd)//8,'candidate_nominal_duration_s':len(cmd)*.04,
        'timing_rule':'per5mm smoothstep, minimum8 commands; multiple8 ceil(1.5*max_joint_delta/.02/8); not a calibrated dynamic bound',
        'timing_segments':timing,'candidate_static_passed':not failures,'first_path_failure':first,
        'first_archived_IK_failure':first_ik,'first_discrete_mechanical_failure':first_mech,
        'legal_complete_prefix_commands':prefix_count,'legal_complete_prefix_blocks':prefix_count//8,'legal_prefix_duration_s':prefix_count*.04,
        'legal_prefix_projected_mm':float((prefix_pose[:3,3]-pose0[:3,3])@direction*1000),
        'legal_prefix_actual_static_ee_world_m':prefix_pose[:3,3].tolist(),
        'static_sample_count':len(samples),'max_nominal_adjacent_rad':max_jump,'max_proxy_servo_adjacent_rad':max_servo_jump,
        'max_orientation_error_rad':max_rot,'max_corridor_error_mm':max_corridor*1000,
        'cell0_center_world_m':json.load(open(R/'sources/task_geometry_and_expert_sources.json'))['worksite']['cell_center_world_m'],
        'cell0_source':'archived actual scene cell0 geometry',
        'target_kind':'EE carry intermediate along registered horizontal route; not object center or release pose',
        'held_object_offset':'unknown; frozen conditional payload envelope unchanged',
        'fixed_orientation_tolerance_rad':.05,'position_tolerance_m':.0025,
        'unknowns':['future actual public encoder/servo targets','dynamic following on longer path','payload offset/slip','public carry/support/release evidence','continuous collision certificate'],
        'servo_proxy_is_executable_matrix':False,'continuous_collision_certified':False,
        'old_640_step_permission_reused':False,'new_local_simulation_executed':False,
        'servo_proxy_geometry_checked':True,'checker_sha256':sha(__file__),'registration_sha256':sha(R/'registration.json'),
        'frozen_sources_sha256':before,'frozen_sources_unchanged':all(sha(p)==h for p,h in before.items()),
        'wall_seconds':time.monotonic()-start}
    save('path_result.json',final)
    print(json.dumps({k:final[k] for k in ('candidate_nominal_commands','candidate_nominal_duration_s','candidate_static_passed','legal_prefix_projected_mm','legal_prefix_duration_s','first_discrete_mechanical_failure')},ensure_ascii=False))

if __name__=='__main__':main()
