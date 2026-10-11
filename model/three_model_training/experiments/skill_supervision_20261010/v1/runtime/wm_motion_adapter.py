"""Frozen WM joint-response forecast -> static FK endpoint, without future observations.

The direct XYZ head stays in a separate diagnostic archive. Risk and commands are unchanged.
"""
import copy,json
import numpy as np
from task_relative import ee_reference
from fusion_core import check_public,command_digest,compact

SOURCE='static_FK_of_current_public_encoder_plus_WM_predicted_joint_delta'
def adapt_motion(state,motion,calibration):
    s=np.asarray(state,dtype=np.float64);md=np.asarray(motion,dtype=np.float64)
    if s.shape!=(21,) or md.shape!=(10,) or not np.isfinite(s).all() or not np.isfinite(md).all():
        raise ValueError('invalid_public_state_or_WM_motion_no_clipping_or_fallback')
    if calibration['ee_reference']['name']!='ee_center_site':raise ValueError('unsupported_EE_reference')
    q1=s[:6]+md[:6]
    if not np.isfinite(q1).all():raise ValueError('invalid_forecast_joint_state')
    p0=ee_reference(calibration,s[:6]);p1=ee_reference(calibration,q1);xyz=p1-p0
    if not np.isfinite(xyz).all():raise ValueError('nonfinite_static_FK')
    adapted=md.copy();adapted[7:10]=xyz
    return adapted.tolist(),{'source':SOURCE,'current_encoder_q_rad':s[:6].tolist(),
        'WM_joint_delta_rad':md[:6].tolist(),'forecast_measured_q_rad':q1.tolist(),
        'current_EE_world_m':p0.tolist(),'forecast_EE_world_m':p1.tolist(),
        'raw_WM_motion10':md.tolist(),'adapted_motion10':adapted.tolist(),
        'direct_vs_joint_FK_difference_m':float(np.linalg.norm(md[7:]-xyz)),
        'difference_is_not_error_bound_or_confidence':True,'physics_steps':0,
        'future_truth_inputs':False,'control_target_used_as_future_state':False}

def adapt_forecasts(forecasts,state,calibration):
    result=copy.deepcopy(forecasts);archive=[]
    for old,new in zip(forecasts,result):
        if command_digest(np.asarray(old['actions'],dtype=np.float32))!=old['command_sha256']:
            raise ValueError('WM_final_command_binding_mismatch')
        new['motion_delta'],entry=adapt_motion(state,old['motion_delta'],calibration)
        entry.update(candidate_name=old['name'],candidate_index=old['candidate_index'],command_sha256=old['command_sha256'])
        assert {k:v for k,v in old.items() if k!='motion_delta'}=={k:v for k,v in new.items() if k!='motion_delta'}
        assert old['motion_delta'][:7]==new['motion_delta'][:7]
        archive.append(entry)
    return result,archive

def annotate_request(request):
    """Only truthful output provenance; no reliability, preference or progress claims."""
    out=copy.deepcopy(request);state=json.loads(out['state'])
    state['forecast_contract']['motion']='WM joint delta6 rad,gap delta mm; world EE delta XYZ mm derived by static FK(current encoder q + WM joint delta6) minus FK(q); estimate, not measured motion'
    out['state']=json.dumps(compact(state),separators=(',',':'))
    for name,text in out['questions']['selection']['criteria'].items():
        if name in ('observe','reject','replan'):continue
        item=json.loads(text);item['WM']['EE_motion_source']=SOURCE
        item['task_relative_forecast']['source']=SOURCE
        out['questions']['selection']['criteria'][name]=json.dumps(compact(item),separators=(',',':'))
    return out
