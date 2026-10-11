"""Public encoder/static FK and RGB-track 2D context. No simulator input."""
import copy,json
import numpy as np
from public_association import camera_poses
from fusion_core import compact

def project_front(point_world_m,calibration,q):
    camera=calibration['cameras'][0]
    pose=camera_poses(calibration,q)[0]
    xyz=pose[:3,:3].T@(np.asarray(point_world_m)-pose[:3,3])
    if not np.isfinite(xyz).all() or xyz[2]>=-1e-9:
        return {'valid':False,'reason':'reference_behind_camera_or_nonfinite'}
    f=36./np.tan(np.deg2rad(camera['fovy'])/2)
    pixel=np.array([63.5+f*xyz[0]/(-xyz[2]),63.5-f*xyz[1]/(-xyz[2])])
    return {'valid':True,'pixel_128':pixel.tolist(),'inside_content':bool(0<=pixel[0]<=127 and 28<=pixel[1]<=99)}

def ee_reference(calibration,q):
    # The same public hinge-chain implementation, with a static site attachment.
    frame={'cameras':[calibration['ee_reference']]}
    return camera_poses(frame,q)[0][:3,3]

def observed_relation(calibration,state,track,t):
    q=np.asarray(state,dtype=float)
    if q.shape!=(21,) or not np.isfinite(q).all():
        return {'valid':False,'reason':'invalid_public_encoder_state'}
    if track['conflict']:
        region=None;meaning='ambiguous_public_correspondence'
    elif track['front'] is None:
        region=None;meaning='target_not_reconfirmed_in_current_front_frame'
    else:
        region=copy.deepcopy(track['front']);meaning='existing_public_track_proposal_not_verified_object_identity'
    point=ee_reference(calibration,q[:6]);projection=project_front(point,calibration,q[:6])
    result={'valid':True,'time_s':t,'reference':'fixed front_rgb, current image;128x128 letterbox, content128x72',
        'target_region':region,'target_relation_status':meaning,
        'target_identity_verified':False,'target_depth_m':None,'three_dimensional_target_distance_m':None,
        'ee_reference_name':calibration['ee_reference']['name'],'ee_world_m':point.tolist(),
        'ee_image_projection':projection,'is_grasp_geometry_or_grasp_confirmation':False}
    if region is not None and projection['valid']:
        center=np.asarray(region['center'])*127
        result['target_center_minus_ee_pixels']=(center-np.asarray(projection['pixel_128'])).tolist()
    return result

def predicted_relation(calibration,state,observed,motion_delta):
    md=np.asarray(motion_delta,dtype=float)
    if not observed.get('valid') or md.shape!=(10,) or not np.isfinite(md).all():
        return {'valid':False,'reason':'missing_valid_public_reference_or_WM_motion'}
    projection=project_front(np.asarray(observed['ee_world_m'])+md[7:10],calibration,np.asarray(state)[:6])
    result={'source':'original_WM_world_EE_delta','horizon_s':.320,
        'reference':'same current fixed front camera; compared to current observed target region, not a target trajectory forecast',
        'predicted_ee_image_projection':projection,'target_future_motion':'unknown',
        'depth_alignment_orientation_and_path_clearance':'unknown',
        'not_action_value_or_grasp_readiness':True}
    current=observed['ee_image_projection'];region=observed['target_region']
    if projection['valid'] and current['valid']:
        result['predicted_ee_pixel_delta']=(np.asarray(projection['pixel_128'])-current['pixel_128']).tolist()
        if region is not None:
            center=np.asarray(region['center'])*127
            result['current_image_center_offset_norm_pixels']=float(np.linalg.norm(center-current['pixel_128']))
            result['predicted_image_center_offset_norm_pixels']=float(np.linalg.norm(center-projection['pixel_128']))
    return result

def revised_request(request,public,observed,calibration):
    out=copy.deepcopy(request)
    state=json.loads(out['state']);state['task_relative_observation']=observed
    out['state']=json.dumps(compact(state),separators=(',',':'))
    for c in public['candidates']:
        item=json.loads(out['questions']['selection']['criteria'][c['name']])
        item['task_relative_forecast']=predicted_relation(calibration,public['motor_state'],observed,c['motion_delta'])
        out['questions']['selection']['criteria'][c['name']]=json.dumps(compact(item),separators=(',',':'))
    return out
