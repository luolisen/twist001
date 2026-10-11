"""Public feature extraction, deliberately not a validated load/deposit detector."""
import numpy as np
from task_relative import observed_relation
from evidence_interfaces import unresolved_support,unresolved_placement

def public_features(observation, track, calibration, actual_receipts=(), previous=None):
    s=np.asarray(observation['state']);t=float(observation['time'])
    valid=bool(s.shape==(21,) and np.isfinite(s).all())
    if not valid: raise ValueError('invalid_public_encoders')
    relation=observed_relation(calibration,s,track,t)
    observed={'time_s':t,'valid_encoders':valid,'measured_gap_m':float(s[6]),
              'last_applied_clamp_target_m':float(s[20]),
              'gap_minus_command_m':float(s[6]-s[20]),
              'command_on_closing_side':bool(s[20]<s[6]),
              'front_reconfirmed':track['front'] is not None,
              'wrist_reconfirmed':track['wrist'] is not None,
              'target_conflict':bool(track['conflict']),
              'front_region':track['front'],'wrist_region':track['wrist'],
              'ee_world_m':relation['ee_world_m'],
              'ee_pixel_128':relation['ee_image_projection'],
              'target_minus_ee_pixel_128':relation.get('target_center_minus_ee_pixels'),
              'actual_completed_blocks':len(actual_receipts)}
    if previous is not None:
        p=previous['observed'];dt=t-p['time_s']
        if dt>0:
            observed['observed_interval_s']=dt
            observed['observed_ee_delta_m']=(np.array(observed['ee_world_m'])-p['ee_world_m']).tolist()
            observed['observed_gap_delta_m']=observed['measured_gap_m']-p['measured_gap_m']
            if track['front'] is not None and p['front_region'] is not None:
                observed['front_region_pixel_delta']=(127*(np.array(track['front']['center'])-p['front_region']['center'])).tolist()
            if track['wrist'] is not None and p['wrist_region'] is not None:
                observed['wrist_region_pixel_delta']=(127*(np.array(track['wrist']['center'])-p['wrist_region']['center'])).tolist()
    # These necessary provenance/visibility checks do not prove physical load.
    prerequisites={'current_encoder_and_command_valid':True,
                   'visual_association_current_and_unambiguous':bool((track['front'] or track['wrist']) and not track['conflict']),
                   'has_actual_execution_receipt':bool(actual_receipts),
                   'temporal_public_motion_available':previous is not None and t>previous['observed']['time_s']}
    return {'observed':observed,'forecast':{},'handoff_prerequisites':prerequisites,
            'unknown':{'target_depth':None,'held_object_offset':None,'actual_load_retention':None,
                       'worksite_support':None,'deposit_complete':None},
            'carry_evidence':{'established':False,'source':'public_observation_and_execution',
                              'validator_id':None,'observation_time_s':t,
                              'reason':'public_features_not_validated_against_independent_carry_and_empty_or_supported_cases'},
            'support_evidence':unresolved_support(t),
            'placement_evidence':unresolved_placement(t),
            'automatic_transport_permitted':False,'automatic_release_permitted':False}
