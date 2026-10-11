"""One preregistered external execution aid; no expert or privileged input."""
import numpy as np
from arm_residual import scale_arm_residual


def apply_intervention(commands, public_state, mode):
    original=np.asarray(commands)
    if mode=='none':
        corrected=original.copy()
    elif mode=='joint_residual_half':
        corrected=scale_arm_residual(original,np.asarray(public_state)[:6],gain=.5)
    else:
        raise ValueError('unregistered correction mode')
    if not np.array_equal(original[:,6],corrected[:,6]):
        raise ValueError('single-factor correction changed gripper targets')
    return corrected,{'mode':mode,'gain':.5 if mode!='none' else 1.,
                      'anchor':'public_measured_six_joint_angles_at_block_start',
                      'gripper_source_preserved':True,'clipping':False,'timing_changed':False,
                      'not_pure_unassisted_VLA':mode!='none'}
