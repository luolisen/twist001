"""Explicit execution aid, never a claim of raw-policy legality or physical safety."""
import numpy as np


def project_arm_targets(commands, actuator_limits):
    original = np.asarray(commands)
    limits = np.asarray(actuator_limits, dtype=np.float64)
    if original.shape != (8, 7) or not np.issubdtype(original.dtype, np.floating):
        raise ValueError('expected eight seven-channel floating commands')
    if limits.shape != (7, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError('invalid frozen actuator limits')
    if not np.isfinite(original).all():
        raise ValueError('nonfinite source command')
    low = limits[:6, 0].astype(np.float32)
    high = limits[:6, 1].astype(np.float32)
    low = np.where(low.astype(np.float64) < limits[:6, 0], np.nextafter(low, np.float32(np.inf)), low)
    high = np.where(high.astype(np.float64) > limits[:6, 1], np.nextafter(high, np.float32(-np.inf)), high)
    if np.any(low > high):
        raise ValueError('no representable float32 command in actuator interval')
    result = original.astype(np.float32, copy=True)
    result[:, :6] = np.clip(result[:, :6], low, high)
    source = original.astype(np.float32)
    delta = result.astype(np.float64) - source.astype(np.float64)
    changed = np.argwhere(delta[:, :6] != 0)
    if not np.array_equal(source[:, 6], result[:, 6]):
        raise ValueError('projection changed gripper')
    if np.any(result[:, :6] < limits[:6, 0]) or np.any(result[:, :6] > limits[:6, 1]):
        raise ValueError('projected command outside unchanged limits')
    record = {'mode': 'explicit_arm_limit_projection', 'clipping': True,
              'limits_relaxed': False, 'timing_changed': False, 'gripper_source_preserved': True,
              'not_pure_unassisted_VLA': True, 'not_physical_safety_certificate': True,
              'changed_count': int(len(changed)), 'max_abs_delta_rad': float(np.abs(delta[:, :6]).max()),
              'per_joint_max_abs_delta_rad': np.abs(delta[:, :6]).max(axis=0).tolist(),
              'changed_command_joint_indices': changed.tolist(),
              'frozen_actuator_limits': limits.tolist(),
              'float32_inward_arm_limits': np.stack((low, high), axis=1).tolist()}
    return result, record
