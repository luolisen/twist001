"""Single-factor execution aid for a bounded online VLA comparison.

Inputs are already denormalized absolute joint/gap targets in rad/metres.
This helper neither changes timing nor clips limits; caller must validate its
returned candidate with the same deterministic execution boundary as baseline.
"""
import numpy as np


def scale_arm_residual(commands, current_joint_positions, gain=0.5):
    """Keep every gripper target; scale all six arm target residuals together.

    Freeze the six public encoder values once at the beginning of each block.
    Do not anchor to a previous control target or hidden object state. A gain of
    1.0 is the matched baseline. The registered intervention is 0.5; its value
    must not be tuned in response to the result of the bounded comparison.
    """
    original = np.asarray(commands)
    state = np.asarray(current_joint_positions, dtype=np.float64)
    if original.shape != (8, 7) or not np.issubdtype(original.dtype, np.floating):
        raise ValueError("Expected eight floating seven-channel absolute targets")
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("Expected six finite measured joint angles in radians")
    if not np.all(np.isfinite(original)):
        raise ValueError("Nonfinite source action")
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("Gain must be finite and in (0,1]")
    # Exact identity on baseline also preserves source precision and signed zero.
    if gain == 1:
        return original.copy()
    result = original.astype(np.float64, copy=True)
    result[:, :6] = state + gain * (result[:, :6] - state)
    return result
