"""Pure NumPy diagnostics for absolute actuator position targets.

This helper does not change commands, relax limits, choose a policy action,
execute physics, or decide whether an in-range action is physically safe.
"""
from __future__ import annotations

import numpy as np


def diagnose_candidates(candidates, control_ranges, actuator_names):
    """Isolate malformed/out-of-range candidates; return valid names in order.

    candidates maps each unique candidate name to [time, actuator] commands.
    The fixed 1e-7 tolerance reproduces the original control-target check.
    """
    limits = np.asarray(control_ranges)
    names = list(actuator_names)
    if limits.shape != (len(names), 2):
        raise ValueError("control_ranges must contain one lower/upper pair per actuator")
    if not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("control limits must be finite and ordered")
    reports, retained, rejected = [], [], []
    for name, commands in candidates.items():
        action = np.asarray(commands)
        entry = {"candidate": name, "valid_control_targets": False,
                 "violations": [], "shape_error": None}
        if action.ndim != 2 or action.shape[1] != len(names) or action.shape[0] == 0:
            entry["shape_error"] = {"actual": list(action.shape),
                                    "expected": ["positive_time_steps", len(names)]}
        else:
            finite = np.isfinite(action)
            low = finite & (action < limits[:, 0] - 1e-7)
            high = finite & (action > limits[:, 1] + 1e-7)
            for t, c in np.argwhere(~finite | low | high):
                value = action[t, c]
                reason = "nonfinite" if not finite[t, c] else "below_lower" if low[t, c] else "above_upper"
                entry["violations"].append({
                    "candidate": name, "time_step": int(t), "channel": int(c),
                    "actuator_name": names[c],
                    "value": float(value) if finite[t, c] else None,
                    "nonfinite_value": None if finite[t, c] else str(value),
                    "lower": float(limits[c, 0]), "upper": float(limits[c, 1]),
                    "reason": reason,
                    "excess": None if not finite[t, c] else float(limits[c, 0] - value if low[t, c] else value - limits[c, 1]),
                    "tolerance": 1e-7,
                })
            entry["valid_control_targets"] = not entry["violations"]
        (retained if entry["valid_control_targets"] else rejected).append(name)
        reports.append(entry)
    return {"candidate_reports": reports, "retained_candidate_names": retained,
            "rejected_candidate_names": rejected,
            "stop_required": not retained,
            "stop_reason": "no_legal_control_target_candidate" if not retained else None,
            "tolerance": 1e-7, "action_values_changed": False,
            "physical_safety_or_task_progress_proven": False}
