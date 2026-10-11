"""Finite Transport adapter for dynamically generated, publicly bound servo inputs.

The frozen 16x7 matrix identifies nominal geometry only. Every actual servo input
has its own binding, and only the actual executed matrix reaches the receipt.
No physical stepping, force observation, compensation calculation or clipping.
"""
import hashlib
from copy import deepcopy
import numpy as np
from transport_guard import TransportGuard


class FeedforwardTransportGuard(TransportGuard):
    def __init__(self, contract, nominal_commands, router, rules, prior_public_guard):
        super().__init__(contract, nominal_commands, router, rules, prior_public_guard)
        self.method_sha256 = contract.get('gravity_method_sha256')
        if not isinstance(self.method_sha256, str) or len(self.method_sha256) != 64:
            self.deny('feedforward_method_fingerprint_missing')
        self.parameters_sha256 = contract.get('gravity_parameters_sha256')
        if not isinstance(self.parameters_sha256, str) or len(self.parameters_sha256) != 64:
            self.deny('feedforward_parameters_fingerprint_missing')
        initial_servo = np.asarray(contract.get('initial_servo_command'), dtype=float)
        if initial_servo.shape != (7,) or not np.isfinite(initial_servo).all():
            self.deny('feedforward_initial_servo_binding_missing')
        self.bindings = []
        self.last_servo = initial_servo[:6].copy()

    def prepare_block(self, commands, t, wall, context):
        # Original finite block reservation, lease and nominal sequence checks.
        super().prepare_block(commands, t, wall, context)
        self.block['nominal_sha256'] = self.block.pop('sha256')
        self.block['actual_commands'] = []

    def bind_command(self, observation, nominal, servo, record, geometry_check, wall):
        t = float(observation['time'])
        self.observe(observation, wall)
        if not self.block or not self.router.inflight:
            self.deny('feedforward_command_without_block')
        step = self.block['steps']
        local_index = step // 40
        if step % 40 or local_index != len(self.block['actual_commands']):
            self.deny('feedforward_command_resampling_or_unsettled_prefix')
        if abs(t - (self.block['start'] + step * .001)) > 1e-7:
            self.deny('feedforward_command_time_invalid')
        index = self.cursor + local_index
        expected_nominal = self.commands[index]
        nominal = np.asarray(nominal)
        servo = np.asarray(servo)
        if (nominal.shape != (7,) or nominal.dtype != np.float32 or
                not np.array_equal(nominal, expected_nominal)):
            self.deny('feedforward_nominal_plan_changed')
        if servo.shape != (7,) or servo.dtype != np.float32 or not np.isfinite(servo).all():
            self.deny('feedforward_servo_invalid')
        state = np.asarray(observation['state'])
        if (record.get('command_index') != index or record.get('public_time_s') != t or
                record.get('method_sha256') != self.method_sha256 or
                record.get('parameters_sha256') != self.parameters_sha256 or
                not np.array_equal(np.asarray(record.get('q_public_rad')), state[:6]) or
                not np.array_equal(np.asarray(record.get('nominal_command')), nominal) or
                not np.array_equal(np.asarray(record.get('servo_command')), servo)):
            self.deny('feedforward_public_or_method_binding_invalid')
        if 'public_state' in record and not np.array_equal(np.asarray(record['public_state']), state):
            self.deny('feedforward_full_public_state_binding_invalid')
        if ('public_state_sha256' in record and record['public_state_sha256'] !=
                hashlib.sha256(state.astype(np.float32).tobytes()).hexdigest()):
            self.deny('feedforward_public_state_fingerprint_invalid')
        gravity = np.asarray(record.get('known_gravity_nm'), dtype=float)
        if gravity.shape != (6,) or not np.isfinite(gravity).all():
            self.deny('feedforward_gravity_invalid')
        expected_servo = nominal.copy()
        expected_servo[:6] = (nominal[:6].astype(float) + gravity / 100.).astype(np.float32)
        if not np.array_equal(servo, expected_servo) or servo[6] != np.float32(self.c['clamp_target_m']):
            self.deny('feedforward_law_or_clamp_changed')
        bounds = np.asarray(self.c['control_ranges'])
        limits = np.asarray(self.c['mechanical_ranges'])
        if np.any(servo < bounds[:, 0]) or np.any(servo > bounds[:, 1]):
            self.deny('feedforward_servo_control_range')
        if np.any(servo[:6] < limits[:, 0]) or np.any(servo[:6] > limits[:, 1]):
            self.deny('feedforward_servo_mechanical_range')
        if np.max(np.abs(servo[:6].astype(float) - self.last_servo)) > self.c['joint_target_delta_rad']:
            self.deny('feedforward_servo_sequence_discontinuous')
        if geometry_check is None:
            self.deny('transport_geometry_protection_missing')
        violations = geometry_check(servo[:6])
        if violations:
            self.deny('feedforward_servo_geometry:' + str(violations[0]))
        binding = deepcopy(record)
        binding['actual_command_sha256'] = hashlib.sha256(servo.tobytes()).hexdigest()
        binding['nominal_block_sha256'] = self.block['nominal_sha256']
        binding['lease_version'] = self.lease_version
        self.bindings.append(binding)
        self.block['actual_commands'].append(servo.copy())
        self.last_servo = servo[:6].astype(float)
        self.router.inflight['current_command_sha256'] = binding['actual_command_sha256']
        self.router.inflight['current_command_index'] = index
        self.router.log('dynamic_servo_bound', binding=binding)
        return servo.copy()

    def step(self, public_state, t, wall, geometry_check):
        # Same actual-state, lease, geometric and receipt-time protections as the
        # frozen Transport guard; only expected-command provenance differs.
        self.permission(t, wall)
        s = np.asarray(public_state)
        if s.shape != (21,) or not np.isfinite(s).all():
            self.deny('transport_encoder_invalid')
        if not 0 <= s[6] <= .08:
            self.deny('transport_gripper_mechanical_gap_range')
        p = self.router.inflight
        if (not self.block or not p or p['lease_version'] != self.lease_version or
                p['owners'] != self.c['owners'] or
                p.get('nominal_plan_sha256') != self.block['nominal_sha256'] or
                p.get('method_sha256') != self.method_sha256 or
                p.get('parameters_sha256') != self.parameters_sha256 or
                p.get('binding_kind') != 'public_state_dynamic_gravity_servo'):
            self.deny('transport_inflight_control_binding_invalid')
        expected_time = self.block['start'] + (self.block['steps'] + 1) * .001
        if abs(t - expected_time) > 1e-7:
            self.deny('transport_physics_receipt_time_invalid')
        if self.steps >= self.c['max_steps'] or self.block['steps'] >= 320:
            self.deny('transport_step_budget')
        if self.last_observation_time is None or t - self.last_observation_time > self.c['observation_dt_s'] + 1e-9:
            self.deny('transport_public_observation_expired')
        limits = np.asarray(self.c['mechanical_ranges'])
        q = s[:6]
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            self.deny('transport_actual_mechanical_range')
        local_index = self.block['steps'] // 40
        if len(self.block['actual_commands']) != local_index + 1:
            self.deny('feedforward_current_servo_not_bound')
        command = self.block['actual_commands'][local_index]
        if (p.get('current_command_index') != self.cursor + local_index or
                p.get('current_command_sha256') != hashlib.sha256(command.tobytes()).hexdigest() or
                not np.array_equal(s[14:21].astype(np.float32), command)):
            self.deny('transport_actual_command_mismatch')
        if geometry_check is None:
            self.deny('transport_geometry_protection_missing')
        violations = geometry_check(q)
        if violations:
            self.deny('transport_geometry:' + str(violations[0]))
        self.telemetry.append({'time_s': t, 'q_rad': q.tolist(),
                               'qvel_rad_s': s[7:13].tolist(),
                               'tracking_error_rad': (command[:6] - q).tolist(),
                               'nominal_tracking_error_rad': (self.commands[self.cursor + local_index, :6] - q).tolist(),
                               'dynamic_safety_limits_validated': False})
        self.steps += 1
        self.block['steps'] += 1
        self.last_step_time = t

    def actual_prefix_matches(self, commands, steps):
        if not self.block or steps not in (self.block['steps'], self.block['steps'] + (1 if self.reason else 0)):
            self.deny('transport_guard_receipt_mismatch')
        commands = np.asarray(commands)
        count = (steps + 39) // 40
        actual = np.asarray(self.block['actual_commands'][:count], dtype=np.float32).reshape(-1, 7)
        if (commands.shape != (count, 7) or commands.dtype != np.float32 or
                not np.array_equal(commands, actual)):
            self.deny('feedforward_actual_prefix_receipt_changed')
        if steps == 320 and count != 8:
            self.deny('feedforward_missing_actual_commands')


class FeedforwardExecution:
    def __init__(self, router, transport_guard):
        self.router = router
        self.transport_guard = transport_guard

    def start_block(self, nominal_commands, action_id, context, t, wall):
        g = self.transport_guard
        g.prepare_block(nominal_commands, t, wall, context)
        # Reuse frozen router health, ownership, clamp and reservation validation.
        self.router.begin(action_id, nominal_commands, self.router.lease['owners'], context, t, wall, None)
        p = self.router.inflight
        p['nominal_plan_sha256'] = p.pop('sha256')
        p['binding_kind'] = 'public_state_dynamic_gravity_servo'
        p['method_sha256'] = g.method_sha256
        p['parameters_sha256'] = g.parameters_sha256
        event = self.router.events[-1]
        event['kind'] = 'selected_nominal_plan_begin'
        event['nominal_plan_sha256'] = event.pop('sha256')
        event['actual_commands_generated_from_future_public_observations'] = False
        event['actual_command_binding'] = 'generated and bound separately at each current 40ms observation'
        return np.asarray(nominal_commands).copy()

    def bind_command(self, observation, nominal, servo, record, geometry_check, wall):
        return self.transport_guard.bind_command(observation, nominal, servo, record, geometry_check, wall)

    def observe_executed_step(self, public_state, t, wall, geometry_check):
        return self.transport_guard.step(public_state, t, wall, geometry_check)

    def settle_block(self, receipt_id, action_id, actual_commands, steps, end):
        g = self.transport_guard
        actual_commands = np.asarray(actual_commands)
        if (type(steps) is not int or not 0 <= steps <= 320 or
                actual_commands.shape != ((steps + 39) // 40, 7) or
                actual_commands.dtype != np.float32 or not np.isfinite(actual_commands).all()):
            raise RuntimeError('invalid_dynamic_servo_prefix_receipt')
        h = hashlib.sha256(actual_commands.tobytes()).hexdigest()
        facts = (action_id, h, steps, end)
        if receipt_id in self.router.receipts:
            if self.router.receipts[receipt_id] != facts:
                raise RuntimeError('conflicting_repeat_receipt')
            return False
        g.actual_prefix_matches(actual_commands, steps)
        p = self.router.inflight
        if (not p or p['id'] != action_id or p['lease_version'] != g.lease_version or
                p['owners'] != g.c['owners'] or
                abs(end - p['start_s'] - steps * .001) > 1e-7):
            raise RuntimeError('wrong_dynamic_execution_prefix')
        generated_count = len(g.block['actual_commands'])
        applied_count = (steps + 39) // 40
        self.router.log('actual_servo_matrix_finalized' if steps == 320 else 'actual_servo_prefix_finalized', action_id=action_id,
                        actual_servo_sha256=h, steps=steps,
                        generated_commands=generated_count, applied_commands=applied_count,
                        nominal_plan_sha256=p['nominal_plan_sha256'])
        if self.router.lease and steps:
            self.router.lease['used_blocks'] += 1
        self.router.receipts[receipt_id] = facts
        self.router.inflight = None
        self.router.log('actual_prefix_receipt', id=receipt_id, steps=steps, end_s=end,
                        actual_servo_sha256=h, complete_commands=steps // 40,
                        partial_command_steps=steps % 40)
        if 0 < steps < 320:
            self.router.revoke('partial_prefix_no_implicit_replay')
        g.settle(steps)
        return True
