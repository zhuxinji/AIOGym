from __future__ import annotations

import json

import numpy as np
import pytest

import aiogym.experimental.three_tank_hardware as hardware
from aiogym.scenarios import three_tank
from aiogym.experimental.three_tank_hardware.calibration import (
    calibration_template,
    validate_calibration,
)
from aiogym.experimental.three_tank_hardware.hardware import (
    HardwareSample,
    SafetyConfig,
    SafetyGuardian,
    ThreeTankHardwareEnv,
)


def _calibration(*, measured):
    record = calibration_template(rig_id="lab-three-tank")
    if measured:
        record["calibration_id"] = "calibration-001"
        record["status"] = "validated"
        record["fit_dataset_ids"] = ["fit-001"]
        record["validation_dataset_ids"] = ["validation-001"]
        record["validation_metrics"] = {"normalized_rmse": 0.01}
        record.pop("calibration_hash")
        for row in record["parameters"].values():
            row["estimate"] = 1.0
            row["standard_uncertainty"] = 0.01
            row["source_dataset"] = "fit-001"
        record = validate_calibration(record, require_measured=True)
    return record


def test_hardware_api_is_only_exported_from_experimental_package():
    assert not hasattr(three_tank, "ThreeTankHardwareEnv")
    assert hardware.ThreeTankHardwareEnv is ThreeTankHardwareEnv
    assert hardware.SafetyConfig is SafetyConfig
    assert hardware.calibration_template is calibration_template
    assert callable(hardware.validate_real_step_record)


def test_hardware_sample_uses_only_three_level_measurements():
    sample = HardwareSample(
        measurement=(0.225, 0.225, 0.225),
        flow_measurement=(0.0, 0.0, 0.0, 0.0),
        source_monotonic_time_s=0.0,
        received_monotonic_time_s=0.1,
        wall_time_utc="2026-08-08T00:00:00Z",
        interlocks={
            "watchdog_healthy": True,
            "emergency_stop": False,
            "reservoir_available": True,
        },
    )
    assert sample.measurement == pytest.approx((0.225, 0.225, 0.225))
    assert sample.boundary == {}


def _sample(index=0, *, applied=None, interlocks=None, measurement=None):
    return HardwareSample(
        measurement=tuple(
            measurement or [0.225, 0.225, 0.225]
        ),
        flow_measurement=tuple([3.0 / 60000.0] * 4),
        source_monotonic_time_s=float(index),
        received_monotonic_time_s=float(index) + 0.1,
        wall_time_utc="2026-08-08T00:00:00Z",
        applied_action=None if applied is None else tuple(applied),
        boundary={},
        interlocks=interlocks
        or {
            "watchdog_healthy": True,
            "emergency_stop": False,
            "reservoir_available": True,
        },
        raw_channels={"LT301": 1234 + index},
    )


@pytest.mark.parametrize(
    "missing",
    ["watchdog_healthy", "emergency_stop", "reservoir_available"],
)
def test_hardware_sample_requires_every_safety_interlock(missing):
    interlocks = {
        "watchdog_healthy": True,
        "emergency_stop": False,
        "reservoir_available": True,
    }
    del interlocks[missing]
    with pytest.raises(ValueError, match="missing required fields"):
        HardwareSample(
            measurement=(0.225, 0.225, 0.225),
            flow_measurement=(0.0, 0.0, 0.0, 0.0),
            source_monotonic_time_s=0.0,
            received_monotonic_time_s=0.1,
            wall_time_utc="2026-08-08T00:00:00Z",
            boundary={},
            interlocks=interlocks,
        )


class FakeTransport:
    def __init__(self, samples):
        self.samples = list(samples)
        self.actions = []
        self.closed = False

    def reset(self):
        return self.samples.pop(0)

    def exchange(self, action):
        self.actions.append(action)
        return self.samples.pop(0)

    def close(self):
        self.closed = True


def test_shadow_mode_never_writes_recommended_action():
    transport = FakeTransport(
        [
            _sample(0, applied=[0.45, 0.24, 0.24, 0.24]),
            _sample(1, applied=[0.45, 0.24, 0.24, 0.24]),
        ]
    )
    env = ThreeTankHardwareEnv(
        transport,
        mode="shadow",
        calibration=_calibration(measured=False),
    )
    try:
        observation, info = env.reset(seed=0)
        assert observation.shape == (10,)
        assert observation[3:7] == pytest.approx(
            np.asarray([3.0 / 60000.0] * 4)
            / np.asarray(env.model.parameter("flow_observation_scale"))
        )
        assert info["backend_kind"] == "real"
        _, reward, terminated, truncated, next_info = env.step([0.0] * 4)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert transport.actions == [None]
        assert next_info["hardware_mode"] == "shadow"
        assert next_info["resolved_action"].shape == (4,)
    finally:
        env.close()
    assert transport.closed


def test_hardware_environment_writes_self_identifying_rows(tmp_path):
    transport = FakeTransport(
        [
            _sample(0, applied=[0.45, 0.24, 0.24, 0.24]),
            _sample(1, applied=[0.45, 0.24, 0.24, 0.24]),
        ]
    )
    path = tmp_path / "real.jsonl"
    env = ThreeTankHardwareEnv(
        transport,
        mode="shadow",
        calibration=_calibration(measured=False),
        log_writer=hardware.RealLogWriter(path),
    )
    try:
        env.reset()
        _, _, _, _, info = env.step([0.5, 0.4, 0.3, 0.2])
    finally:
        env.close()
    row = json.loads(path.read_text(encoding="utf-8"))
    assert hardware.validate_real_step_record(row) == row
    assert row["schema_version"].endswith(".v10")
    assert row["measurement"] == pytest.approx([0.225, 0.225, 0.225])
    assert row["commanded_action"] == pytest.approx([0.5, 0.4, 0.3, 0.2])
    assert row["hardware_mode"] == "shadow"
    assert row["scenario_id"] == info["scenario_id"] == "three_tank"
    assert row["benchmark_id"] == info["benchmark_id"] == "tracking"
    assert row["reward_id"] == info["reward_id"] == "regulation"
    assert row["calibration_hash"] == info["calibration_hash"]
    assert row["transition_disturbance"] == info["transition_disturbance"]
    assert row["disturbance"] == info["disturbance"]


def test_closed_loop_requires_arming_and_measured_calibration():
    with pytest.raises(ValueError, match="armed=True"):
        ThreeTankHardwareEnv(
            FakeTransport([_sample()]),
            mode="closed-loop",
            calibration=_calibration(measured=False),
        )
    with pytest.raises(ValueError, match="not measured"):
        ThreeTankHardwareEnv(
            FakeTransport([_sample()]),
            mode="closed-loop",
            armed=True,
            calibration=_calibration(measured=False),
        )

def test_guardian_is_fail_closed_and_interlocks_bypass_slew_limits():
    guardian = SafetyGuardian(
        SafetyConfig(maximum_action_step=(0.01, 0.01, 0.01, 0.01))
    )
    unhealthy = _sample(
        interlocks={
            "watchdog_healthy": False,
            "emergency_stop": False,
            "reservoir_available": True,
        }
    )
    decision = guardian.apply(
        [1.0] * 4,
        sample=unhealthy,
        previous_action=[0.5] * 4,
    )
    assert decision.emergency
    assert decision.applied_action == guardian.config.emergency_action

    process_trip = _sample(measurement=[0.42, 0.225, 0.225])
    decision = guardian.apply(
        [1.0] * 4,
        sample=process_trip,
        previous_action=[0.5] * 4,
    )
    assert not decision.emergency
    assert decision.applied_action[0] == 0.0


def test_default_hardware_guardian_keeps_its_independent_slew_limit():
    assert SafetyConfig().maximum_action_step == (0.05, 0.08, 0.08, 0.08)


def test_closed_loop_emergency_is_written_and_terminates():
    unhealthy = {
        "watchdog_healthy": False,
        "emergency_stop": False,
        "reservoir_available": True,
    }
    transport = FakeTransport(
        [
            _sample(0, applied=[0.4] * 4, interlocks=unhealthy),
            _sample(1, applied=[0.0, 0.0, 0.0, 1.0], interlocks=unhealthy),
        ]
    )
    env = ThreeTankHardwareEnv(
        transport,
        mode="closed-loop",
        armed=True,
        calibration=_calibration(measured=True),
    )
    try:
        env.reset()
        _, _, terminated, _, info = env.step([0.0] * 4)
        assert terminated
        assert transport.actions[0] == env.guardian.config.emergency_action
        assert info["safety"]["emergency"]
    finally:
        env.close()


def test_hardware_policy_command_is_a_direct_four_actuator_action():
    transport = FakeTransport(
        [
            _sample(0, applied=[0.45, 0.24, 0.24, 0.24]),
            _sample(1, applied=[0.45, 0.24, 0.24, 0.24]),
        ]
    )
    env = ThreeTankHardwareEnv(
        transport,
        mode="shadow",
        calibration=_calibration(measured=False),
    )
    try:
        env.reset()
        command = np.asarray([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
        _, _, _, _, info = env.step(command)
    finally:
        env.close()
    assert env.action_space.shape == (4,)
    assert info["commanded_action"] == pytest.approx(command)
    assert info["resolved_action"] == pytest.approx(command)


def test_hardware_episode_terminates_on_a_measured_hard_process_limit():
    transport = FakeTransport(
        [
            _sample(0, applied=[0.4] * 4),
            _sample(
                1,
                applied=[0.4] * 4,
                measurement=[0.51, 0.225, 0.225],
            ),
        ]
    )
    env = ThreeTankHardwareEnv(
        transport,
        mode="shadow",
        calibration=_calibration(measured=False),
    )
    try:
        env.reset()
        _, _, terminated, _, info = env.step([0.0] * 4)
        assert terminated
        assert info["constraint_costs"] == {"tank_overflow_limit": 1.0}
        assert transport.actions == [None]
    finally:
        env.close()


def test_hardware_rejects_an_infeasible_reference_before_writing_an_action():
    transport = FakeTransport(
        [
            _sample(0, applied=[0.4] * 4),
            _sample(1, applied=[0.0, 0.0, 0.0, 1.0]),
        ]
    )
    env = ThreeTankHardwareEnv(
        transport,
        mode="closed-loop",
        armed=True,
        calibration=_calibration(measured=True),
    )
    try:
        env.reset()
        env._reference_state[1] = 80.0
        _, _, terminated, _, info = env.step([1.0] * 4)
        assert terminated
        assert transport.actions == [env.guardian.config.emergency_action]
        assert info["safety"]["reasons"] == ["infeasible_reference"]
        assert not info["reference_feasibility"]["accepted"]
        assert not info["transition_reference_feasibility"]["accepted"]
    finally:
        env.close()
