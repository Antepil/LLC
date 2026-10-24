"""Tests for strict MATLAB trajectory loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from llc_dynamics.support import MATLABTrajectory, load_matlab_trajectory


class MATLABTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "trajectory.mat"
        self.path.touch()

    @staticmethod
    def fixed_trajectory() -> dict[str, object]:
        return {
            "schema_version": "llc-closed-loop-fixed-grid-trajectory-v3",
            "trajectory_id": "fixed-vin-example",
            "state_names": np.array(["i_Lr", "v_Cr", "i_Lm", "v_o"], dtype=object),
            "state_times_s": np.array([0.0, 1.0e-7, 2.0e-7]),
            "state_values": np.array(
                [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
            ),
            "gate_names": np.array(
                ["Q1_A_high", "Q2_A_low", "Q3_B_high", "Q4_B_low"],
                dtype=object,
            ),
            "initial_gate_values": np.array([0, 0, 0, 0], dtype=np.uint8),
            "gate_event_times_s": np.array([5.0e-8, 1.5e-7]),
            "gate_values_after": np.array(
                [[1, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8
            ),
            "condition": {"vin_v": 400.0, "load_ohm": 2.304},
        }

    def load(self, raw: dict[str, object], **kwargs: object) -> MATLABTrajectory:
        with patch(
            "llc_dynamics.support.adapters.matlab_trajectory._load_mat_file",
            return_value=raw,
        ):
            return load_matlab_trajectory(self.path, **kwargs)

    def test_loads_fixed_vin_trajectory_into_readonly_canonical_arrays(self) -> None:
        trajectory = self.load(self.fixed_trajectory())

        self.assertIsInstance(trajectory, MATLABTrajectory)
        self.assertEqual(trajectory.trajectory_id, "fixed-vin-example")
        self.assertEqual(trajectory.states.shape, (3, 4))
        np.testing.assert_array_equal(trajectory.vin_times_s, trajectory.times_s)
        np.testing.assert_array_equal(trajectory.vin_values_v, [400.0, 400.0, 400.0])
        np.testing.assert_array_equal(trajectory.initial_gates, [0, 0, 0, 0])
        self.assertFalse(trajectory.states.flags.writeable)
        self.assertFalse(trajectory.gates_after.flags.writeable)

    def test_loads_dynamic_vin_from_named_excitation_column(self) -> None:
        raw = self.fixed_trajectory()
        raw.update(
            {
                "schema_version": "llc-sinusoidal-excitation-fixed-grid-trajectory-v1",
                "state_excitation_names": np.array(
                    ["vin_v", "reference_voltage_v"], dtype=object
                ),
                "state_excitation_values": np.array(
                    [[390.0, 48.0], [400.0, 48.0], [410.0, 48.0]]
                ),
                "fixed_load_ohm": 2.304,
            }
        )
        del raw["condition"]

        trajectory = self.load(raw)

        np.testing.assert_array_equal(trajectory.vin_values_v, [390.0, 400.0, 410.0])

    def test_can_explicitly_accept_other_loads_for_future_conditioned_models(self) -> None:
        raw = self.fixed_trajectory()
        raw["condition"] = {"vin_v": 400.0, "load_ohm": 1.8432}

        trajectory = self.load(raw, expected_load_ohm=None)

        self.assertEqual(trajectory.load_resistance_ohm, 1.8432)

    def test_rejects_load_outside_fixed_load_contract(self) -> None:
        raw = self.fixed_trajectory()
        raw["condition"] = {"vin_v": 400.0, "load_ohm": 2.7648}

        with self.assertRaisesRegex(ValueError, "fixed-load contract"):
            self.load(raw)

    def test_rejects_missing_required_field(self) -> None:
        raw = self.fixed_trajectory()
        del raw["gate_event_times_s"]

        with self.assertRaisesRegex(ValueError, "gate_event_times_s"):
            self.load(raw)

    def test_rejects_state_contract_reordering(self) -> None:
        raw = self.fixed_trajectory()
        raw["state_names"] = ["v_Cr", "i_Lr", "i_Lm", "v_o"]

        with self.assertRaisesRegex(ValueError, "state_names must equal"):
            self.load(raw)

    def test_rejects_nonchanging_gate_event(self) -> None:
        raw = self.fixed_trajectory()
        raw["gate_values_after"] = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )

        with self.assertRaisesRegex(ValueError, "must change"):
            self.load(raw)

    def test_rejects_bridge_leg_overlap(self) -> None:
        raw = self.fixed_trajectory()
        raw["gate_values_after"] = np.array(
            [[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )

        with self.assertRaisesRegex(ValueError, "bridge-leg overlap"):
            self.load(raw)

    def test_rejects_gate_event_outside_state_coverage(self) -> None:
        raw = self.fixed_trajectory()
        raw["gate_event_times_s"] = np.array([5.0e-8, 3.0e-7])

        with self.assertRaisesRegex(ValueError, "state time range"):
            self.load(raw)

    def test_rejects_dynamic_vin_with_wrong_length(self) -> None:
        raw = self.fixed_trajectory()
        raw.update(
            {
                "schema_version": "llc-sinusoidal-excitation-fixed-grid-trajectory-v1",
                "state_excitation_names": ["vin_v", "reference_voltage_v"],
                "state_excitation_values": [[400.0, 48.0]],
                "fixed_load_ohm": 2.304,
            }
        )
        del raw["condition"]

        with self.assertRaisesRegex(ValueError, "rows must match"):
            self.load(raw)


if __name__ == "__main__":
    unittest.main()
