"""Tests for trajectory window indexing, event clipping, and batching."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from llc_dynamics.support import (
    LLCNormalizer,
    MATLABTrajectory,
    TrajectoryWindowDataset,
    WindowSpec,
    collate_rollout_windows,
)


def trajectory(
    *,
    trajectory_id: str = "example",
    event_times: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> MATLABTrajectory:
    times = np.arange(6, dtype=np.float64)
    states = np.column_stack((times, 2 * times, 3 * times, 4 * times))
    event_states = {
        1.0: [1, 0, 0, 1],
        2.0: [0, 0, 0, 0],
        4.0: [0, 1, 1, 0],
    }
    gates_after = np.asarray(
        [event_states[value] for value in event_times], dtype=np.uint8
    ).reshape(-1, 4)
    return MATLABTrajectory(
        source_path=Path("/tmp/example.mat"),
        schema_version="llc-closed-loop-fixed-grid-trajectory-v3",
        trajectory_id=trajectory_id,
        times_s=times,
        states=states,
        initial_gates=np.zeros(4, dtype=np.uint8),
        event_times_s=np.asarray(event_times, dtype=np.float64),
        gates_after=gates_after,
        vin_times_s=times.copy(),
        vin_values_v=400.0 + 10.0 * times,
        load_resistance_ohm=2.304,
    )


class WindowSpecTests(unittest.TestCase):
    def test_builds_from_root_config_and_rejects_contract_drift(self) -> None:
        spec = WindowSpec.from_mapping(
            {"window": {"length_intervals": 100, "stride_intervals": 25}}
        )

        self.assertEqual(spec, WindowSpec(100, 25))
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            WindowSpec.from_mapping({"length_intervals": 100})
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            WindowSpec.from_mapping(
                {
                    "length_intervals": 100,
                    "stride_intervals": 25,
                    "pad_tail": True,
                }
            )

    def test_requires_positive_integer_length_and_stride(self) -> None:
        for arguments, error_type in (
            ({"length_intervals": True, "stride_intervals": 1}, TypeError),
            ({"length_intervals": 2, "stride_intervals": 0}, ValueError),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(error_type):
                    WindowSpec(**arguments)


class TrajectoryWindowDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = LLCNormalizer(
            state_scales=[1.0, 2.0, 3.0, 4.0],
            vin_reference_v=400.0,
            time_reference_s=2.0,
            dtype=torch.float64,
        )

    def dataset(
        self,
        source: MATLABTrajectory | None = None,
        *,
        length: int = 2,
        stride: int = 2,
    ) -> TrajectoryWindowDataset:
        return TrajectoryWindowDataset(
            [trajectory() if source is None else source],
            normalizer=self.normalizer,
            spec=WindowSpec(length, stride),
        )

    def test_indexes_only_complete_windows_and_reports_tail(self) -> None:
        dataset = self.dataset(length=2, stride=2)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.window_counts, (2,))
        self.assertEqual(dataset.trailing_intervals, (1,))
        self.assertEqual([dataset[index].start_index for index in range(2)], [0, 2])

    def test_normalizes_state_time_and_vin_without_teacher_forcing(self) -> None:
        window = self.dataset()[0]

        torch.testing.assert_close(
            window.tau, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            window.target_z,
            torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0] * 4],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(window.z0, window.target_z[0])
        self.assertEqual(window.z0.data_ptr(), window.target_z[0].data_ptr())
        torch.testing.assert_close(
            window.vin_normalized,
            torch.tensor([1.0, 1.025, 1.05], dtype=torch.float64),
        )

    def test_start_event_sets_initial_gate_and_end_event_is_excluded(self) -> None:
        window = self.dataset()[1]  # physical interval [2, 4)

        torch.testing.assert_close(
            window.initial_gates,
            torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        self.assertEqual(window.event_tau.numel(), 0)
        self.assertEqual(window.gates_after.shape, (0, 4))

    def test_keeps_only_strictly_internal_events_with_relative_time(self) -> None:
        dataset = self.dataset(length=3, stride=1)
        window = dataset[1]  # physical interval [1, 4)

        torch.testing.assert_close(
            window.initial_gates,
            torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            window.event_tau, torch.tensor([0.5], dtype=torch.float64)
        )
        torch.testing.assert_close(
            window.gates_after,
            torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        )

    def test_handles_window_with_no_gate_events(self) -> None:
        source = trajectory(event_times=())
        window = self.dataset(source)[0]

        torch.testing.assert_close(
            window.initial_gates, torch.zeros(4, dtype=torch.float64)
        )
        self.assertEqual(window.event_tau.shape, (0,))
        self.assertEqual(window.gates_after.shape, (0, 4))

    def test_rejects_duplicate_ids_short_trajectory_and_gpu_normalizer(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate trajectory_id"):
            TrajectoryWindowDataset(
                [trajectory(), trajectory()],
                normalizer=self.normalizer,
                spec=WindowSpec(2, 1),
            )
        with self.assertRaisesRegex(ValueError, "fewer than window length"):
            self.dataset(length=6)
        if torch.cuda.is_available():
            gpu_normalizer = LLCNormalizer(
                state_scales=[1.0] * 4,
                vin_reference_v=400.0,
                time_reference_s=1.0,
                device="cuda",
            )
            with self.assertRaisesRegex(ValueError, "must be on CPU"):
                TrajectoryWindowDataset(
                    [trajectory()],
                    normalizer=gpu_normalizer,
                    spec=WindowSpec(2, 1),
                )

    def test_collate_stacks_fixed_fields_and_keeps_ragged_events(self) -> None:
        dataset = self.dataset(length=2, stride=1)
        batch = collate_rollout_windows([dataset[0], dataset[1]])

        self.assertEqual(batch.target_z.shape, (2, 3, 4))
        self.assertEqual(batch.vin_normalized.shape, (2, 3))
        self.assertEqual(batch.initial_gates.shape, (2, 4))
        self.assertEqual(len(batch.event_tau), 2)
        self.assertEqual([events.numel() for events in batch.event_tau], [1, 1])
        moved = batch.to("cpu")
        torch.testing.assert_close(moved.target_z, batch.target_z)

    def test_collate_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            collate_rollout_windows([])


if __name__ == "__main__":
    unittest.main()
