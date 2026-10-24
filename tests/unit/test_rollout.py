"""Tests for differentiable event-aware LLC rollouts."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch
from torch import Tensor, nn

from llc_dynamics.core import (
    RolloutSolverConfig,
    rollout_batch,
    rollout_window,
)
from llc_dynamics.support import RolloutWindow, collate_rollout_windows

DTYPE = torch.float64


class InputField(nn.Module):
    """Return a derivative selected by state, first gate, Vin, and a parameter."""

    def __init__(self, *, state_gain: float = 0.0, parameter_gain: float = 0.0) -> None:
        super().__init__()
        self.state_gain = state_gain
        self.weight = nn.Parameter(torch.tensor(parameter_gain, dtype=DTYPE))

    def forward(self, inputs: Tensor) -> Tensor:
        state = inputs[:4]
        value = self.state_gain * state + inputs[4] + inputs[8] + self.weight
        return value


def make_window(
    *,
    trajectory_id: str = "trajectory",
    tau: tuple[float, ...] = (0.0, 1.0),
    z0: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    initial_gates: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    event_tau: tuple[float, ...] = (),
    gates_after: tuple[tuple[float, ...], ...] = (),
    vin: tuple[float, ...] | None = None,
) -> RolloutWindow:
    time = torch.tensor(tau, dtype=DTYPE)
    initial = torch.tensor(z0, dtype=DTYPE)
    targets = torch.zeros((len(tau), 4), dtype=DTYPE)
    targets[0] = initial
    if vin is None:
        vin = tuple(0.0 for _ in tau)
    return RolloutWindow(
        trajectory_id=trajectory_id,
        start_index=7,
        start_time_s=1.0e-4,
        tau=time,
        z0=initial,
        target_z=targets,
        initial_gates=torch.tensor(initial_gates, dtype=DTYPE),
        event_tau=torch.tensor(event_tau, dtype=DTYPE),
        gates_after=torch.tensor(gates_after, dtype=DTYPE).reshape(-1, 4),
        vin_normalized=torch.tensor(vin, dtype=DTYPE),
    )


class RolloutSolverConfigTests(unittest.TestCase):
    def test_parses_strict_rk4_configuration(self) -> None:
        config = RolloutSolverConfig.from_mapping(
            {"solver": {"method": "rk4", "max_step_tau": 0.01}}
        )

        self.assertEqual(config, RolloutSolverConfig("rk4", 0.01))
        with self.assertRaisesRegex(ValueError, "must be 'rk4'"):
            RolloutSolverConfig("euler", 0.01)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            RolloutSolverConfig("rk4", 0.0)
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            RolloutSolverConfig.from_mapping(
                {"method": "rk4", "max_step_tau": 0.01, "adaptive": True}
            )


class EventAwareRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RolloutSolverConfig("rk4", 0.1)

    def test_constant_derivative_matches_analytic_solution(self) -> None:
        model = InputField(parameter_gain=2.0)
        window = make_window(tau=(0.0, 0.4, 1.0))

        result = rollout_window(model, window, self.config)

        expected = torch.tensor([[0.0] * 4, [0.8] * 4, [2.0] * 4], dtype=DTYPE)
        torch.testing.assert_close(result, expected, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(result[0], window.z0)

    def test_multiple_gate_events_split_one_output_interval(self) -> None:
        model = InputField()
        window = make_window(
            event_tau=(0.25, 0.75),
            gates_after=((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        )

        result = rollout_window(model, window, self.config)

        torch.testing.assert_close(result[-1], torch.full((4,), 0.5, dtype=DTYPE))

    def test_event_at_output_node_changes_only_following_interval(self) -> None:
        model = InputField()
        window = make_window(
            tau=(0.0, 0.5, 1.0),
            event_tau=(0.5,),
            gates_after=((1.0, 0.0, 0.0, 0.0),),
        )

        result = rollout_window(model, window, self.config)

        torch.testing.assert_close(
            result,
            torch.tensor([[0.0] * 4, [0.0] * 4, [0.5] * 4], dtype=DTYPE),
        )

    def test_rk4_queries_linearly_interpolated_vin_at_stage_times(self) -> None:
        model = InputField()
        window = make_window(vin=(0.0, 1.0))

        result = rollout_window(model, window, RolloutSolverConfig("rk4", 1.0))

        torch.testing.assert_close(result[-1], torch.full((4,), 0.5, dtype=DTYPE))

    def test_step_halving_shows_fourth_order_convergence(self) -> None:
        model = InputField(state_gain=1.0)
        window = make_window(z0=(1.0, 1.0, 1.0, 1.0))
        coarse = rollout_window(model, window, RolloutSolverConfig("rk4", 0.25))
        fine = rollout_window(model, window, RolloutSolverConfig("rk4", 0.125))
        exact = math.e
        coarse_error = abs(float(coarse[-1, 0]) - exact)
        fine_error = abs(float(fine[-1, 0]) - exact)

        self.assertGreater(coarse_error / fine_error, 12.0)

    def test_gradient_matches_analytic_parameter_gradient(self) -> None:
        model = InputField(parameter_gain=0.75)
        window = make_window(tau=(0.0, 2.0))

        result = rollout_window(model, window, self.config)
        loss = result[-1].sum()
        loss.backward()

        torch.testing.assert_close(model.weight.grad, torch.tensor(8.0, dtype=DTYPE))

    def test_gradient_reaches_initial_state(self) -> None:
        model = InputField(state_gain=1.0)
        window = make_window(z0=(1.0, 1.0, 1.0, 1.0))
        differentiable_z0 = window.z0.clone().requires_grad_()

        result = rollout_window(
            model, replace(window, z0=differentiable_z0), self.config
        )
        result[-1].sum().backward()

        self.assertIsNotNone(differentiable_z0.grad)
        self.assertTrue(bool(torch.isfinite(differentiable_z0.grad).all()))
        self.assertTrue(bool((differentiable_z0.grad > 0).all()))

    def test_prediction_does_not_read_future_targets(self) -> None:
        model = InputField(parameter_gain=1.0)
        window = make_window(tau=(0.0, 0.5, 1.0))
        changed = replace(window, target_z=torch.full_like(window.target_z, 999.0))

        original_result = rollout_window(model, window, self.config)
        changed_result = rollout_window(model, changed, self.config)

        torch.testing.assert_close(original_result, changed_result)

    def test_batch_matches_individual_rollouts_and_preserves_gradients(self) -> None:
        model = InputField(parameter_gain=0.5)
        first = make_window(trajectory_id="first", tau=(0.0, 0.5, 1.0))
        second = make_window(
            trajectory_id="second",
            tau=(0.0, 0.5, 1.0),
            event_tau=(0.25,),
            gates_after=((1.0, 0.0, 0.0, 0.0),),
        )
        batch = collate_rollout_windows([first, second])

        result = rollout_batch(model, batch, self.config)

        expected = torch.stack(
            [
                rollout_window(model, first, self.config),
                rollout_window(model, second, self.config),
            ]
        )
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(model.weight.grad)))

    def test_rejects_invalid_time_gate_and_model_contracts(self) -> None:
        model = InputField()
        cases = (
            (
                replace(make_window(), tau=torch.tensor([0.1, 1.0], dtype=DTYPE)),
                "start at zero",
            ),
            (
                replace(make_window(), tau=torch.tensor([0.0, 0.0], dtype=DTYPE)),
                "strictly increasing",
            ),
            (
                replace(
                    make_window(),
                    initial_gates=torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=DTYPE),
                ),
                "binary",
            ),
            (
                make_window(event_tau=(1.0,), gates_after=((1.0, 0.0, 0.0, 0.0),)),
                "strictly inside",
            ),
        )
        for window, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                rollout_window(model, window, self.config)

        wrong_dtype = InputField().float()
        with self.assertRaisesRegex(TypeError, "parameters must match"):
            rollout_window(wrong_dtype, make_window(), self.config)

    def test_reports_context_for_nonfinite_dynamics(self) -> None:
        class NonfiniteField(nn.Module):
            def forward(self, inputs: Tensor) -> Tensor:
                return torch.full((4,), float("nan"), dtype=inputs.dtype)

        with self.assertRaisesRegex(
            FloatingPointError, "trajectory.*window start index 7.*tau="
        ):
            rollout_window(NonfiniteField(), make_window(), self.config)

    def test_rejects_inconsistent_batch_metadata(self) -> None:
        window = make_window()
        batch = collate_rollout_windows([window])
        malformed = replace(batch, event_tau=())

        with self.assertRaisesRegex(ValueError, "event tables"):
            rollout_batch(InputField(), malformed, self.config)


if __name__ == "__main__":
    unittest.main()
