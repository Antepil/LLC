"""Tests for normalized LLC rollout losses."""

import unittest

import torch

from llc_dynamics.core import NormalizedRolloutMSE, build_rollout_loss


LOSS_CONFIG = {
    "type": "normalized_rollout_mse",
    "state_dim": 4,
    "exclude_initial_state": True,
    "reduction": "mean",
}


class NormalizedRolloutMSETests(unittest.TestCase):
    def test_excludes_initial_node_and_averages_all_future_elements(self) -> None:
        loss = NormalizedRolloutMSE()
        target = torch.zeros(2, 3, 4, dtype=torch.float64)
        predicted = torch.zeros_like(target)
        predicted[:, 0] = 1000.0
        predicted[0, 1, 0] = 2.0
        predicted[1, 2, 3] = 4.0

        actual = loss(predicted, target)
        expected = torch.tensor((2.0**2 + 4.0**2) / 16, dtype=torch.float64)
        self.assertTrue(torch.equal(actual, expected))

    def test_preserves_gradient_to_predicted_future_states(self) -> None:
        loss = NormalizedRolloutMSE()
        predicted = torch.ones(2, 5, 4, dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(predicted)
        loss(predicted, target).backward()
        self.assertTrue(torch.equal(predicted.grad[:, 0], torch.zeros(2, 4)))
        self.assertTrue(torch.all(predicted.grad[:, 1:] > 0))

    def test_builds_only_the_fixed_loss_contract(self) -> None:
        self.assertIsInstance(build_rollout_loss({"loss": LOSS_CONFIG}), NormalizedRolloutMSE)
        with self.assertRaisesRegex(ValueError, "exclude_initial_state"):
            build_rollout_loss(dict(LOSS_CONFIG, exclude_initial_state=False))

    def test_rejects_bad_shapes_types_devices_and_nonfinite_values(self) -> None:
        loss = NormalizedRolloutMSE()
        valid = torch.zeros(2, 3, 4, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "equal shapes"):
            loss(valid, torch.zeros(2, 4, 4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "initial and predicted"):
            loss(valid[:, :1], valid[:, :1])
        with self.assertRaisesRegex(TypeError, "same dtype"):
            loss(valid, valid.float())
        invalid = valid.clone()
        invalid[0, 1, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            loss(invalid, valid)


if __name__ == "__main__":
    unittest.main()
