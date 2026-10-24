"""Tests for fixed-load LLC normalization."""

import unittest

import torch

from llc_dynamics.support import LLCNormalizer, NormalizationConfig


NORMALIZATION_CONFIG = {
    "network_input": [
        "z_i_Lr",
        "z_v_Cr",
        "z_i_Lm",
        "z_v_o",
        "Q1_A_high",
        "Q2_A_low",
        "Q3_B_high",
        "Q4_B_low",
        "Vin_over_V_ref",
    ],
    "network_output": [
        "dz_i_Lr_dtau",
        "dz_v_Cr_dtau",
        "dz_i_Lm_dtau",
        "dz_v_o_dtau",
    ],
    "state_scale_rule": "train_rms_scale_only",
    "state_scale_epsilon": 1.0e-12,
    "state_scales": None,
    "vin_reference_v": 400.0,
    "time_reference_s": 1.0 / 110000.0,
}


class NormalizationTests(unittest.TestCase):
    def test_fit_uses_uncentered_training_rms_per_state(self) -> None:
        config = NormalizationConfig.from_mapping(NORMALIZATION_CONFIG)
        states = torch.tensor(
            [
                [[3.0, 0.0, 5.0, 8.0], [4.0, 12.0, 0.0, 6.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        normalizer = config.fit(states)
        expected = states.square().mean(dim=(0, 1)).sqrt()
        self.assertTrue(torch.equal(normalizer.state_scales, expected))

    def test_state_and_derivative_transforms_are_invertible(self) -> None:
        normalizer = LLCNormalizer(
            state_scales=[2.0, 4.0, 5.0, 10.0],
            vin_reference_v=400.0,
            time_reference_s=0.25,
        )
        states = torch.tensor(
            [[[2.0, -8.0, 15.0, 40.0]]], dtype=torch.float64
        )
        derivatives = torch.tensor(
            [[[4.0, -8.0, 20.0, 80.0]]], dtype=torch.float64
        )
        normalized_states = normalizer.normalize_states(states)
        normalized_derivatives = normalizer.normalize_state_derivatives(derivatives)

        self.assertTrue(
            torch.equal(normalizer.denormalize_states(normalized_states), states)
        )
        self.assertTrue(
            torch.equal(
                normalizer.denormalize_state_derivatives(normalized_derivatives),
                derivatives,
            )
        )
        self.assertTrue(
            torch.equal(
                normalizer.normalize_vin(torch.tensor([400.0], dtype=torch.float64)),
                torch.ones(1, dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                normalizer.normalize_time(torch.tensor([0.5], dtype=torch.float64)),
                torch.tensor([2.0], dtype=torch.float64),
            )
        )

    def test_resolved_config_round_trips_and_is_checkpoint_safe(self) -> None:
        unresolved = NormalizationConfig.from_mapping(NORMALIZATION_CONFIG)
        normalizer = LLCNormalizer(
            state_scales=[1.0, 2.0, 3.0, 4.0],
            vin_reference_v=400.0,
            time_reference_s=1.0 / 110000.0,
        )
        resolved = unresolved.resolved(normalizer)
        rebuilt = NormalizationConfig.from_mapping(resolved.as_mapping()).build()
        self.assertTrue(torch.equal(rebuilt.state_scales, normalizer.state_scales))
        self.assertEqual(resolved.as_mapping()["state_scales"], [1.0, 2.0, 3.0, 4.0])

    def test_unresolved_config_cannot_build_and_resolved_config_cannot_refit(self) -> None:
        unresolved = NormalizationConfig.from_mapping(NORMALIZATION_CONFIG)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            unresolved.build()
        resolved_mapping = dict(NORMALIZATION_CONFIG, state_scales=[1.0] * 4)
        resolved = NormalizationConfig.from_mapping(resolved_mapping)
        with self.assertRaisesRegex(ValueError, "refit"):
            resolved.fit(torch.ones(2, 4, dtype=torch.float64))

    def test_invalid_training_states_and_runtime_inputs_are_rejected(self) -> None:
        config = NormalizationConfig.from_mapping(NORMALIZATION_CONFIG)
        states = torch.ones(3, 4, dtype=torch.float64)
        states[:, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "i_Lm"):
            config.fit(states)

        normalizer = LLCNormalizer(
            state_scales=[1.0] * 4,
            vin_reference_v=400.0,
            time_reference_s=1.0,
        )
        with self.assertRaisesRegex(TypeError, "dtype"):
            normalizer.normalize_states(torch.ones(2, 4, dtype=torch.float32))
        invalid = torch.ones(2, 4, dtype=torch.float64)
        invalid[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            normalizer.normalize_states(invalid)


if __name__ == "__main__":
    unittest.main()
