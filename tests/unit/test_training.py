"""Tests for training configuration, normalization, and optimization epochs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from llc_dynamics.core import LLCNeuralODE, NormalizedRolloutMSE, RolloutSolverConfig
from llc_dynamics.support import (
    LLCNormalizer,
    MATLABTrajectory,
    NormalizationConfig,
    TrajectoryWindowDataset,
    WindowSpec,
    collate_rollout_windows,
)
from llc_dynamics.training import (
    TrainingConfig,
    fit_training_normalizer,
    run_epoch,
    train_from_config,
)


def trajectory(identifier: str, multiplier: float = 1.0) -> MATLABTrajectory:
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    base = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2]],
        dtype=np.float64,
    )
    return MATLABTrajectory(
        source_path=Path(f"/tmp/{identifier}.mat"),
        schema_version="llc-continuous-random-excitation-fixed-grid-trajectory-v1",
        trajectory_id=identifier,
        times_s=times,
        states=base * multiplier,
        initial_gates=np.zeros(4, dtype=np.uint8),
        event_times_s=np.empty(0, dtype=np.float64),
        gates_after=np.empty((0, 4), dtype=np.uint8),
        vin_times_s=times,
        vin_values_v=np.full(3, 400.0),
        load_resistance_ohm=2.304,
    )


def normalization_config() -> NormalizationConfig:
    return NormalizationConfig.from_mapping(
        {
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
            "time_reference_s": 1.0,
        }
    )


def root_config(*, epochs: int) -> dict[str, object]:
    normalization = normalization_config().as_mapping()
    return {
        "model": {
            "input_dim": 9,
            "output_dim": 4,
            "hidden_dims": [4],
            "activation": "SiLU",
            "dtype": "float64",
            "device": "cpu",
            "initialization": {
                "weights": "xavier_uniform",
                "gain": 1.0,
                "bias": "zeros",
            },
        },
        "normalization": normalization,
        "condition": {"load_resistance_ohm": 2.304},
        "window": {"length_intervals": 2, "stride_intervals": 2},
        "solver": {"method": "rk4", "max_step_tau": 0.1},
        "loss": {
            "type": "normalized_rollout_mse",
            "state_dim": 4,
            "exclude_initial_state": True,
            "reduction": "mean",
        },
        "training": {
            "data_directory": "data/raw/formal",
            "file_pattern": "*.mat",
            "split_directories": {
                "train": "train",
                "validation": "validation",
                "test": "test",
            },
            "epochs": epochs,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "seed": 7,
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
            },
            "gradient_clip_norm": 1.0,
            "resume_last": True,
        },
        "checkpoint": {
            "directory": "artifacts/checkpoints",
            "topk": {
                "monitor_key": "val_rollout_mse",
                "mode": "min",
                "k": 2,
                "format_str": "epoch={epoch:04d}-val={val_rollout_mse:.6e}.ckpt",
            },
            "save_last_ckpt": True,
            "save_last_snapshot": False,
        },
    }


class TrainingConfigTests(unittest.TestCase):
    def test_parses_strict_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = TrainingConfig.from_mapping(
                {
                    "training": {
                        "data_directory": "data/raw",
                        "file_pattern": "*.mat",
                        "split_directories": {
                            "train": "train",
                            "validation": "validation",
                            "test": "test",
                        },
                        "epochs": 2,
                        "batch_size": 4,
                        "num_workers": 0,
                        "pin_memory": False,
                        "seed": 7,
                        "optimizer": {
                            "type": "AdamW",
                            "learning_rate": 1.0e-3,
                            "weight_decay": 0.0,
                        },
                        "gradient_clip_norm": 1.0,
                        "resume_last": True,
                    }
                },
                project_root=Path(temporary),
            )
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.optimizer_type, "AdamW")

    def test_fit_uses_all_training_nodes_only(self) -> None:
        normalizer = fit_training_normalizer(
            normalization_config(), [trajectory("a"), trajectory("b", 2.0)]
        )
        all_states = np.concatenate(
            [trajectory("a").states, trajectory("b", 2.0).states], axis=0
        )
        expected = np.sqrt(np.mean(all_states**2, axis=0))
        np.testing.assert_allclose(normalizer.state_scales.numpy(), expected)


class EpochTests(unittest.TestCase):
    def test_training_updates_parameters_and_evaluation_does_not(self) -> None:
        source = trajectory("train")
        normalizer = LLCNormalizer(
            state_scales=[1.0, 2.0, 3.0, 4.0],
            vin_reference_v=400.0,
            time_reference_s=1.0,
            dtype=torch.float64,
        )
        dataset = TrajectoryWindowDataset(
            [source], normalizer=normalizer, spec=WindowSpec(2, 2)
        )
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_rollout_windows)
        torch.manual_seed(3)
        model = LLCNeuralODE(hidden_dims=[4], device="cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        mse, steps = run_epoch(
            model=model,
            loader=loader,
            solver=RolloutSolverConfig("rk4", 0.1),
            loss_function=NormalizedRolloutMSE(),
            device=torch.device("cpu"),
            optimizer=optimizer,
            gradient_clip_norm=1.0,
        )

        self.assertTrue(np.isfinite(mse))
        self.assertEqual(steps, 1)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )
        trained = [parameter.detach().clone() for parameter in model.parameters()]
        eval_mse, eval_steps = run_epoch(
            model=model,
            loader=loader,
            solver=RolloutSolverConfig("rk4", 0.1),
            loss_function=NormalizedRolloutMSE(),
            device=torch.device("cpu"),
        )
        self.assertTrue(np.isfinite(eval_mse))
        self.assertEqual(eval_steps, 0)
        self.assertTrue(
            all(torch.equal(old, new) for old, new in zip(trained, model.parameters()))
        )

    def test_end_to_end_training_saves_and_resumes_checkpoint(self) -> None:
        splits = {
            "train": (trajectory("train"),),
            "validation": (trajectory("validation"),),
            "test": (trajectory("test"),),
        }
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            with patch(
                "llc_dynamics.training.load_trajectory_splits",
                return_value=splits,
            ):
                first = train_from_config(
                    root_config(epochs=1), project_root=project_root
                )
                resumed = train_from_config(
                    root_config(epochs=2), project_root=project_root
                )

            self.assertEqual(first.start_epoch, 0)
            self.assertEqual(first.completed_epochs, 1)
            self.assertEqual(resumed.start_epoch, 1)
            self.assertEqual(resumed.completed_epochs, 1)
            self.assertEqual(resumed.global_step, 2)
            self.assertTrue(
                (project_root / "artifacts/checkpoints/last.ckpt").is_file()
            )
            self.assertIn("test_rollout_mse", resumed.final_metrics)


if __name__ == "__main__":
    unittest.main()
