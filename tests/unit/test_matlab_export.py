from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from llc_dynamics.core import LLCNeuralODE
from llc_dynamics.support import export_checkpoint_to_matlab, export_matlab_model


def run_config() -> dict[str, object]:
    return {
        "model": {
            "input_dim": 9,
            "output_dim": 4,
            "hidden_dims": [3],
            "activation": "SiLU",
            "dtype": "float64",
            "device": "cpu",
            "initialization": {
                "weights": "xavier_uniform",
                "gain": 1.0,
                "bias": "zeros",
            },
        },
        "normalization": {
            "state_scales": [2.0, 3.0, 4.0, 5.0],
            "vin_reference_v": 400.0,
            "time_reference_s": 1.0e-5,
        },
        "condition": {"load_resistance_ohm": 2.304},
    }


class MATLABExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        torch.manual_seed(42)
        self.model = LLCNeuralODE(hidden_dims=[3], device="cpu")

    def test_export_round_trips_network_contract_and_reference_outputs(self) -> None:
        destination = self.directory / "model.mat"
        export_matlab_model(destination, model=self.model, run_config=run_config())
        raw = loadmat(destination, simplify_cells=True)

        self.assertEqual(raw["schema_version"], "llc-neural-ode-matlab-export-v1")
        np.testing.assert_array_equal(
            raw["state_names"], ["i_Lr", "v_Cr", "i_Lm", "v_o"]
        )
        np.testing.assert_allclose(raw["state_scales"], [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(raw["load_resistance_ohm"], 2.304)
        network = raw["network"]
        self.assertEqual(network["input_dim"], 9)
        self.assertEqual(network["output_dim"], 4)
        self.assertEqual(list(network["activations"]), ["SiLU", "linear"])
        inputs = torch.as_tensor(raw["consistency_inputs"], dtype=torch.float64)
        expected = self.model(inputs).detach().numpy()
        np.testing.assert_allclose(
            raw["consistency_outputs"], expected, rtol=1.0e-15, atol=1.0e-15
        )

    def test_rejects_unresolved_normalization(self) -> None:
        config = run_config()
        config["normalization"]["state_scales"] = None  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "state_scales"):
            export_matlab_model(
                self.directory / "model.mat", model=self.model, run_config=config
            )

    def test_exports_trusted_checkpoint_without_optimizer_objects(self) -> None:
        checkpoint = self.directory / "model.ckpt"
        torch.save(
            {
                "schema_version": 1,
                "model_state": self.model.state_dict(),
                "run_config": run_config(),
            },
            checkpoint,
        )
        destination = self.directory / "from-checkpoint.mat"
        export_checkpoint_to_matlab(checkpoint, destination)
        raw = loadmat(destination, simplify_cells=True)
        inputs = torch.as_tensor(raw["consistency_inputs"], dtype=torch.float64)
        np.testing.assert_allclose(
            raw["consistency_outputs"], self.model(inputs).detach().numpy()
        )


if __name__ == "__main__":
    unittest.main()
