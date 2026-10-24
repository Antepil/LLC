"""Tests for the LLC neural vector field."""

import math
import unittest

import torch
from torch import nn

from llc_dynamics.core import LLCNeuralODE, build_neural_ode


MODEL_CONFIG = {
    "input_dim": 9,
    "output_dim": 4,
    "hidden_dims": [128, 128],
    "activation": "SiLU",
    "dtype": "float64",
    "device": "cpu",
    "initialization": {
        "weights": "xavier_uniform",
        "gain": 1.0,
        "bias": "zeros",
    },
}


class LLCNeuralODETests(unittest.TestCase):
    def test_config_builds_expected_architecture(self) -> None:
        model = build_neural_ode({"model": MODEL_CONFIG})
        linear_layers = [
            module for module in model.network if isinstance(module, nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(9, 128), (128, 128), (128, 4)],
        )
        self.assertEqual(sum(isinstance(m, nn.SiLU) for m in model.network), 2)
        self.assertEqual(model.dtype, torch.float64)

    def test_forward_preserves_leading_dimensions_and_supports_gradients(self) -> None:
        model = LLCNeuralODE()
        inputs = torch.randn(2, 5, 9, dtype=torch.float64, requires_grad=True)
        outputs = model(inputs)
        outputs.square().mean().backward()
        self.assertEqual(outputs.shape, (2, 5, 4))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(outputs).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_and_backward(self) -> None:
        config = dict(MODEL_CONFIG, device="cuda:0")
        model = LLCNeuralODE.from_config(config)
        inputs = torch.randn(
            16, 9, dtype=torch.float64, device="cuda:0", requires_grad=True
        )

        outputs = model(inputs)
        outputs.sum().backward()

        self.assertEqual(outputs.device.type, "cuda")
        self.assertEqual(model.device.type, "cuda")
        self.assertIsNotNone(inputs.grad)

    def test_initialization_is_xavier_uniform_with_zero_bias(self) -> None:
        torch.manual_seed(42)
        model = LLCNeuralODE()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                bound = math.sqrt(6.0 / (module.in_features + module.out_features))
                self.assertTrue(torch.all(module.weight <= bound))
                self.assertTrue(torch.all(module.weight >= -bound))
                self.assertTrue(torch.count_nonzero(module.weight) > 0)
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_forward_rejects_invalid_inputs(self) -> None:
        model = LLCNeuralODE()
        with self.assertRaisesRegex(ValueError, "final dimension 9"):
            model(torch.zeros(3, 8, dtype=torch.float64))
        with self.assertRaisesRegex(TypeError, "dtype"):
            model(torch.zeros(3, 9, dtype=torch.float32))
        invalid = torch.zeros(3, 9, dtype=torch.float64)
        invalid[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            model(invalid)

    def test_config_rejects_dimension_and_initialization_drift(self) -> None:
        bad_dimension = dict(MODEL_CONFIG, input_dim=10)
        with self.assertRaisesRegex(ValueError, "input_dim must be 9"):
            LLCNeuralODE.from_config(bad_dimension)
        bad_initialization = dict(
            MODEL_CONFIG,
            initialization={"weights": "default", "gain": 1.0, "bias": "zeros"},
        )
        with self.assertRaisesRegex(ValueError, "xavier_uniform"):
            LLCNeuralODE.from_config(bad_initialization)


if __name__ == "__main__":
    unittest.main()
