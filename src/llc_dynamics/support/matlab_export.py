"""Export the LLC neural vector field for native MATLAB evaluation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from llc_dynamics.core.neural_ode import LLCNeuralODE, build_neural_ode


_SCHEMA = "llc-neural-ode-matlab-export-v1"
_STATE_NAMES = ("i_Lr", "v_Cr", "i_Lm", "v_o")
_GATE_NAMES = ("Q1_A_high", "Q2_A_low", "Q3_B_high", "Q4_B_low")


def export_matlab_model(
    destination: str | Path,
    *,
    model: LLCNeuralODE,
    run_config: Mapping[str, Any],
    consistency_inputs: Tensor | None = None,
) -> Path:
    """Write one v7 MAT file containing the network and physical contract."""

    if not isinstance(model, LLCNeuralODE):
        raise TypeError("model must be an LLCNeuralODE")
    config = _validated_config(run_config)
    path = _destination(destination)
    inputs = _reference_inputs(model, consistency_inputs)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(inputs)
    finally:
        model.train(was_training)

    layers = [module for module in model.network if isinstance(module, nn.Linear)]
    weights = np.empty((1, len(layers)), dtype=object)
    biases = np.empty((1, len(layers)), dtype=object)
    activations = np.empty((1, len(layers)), dtype=object)
    for index, layer in enumerate(layers):
        weights[0, index] = layer.weight.detach().cpu().numpy().astype(np.float64)
        biases[0, index] = (
            layer.bias.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
        )
        activations[0, index] = (
            model.activation_name if index < len(layers) - 1 else "linear"
        )

    normalization = config["normalization"]
    payload = {
        "schema_version": _SCHEMA,
        "network": {
            "input_dim": model.INPUT_DIM,
            "output_dim": model.OUTPUT_DIM,
            "weights": weights,
            "biases": biases,
            "activations": activations,
        },
        "state_names": np.asarray(_STATE_NAMES, dtype=object).reshape(1, -1),
        "gate_names": np.asarray(_GATE_NAMES, dtype=object).reshape(1, -1),
        "network_input_order": np.asarray(model.INPUT_FEATURES, dtype=object).reshape(1, -1),
        "network_output_order": np.asarray(model.OUTPUT_FEATURES, dtype=object).reshape(1, -1),
        "state_scales": np.asarray(normalization["state_scales"], dtype=np.float64).reshape(1, -1),
        "vin_reference_v": float(normalization["vin_reference_v"]),
        "time_reference_s": float(normalization["time_reference_s"]),
        "load_resistance_ohm": float(config["condition"]["load_resistance_ohm"]),
        "dtype": str(model.dtype).removeprefix("torch."),
        "consistency_inputs": inputs.detach().cpu().numpy().astype(np.float64),
        "consistency_outputs": outputs.detach().cpu().numpy().astype(np.float64),
    }
    try:
        from scipy.io import savemat
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("SciPy is required to export MATLAB model files") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, payload, do_compression=True, long_field_names=True)
    return path


def export_checkpoint_to_matlab(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    consistency_inputs: Tensor | None = None,
) -> Path:
    """Load a trusted project checkpoint and export its model for MATLAB."""

    source = Path(checkpoint).expanduser().resolve()
    if source.suffix != ".ckpt" or not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist or is not .ckpt: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"failed to load trusted checkpoint {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    required = {"schema_version", "model_state", "run_config"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema version")
    run_config = _validated_config(payload["run_config"])
    cpu_config = copy.deepcopy(run_config)
    cpu_config["model"] = dict(cpu_config["model"])
    cpu_config["model"]["device"] = "cpu"
    model = build_neural_ode(cpu_config)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError("checkpoint model state is incompatible") from error
    return export_matlab_model(
        destination,
        model=model,
        run_config=run_config,
        consistency_inputs=consistency_inputs,
    )


def _destination(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("destination must be a string or pathlib.Path")
    result = Path(value).expanduser().resolve()
    if result.suffix.lower() != ".mat":
        raise ValueError("destination must have a .mat suffix")
    return result


def _validated_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("run_config must be a mapping")
    for section in ("model", "normalization", "condition"):
        if not isinstance(value.get(section), Mapping):
            raise ValueError(f"run_config.{section} must be a mapping")
    result = copy.deepcopy(dict(value))
    normalization = result["normalization"]
    scales = normalization.get("state_scales")
    if not isinstance(scales, (list, tuple)) or len(scales) != 4:
        raise ValueError("run_config.normalization.state_scales must be resolved")
    positive = [
        *scales,
        normalization.get("vin_reference_v"),
        normalization.get("time_reference_s"),
        result["condition"].get("load_resistance_ohm"),
    ]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not np.isfinite(item)
        or item <= 0
        for item in positive
    ):
        raise ValueError("normalization references and fixed load must be finite and positive")
    return result


def _reference_inputs(model: LLCNeuralODE, value: Tensor | None) -> Tensor:
    if value is None:
        value = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0.2, -0.5, 0.1, 0.8, 1, 0, 0, 1, 0.9],
                [-1, 0.25, 0.75, -0.1, 0, 1, 1, 0, 1.1],
            ],
            dtype=model.dtype,
            device=model.device,
        )
    if not isinstance(value, Tensor):
        raise TypeError("consistency_inputs must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != model.INPUT_DIM:
        raise ValueError("consistency_inputs must have shape [N, 9] with N > 0")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("consistency_inputs must be finite floating-point values")
    return value.to(dtype=model.dtype, device=model.device)
