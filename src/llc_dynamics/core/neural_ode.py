"""Neural vector field for the normalized LLC state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


class LLCNeuralODE(nn.Module):
    """Map normalized LLC state and inputs to ``dz/dtau``.

    The final input dimension has the fixed order
    ``[z_i_Lr, z_v_Cr, z_i_Lm, z_v_o, Q1, Q2, Q3, Q4, Vin/V_ref]``.
    Leading dimensions are preserved, so both ``[9]`` and ``[..., 9]``
    inputs are supported.
    """

    INPUT_FEATURES = (
        "z_i_Lr",
        "z_v_Cr",
        "z_i_Lm",
        "z_v_o",
        "Q1_A_high",
        "Q2_A_low",
        "Q3_B_high",
        "Q4_B_low",
        "Vin_over_V_ref",
    )
    OUTPUT_FEATURES = (
        "dz_i_Lr_dtau",
        "dz_v_Cr_dtau",
        "dz_i_Lm_dtau",
        "dz_v_o_dtau",
    )
    INPUT_DIM = len(INPUT_FEATURES)
    OUTPUT_DIM = len(OUTPUT_FEATURES)

    _ACTIVATIONS: dict[str, type[nn.Module]] = {"SiLU": nn.SiLU}
    _DTYPES: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "SiLU",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        validated_dims = self._validate_hidden_dims(hidden_dims)
        if activation not in self._ACTIVATIONS:
            supported = ", ".join(sorted(self._ACTIVATIONS))
            raise ValueError(
                f"unsupported activation {activation!r}; supported values: {supported}"
            )
        if dtype not in self._DTYPES.values():
            raise ValueError("dtype must be torch.float32 or torch.float64")
        resolved_device = torch.device(device) if device is not None else None
        if resolved_device is not None and resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {resolved_device} was requested but CUDA is unavailable"
                )

        layer_dims = (self.INPUT_DIM, *validated_dims, self.OUTPUT_DIM)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:], strict=True)
        ):
            layers.append(
                nn.Linear(input_dim, output_dim, dtype=dtype, device=resolved_device)
            )
            if index < len(layer_dims) - 2:
                layers.append(self._ACTIVATIONS[activation]())

        self.network = nn.Sequential(*layers)
        self.hidden_dims = validated_dims
        self.activation_name = activation
        self.reset_parameters()

    @staticmethod
    def _validate_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
        if isinstance(hidden_dims, (str, bytes)):
            raise TypeError("hidden_dims must be a sequence of positive integers")
        values = tuple(hidden_dims)
        if not values:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("hidden_dims must contain only integers")
        if any(value <= 0 for value in values):
            raise ValueError("hidden_dims values must be strictly positive")
        return values

    @property
    def dtype(self) -> torch.dtype:
        """Return the parameter dtype expected by ``forward``."""

        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Return the device expected by ``forward``."""

        return next(self.parameters()).device

    def reset_parameters(self) -> None:
        """Apply Xavier-uniform weights and zero biases to every linear layer."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Evaluate the normalized vector field on a single or batched input."""

        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim == 0 or inputs.shape[-1] != self.INPUT_DIM:
            actual = "scalar" if inputs.ndim == 0 else str(inputs.shape[-1])
            raise ValueError(
                f"inputs must have final dimension {self.INPUT_DIM}; got {actual}"
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")
        if inputs.dtype != self.dtype:
            raise TypeError(f"inputs must use dtype {self.dtype}; got {inputs.dtype}")
        if inputs.device != self.device:
            raise ValueError(f"inputs must be on device {self.device}; got {inputs.device}")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must contain only finite values")
        return self.network(inputs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LLCNeuralODE":
        """Build from the ``model`` mapping in ``configs/base.yaml``."""

        if not isinstance(config, Mapping):
            raise TypeError("model config must be a mapping")
        required = {
            "input_dim",
            "output_dim",
            "hidden_dims",
            "activation",
            "dtype",
            "device",
        }
        missing = sorted(required.difference(config))
        if missing:
            raise ValueError(f"model config is missing required keys: {', '.join(missing)}")
        if config["input_dim"] != cls.INPUT_DIM:
            raise ValueError(f"model.input_dim must be {cls.INPUT_DIM}")
        if config["output_dim"] != cls.OUTPUT_DIM:
            raise ValueError(f"model.output_dim must be {cls.OUTPUT_DIM}")

        dtype_name = config["dtype"]
        if dtype_name not in cls._DTYPES:
            supported = ", ".join(sorted(cls._DTYPES))
            raise ValueError(
                f"unsupported model.dtype {dtype_name!r}; supported values: {supported}"
            )

        cls._validate_initialization_config(config.get("initialization"))
        return cls(
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            dtype=cls._DTYPES[dtype_name],
            device=config["device"],
        )

    @staticmethod
    def _validate_initialization_config(config: Any) -> None:
        expected = {"weights": "xavier_uniform", "gain": 1.0, "bias": "zeros"}
        if not isinstance(config, Mapping):
            raise ValueError("model.initialization must be a mapping")
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"model.initialization.{key} must be {value!r}")


def build_neural_ode(config: Mapping[str, Any]) -> LLCNeuralODE:
    """Build the vector field from a root config or its ``model`` section."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    model_config = config.get("model", config)
    if not isinstance(model_config, Mapping):
        raise TypeError("config.model must be a mapping")
    return LLCNeuralODE.from_config(model_config)
