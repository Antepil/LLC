"""Scale-only normalization for the fixed-load LLC neural ODE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn


_STATE_NAMES = ("i_Lr", "v_Cr", "i_Lm", "v_o")
_NETWORK_INPUT = (
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
_NETWORK_OUTPUT = (
    "dz_i_Lr_dtau",
    "dz_v_Cr_dtau",
    "dz_i_Lm_dtau",
    "dz_v_o_dtau",
)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _require_sequence(value: Any, *, expected: tuple[str, ...], name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if tuple(value) != expected:
        raise ValueError(f"{name} does not match the LLC normalization contract")


@dataclass(frozen=True)
class NormalizationConfig:
    """Validated normalization policy and, optionally, fitted state scales."""

    state_scale_rule: str
    state_scale_epsilon: float
    state_scales: tuple[float, float, float, float] | None
    vin_reference_v: float
    time_reference_s: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> NormalizationConfig:
        """Parse either a root configuration or its normalization section."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        raw = config.get("normalization", config)
        if not isinstance(raw, Mapping):
            raise TypeError("config.normalization must be a mapping")
        expected_keys = {
            "network_input",
            "network_output",
            "state_scale_rule",
            "state_scale_epsilon",
            "state_scales",
            "vin_reference_v",
            "time_reference_s",
        }
        missing = sorted(expected_keys.difference(raw))
        unknown = sorted(set(raw).difference(expected_keys))
        if missing:
            raise ValueError(
                f"normalization config is missing required keys: {', '.join(missing)}"
            )
        if unknown:
            raise ValueError(
                f"normalization config has unknown keys: {', '.join(unknown)}"
            )
        _require_sequence(
            raw["network_input"], expected=_NETWORK_INPUT, name="network_input"
        )
        _require_sequence(
            raw["network_output"], expected=_NETWORK_OUTPUT, name="network_output"
        )
        if raw["state_scale_rule"] != "train_rms_scale_only":
            raise ValueError(
                "normalization.state_scale_rule must be 'train_rms_scale_only'"
            )

        raw_scales = raw["state_scales"]
        scales: tuple[float, float, float, float] | None
        if raw_scales is None:
            scales = None
        else:
            if isinstance(raw_scales, (str, bytes)) or not isinstance(
                raw_scales, Sequence
            ):
                raise TypeError("normalization.state_scales must be a sequence or null")
            if len(raw_scales) != len(_STATE_NAMES):
                raise ValueError("normalization.state_scales must contain four values")
            parsed = tuple(
                _positive_float(value, name=f"state_scales[{index}]")
                for index, value in enumerate(raw_scales)
            )
            scales = (parsed[0], parsed[1], parsed[2], parsed[3])

        return cls(
            state_scale_rule="train_rms_scale_only",
            state_scale_epsilon=_positive_float(
                raw["state_scale_epsilon"], name="state_scale_epsilon"
            ),
            state_scales=scales,
            vin_reference_v=_positive_float(
                raw["vin_reference_v"], name="vin_reference_v"
            ),
            time_reference_s=_positive_float(
                raw["time_reference_s"], name="time_reference_s"
            ),
        )

    def fit(self, training_states: Tensor) -> LLCNormalizer:
        """Fit RMS scales from training states only and return a normalizer."""

        if self.state_scales is not None:
            raise ValueError("state scales are already resolved; refusing to refit them")
        _validate_state_tensor(training_states, name="training_states")
        reduction_dims = tuple(range(training_states.ndim - 1))
        scales = training_states.square().mean(dim=reduction_dims).sqrt()
        if bool((scales <= self.state_scale_epsilon).any()):
            failing = [
                name
                for name, scale in zip(_STATE_NAMES, scales, strict=True)
                if float(scale) <= self.state_scale_epsilon
            ]
            raise ValueError(
                "training RMS scale is too small for states: " + ", ".join(failing)
            )
        return LLCNormalizer(
            state_scales=scales,
            vin_reference_v=self.vin_reference_v,
            time_reference_s=self.time_reference_s,
        )

    def build(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> LLCNormalizer:
        """Build from previously resolved scales."""

        if self.state_scales is None:
            raise ValueError("normalization.state_scales is unresolved; fit training data first")
        return LLCNormalizer(
            state_scales=self.state_scales,
            vin_reference_v=self.vin_reference_v,
            time_reference_s=self.time_reference_s,
            dtype=dtype,
            device=device,
        )

    def resolved(self, normalizer: LLCNormalizer) -> NormalizationConfig:
        """Return the serializable policy containing fitted scales."""

        if not isinstance(normalizer, LLCNormalizer):
            raise TypeError("normalizer must be an LLCNormalizer")
        if normalizer.vin_reference_v != self.vin_reference_v:
            raise ValueError("normalizer Vin reference differs from the policy")
        if normalizer.time_reference_s != self.time_reference_s:
            raise ValueError("normalizer time reference differs from the policy")
        scales = tuple(float(value) for value in normalizer.state_scales.cpu())
        return replace(
            self,
            state_scales=(scales[0], scales[1], scales[2], scales[3]),
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return a checkpoint-safe normalization mapping."""

        return {
            "network_input": list(_NETWORK_INPUT),
            "network_output": list(_NETWORK_OUTPUT),
            "state_scale_rule": self.state_scale_rule,
            "state_scale_epsilon": self.state_scale_epsilon,
            "state_scales": (
                None if self.state_scales is None else list(self.state_scales)
            ),
            "vin_reference_v": self.vin_reference_v,
            "time_reference_s": self.time_reference_s,
        }


class LLCNormalizer(nn.Module):
    """Apply invertible scale-only transforms to LLC states, input and time."""

    def __init__(
        self,
        *,
        state_scales: Sequence[float] | Tensor,
        vin_reference_v: float,
        time_reference_s: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dtype is not None and dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if isinstance(state_scales, Tensor):
            if not state_scales.is_floating_point():
                raise TypeError("state_scales tensor must use a floating-point dtype")
            if dtype is None:
                dtype = state_scales.dtype
        elif isinstance(state_scales, (str, bytes)) or not isinstance(
            state_scales, Sequence
        ):
            raise TypeError("state_scales must be a tensor or sequence")
        if dtype is None:
            dtype = torch.float64
        scales = torch.as_tensor(state_scales, dtype=dtype, device=device)
        if scales.shape != (len(_STATE_NAMES),):
            raise ValueError("state_scales must contain exactly four values")
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("state_scales must be finite and strictly positive")
        self.register_buffer("state_scales", scales.detach().clone())
        self.vin_reference_v = _positive_float(
            vin_reference_v, name="vin_reference_v"
        )
        self.time_reference_s = _positive_float(
            time_reference_s, name="time_reference_s"
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.state_scales.dtype

    @property
    def device(self) -> torch.device:
        return self.state_scales.device

    def normalize_states(self, states: Tensor) -> Tensor:
        self._validate_tensor(states, name="states", final_dimension=4)
        return states / self.state_scales

    def denormalize_states(self, normalized_states: Tensor) -> Tensor:
        self._validate_tensor(
            normalized_states, name="normalized_states", final_dimension=4
        )
        return normalized_states * self.state_scales

    def normalize_vin(self, vin: Tensor) -> Tensor:
        self._validate_tensor(vin, name="vin")
        return vin / self.vin_reference_v

    def normalize_time(self, time_s: Tensor) -> Tensor:
        self._validate_tensor(time_s, name="time_s")
        return time_s / self.time_reference_s

    def normalize_state_derivatives(self, derivatives_per_s: Tensor) -> Tensor:
        """Convert ``dx/dt`` to the network target convention ``dz/dtau``."""

        self._validate_tensor(
            derivatives_per_s, name="derivatives_per_s", final_dimension=4
        )
        return derivatives_per_s * self.time_reference_s / self.state_scales

    def denormalize_state_derivatives(
        self, normalized_derivatives: Tensor
    ) -> Tensor:
        """Convert ``dz/dtau`` back to physical ``dx/dt``."""

        self._validate_tensor(
            normalized_derivatives,
            name="normalized_derivatives",
            final_dimension=4,
        )
        return normalized_derivatives * self.state_scales / self.time_reference_s

    def _validate_tensor(
        self, tensor: Tensor, *, name: str, final_dimension: int | None = None
    ) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.dtype != self.dtype:
            raise TypeError(f"{name} must use dtype {self.dtype}; got {tensor.dtype}")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on device {self.device}; got {tensor.device}")
        if final_dimension is not None and (
            tensor.ndim == 0 or tensor.shape[-1] != final_dimension
        ):
            raise ValueError(f"{name} must have final dimension {final_dimension}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")


def _validate_state_tensor(states: Tensor, *, name: str) -> None:
    if not isinstance(states, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not states.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if states.ndim < 2 or states.shape[-1] != len(_STATE_NAMES):
        raise ValueError(f"{name} must have shape [..., samples, 4]")
    if states.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(states).all()):
        raise ValueError(f"{name} must contain only finite values")
