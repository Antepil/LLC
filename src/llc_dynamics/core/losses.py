"""Training losses for normalized LLC state rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class NormalizedRolloutMSE(nn.Module):
    """Mean squared error over predicted rollout nodes after the initial state."""

    STATE_DIM = 4

    def forward(self, predicted_states: Tensor, target_states: Tensor) -> Tensor:
        """Compare normalized ``[batch, time, 4]`` state sequences.

        Time index zero is excluded because a free rollout is initialized with the
        measured state and therefore has zero error at that node by construction.
        """

        self._validate_inputs(predicted_states, target_states)
        return torch.mean((predicted_states[:, 1:] - target_states[:, 1:]).square())

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NormalizedRolloutMSE:
        """Build from a root configuration or its loss section."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        raw = config.get("loss", config)
        if not isinstance(raw, Mapping):
            raise TypeError("config.loss must be a mapping")
        expected = {"type", "state_dim", "exclude_initial_state", "reduction"}
        missing = sorted(expected.difference(raw))
        unknown = sorted(set(raw).difference(expected))
        if missing:
            raise ValueError(f"loss config is missing required keys: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"loss config has unknown keys: {', '.join(unknown)}")
        required = {
            "type": "normalized_rollout_mse",
            "state_dim": cls.STATE_DIM,
            "exclude_initial_state": True,
            "reduction": "mean",
        }
        for key, value in required.items():
            if raw[key] != value:
                raise ValueError(f"loss.{key} must be {value!r}")
        return cls()

    @classmethod
    def _validate_inputs(cls, predicted: Tensor, target: Tensor) -> None:
        if not isinstance(predicted, Tensor) or not isinstance(target, Tensor):
            raise TypeError("predicted_states and target_states must be tensors")
        if predicted.shape != target.shape:
            raise ValueError("predicted_states and target_states must have equal shapes")
        if predicted.ndim != 3 or predicted.shape[-1] != cls.STATE_DIM:
            raise ValueError("state rollouts must have shape [batch, time, 4]")
        if predicted.shape[0] == 0:
            raise ValueError("state rollouts must contain at least one batch item")
        if predicted.shape[1] < 2:
            raise ValueError("state rollouts must contain an initial and predicted node")
        if not predicted.is_floating_point() or not target.is_floating_point():
            raise TypeError("state rollouts must use floating-point dtypes")
        if predicted.dtype != target.dtype:
            raise TypeError("predicted_states and target_states must use the same dtype")
        if predicted.device != target.device:
            raise ValueError("predicted_states and target_states must share a device")
        if not bool(torch.isfinite(predicted).all()) or not bool(
            torch.isfinite(target).all()
        ):
            raise ValueError("state rollouts must contain only finite values")


def build_rollout_loss(config: Mapping[str, Any]) -> NormalizedRolloutMSE:
    """Build the fixed-load rollout objective from configuration."""

    return NormalizedRolloutMSE.from_config(config)
