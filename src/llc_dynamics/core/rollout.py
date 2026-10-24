"""Differentiable event-aware rollouts for normalized LLC dynamics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from llc_dynamics.support.windowing import RolloutBatch, RolloutWindow


@dataclass(frozen=True, slots=True)
class RolloutSolverConfig:
    """Fixed-step solver policy for event-aware rollouts."""

    method: str
    max_step_tau: float

    def __post_init__(self) -> None:
        if self.method != "rk4":
            raise ValueError("solver.method must be 'rk4'")
        if isinstance(self.max_step_tau, bool) or not isinstance(
            self.max_step_tau, Real
        ):
            raise TypeError("solver.max_step_tau must be a real number")
        value = float(self.max_step_tau)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("solver.max_step_tau must be finite and strictly positive")
        object.__setattr__(self, "max_step_tau", value)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> RolloutSolverConfig:
        """Build from a root configuration or its ``solver`` section."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        raw = config.get("solver", config)
        if not isinstance(raw, Mapping):
            raise TypeError("config.solver must be a mapping")
        expected = {"method", "max_step_tau"}
        missing = sorted(expected.difference(raw))
        unknown = sorted(set(raw).difference(expected))
        if missing:
            raise ValueError(
                f"solver config is missing required keys: {', '.join(missing)}"
            )
        if unknown:
            raise ValueError(f"solver config has unknown keys: {', '.join(unknown)}")
        return cls(method=raw["method"], max_step_tau=raw["max_step_tau"])


def rollout_window(
    model: nn.Module,
    window: RolloutWindow,
    config: RolloutSolverConfig,
) -> Tensor:
    """Roll out one window and return normalized states with shape ``[T, 4]``."""

    if not isinstance(window, RolloutWindow):
        raise TypeError("window must be a RolloutWindow")
    return _rollout(
        model=model,
        config=config,
        trajectory_id=window.trajectory_id,
        start_index=window.start_index,
        tau=window.tau,
        z0=window.z0,
        initial_gates=window.initial_gates,
        event_tau=window.event_tau,
        gates_after=window.gates_after,
        vin_normalized=window.vin_normalized,
    )


def rollout_batch(
    model: nn.Module,
    batch: RolloutBatch,
    config: RolloutSolverConfig,
) -> Tensor:
    """Roll out ragged-event windows and stack them as ``[B, T, 4]``."""

    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(config, RolloutSolverConfig):
        raise TypeError("config must be a RolloutSolverConfig")
    batch_size = len(batch.trajectory_ids)
    if batch_size == 0:
        raise ValueError("batch must contain at least one window")
    fixed = {
        "tau": batch.tau,
        "z0": batch.z0,
        "initial_gates": batch.initial_gates,
        "vin_normalized": batch.vin_normalized,
    }
    expected_leading = {
        "tau": batch_size,
        "z0": batch_size,
        "initial_gates": batch_size,
        "vin_normalized": batch_size,
    }
    for name, tensor in fixed.items():
        if not isinstance(tensor, Tensor) or tensor.ndim == 0:
            raise ValueError(f"batch.{name} must have a batch dimension")
        if tensor.shape[0] != expected_leading[name]:
            raise ValueError(f"batch.{name} batch dimension is inconsistent")
    if (
        len(batch.event_tau) != batch_size
        or len(batch.gates_after) != batch_size
        or batch.start_indices.shape != (batch_size,)
    ):
        raise ValueError("batch metadata and event tables are inconsistent")

    predictions = [
        _rollout(
            model=model,
            config=config,
            trajectory_id=batch.trajectory_ids[index],
            start_index=int(batch.start_indices[index]),
            tau=batch.tau[index],
            z0=batch.z0[index],
            initial_gates=batch.initial_gates[index],
            event_tau=batch.event_tau[index],
            gates_after=batch.gates_after[index],
            vin_normalized=batch.vin_normalized[index],
        )
        for index in range(batch_size)
    ]
    try:
        return torch.stack(predictions)
    except RuntimeError as error:  # pragma: no cover - guarded by batch validation
        raise ValueError(
            "batch windows must produce matching rollout shapes"
        ) from error


def _rollout(
    *,
    model: nn.Module,
    config: RolloutSolverConfig,
    trajectory_id: str,
    start_index: int,
    tau: Tensor,
    z0: Tensor,
    initial_gates: Tensor,
    event_tau: Tensor,
    gates_after: Tensor,
    vin_normalized: Tensor,
) -> Tensor:
    _validate_inputs(
        model=model,
        config=config,
        tau=tau,
        z0=z0,
        initial_gates=initial_gates,
        event_tau=event_tau,
        gates_after=gates_after,
        vin_normalized=vin_normalized,
    )
    node_times = tau.detach().cpu().tolist()
    event_times = event_tau.detach().cpu().tolist()
    state = z0
    gates = initial_gates
    event_index = 0
    predictions = [state]

    for interval_index in range(len(node_times) - 1):
        interval_start = node_times[interval_index]
        interval_stop = node_times[interval_index + 1]

        while (
            event_index < len(event_times)
            and event_times[event_index] == interval_start
        ):
            gates = gates_after[event_index]
            event_index += 1

        segment_start = tau[interval_index]
        while (
            event_index < len(event_times) and event_times[event_index] < interval_stop
        ):
            segment_stop = event_tau[event_index]
            state = _integrate_segment(
                model=model,
                state=state,
                gates=gates,
                start=segment_start,
                stop=segment_stop,
                vin_start_tau=tau[interval_index],
                vin_stop_tau=tau[interval_index + 1],
                vin_start=vin_normalized[interval_index],
                vin_stop=vin_normalized[interval_index + 1],
                max_step_tau=config.max_step_tau,
                context=(trajectory_id, start_index),
            )
            gates = gates_after[event_index]
            event_index += 1
            segment_start = segment_stop

        state = _integrate_segment(
            model=model,
            state=state,
            gates=gates,
            start=segment_start,
            stop=tau[interval_index + 1],
            vin_start_tau=tau[interval_index],
            vin_stop_tau=tau[interval_index + 1],
            vin_start=vin_normalized[interval_index],
            vin_stop=vin_normalized[interval_index + 1],
            max_step_tau=config.max_step_tau,
            context=(trajectory_id, start_index),
        )
        predictions.append(state)

    if event_index != len(event_times):
        raise RuntimeError("internal error: not all gate events were consumed")
    return torch.stack(predictions)


def _integrate_segment(
    *,
    model: nn.Module,
    state: Tensor,
    gates: Tensor,
    start: Tensor,
    stop: Tensor,
    vin_start_tau: Tensor,
    vin_stop_tau: Tensor,
    vin_start: Tensor,
    vin_stop: Tensor,
    max_step_tau: float,
    context: tuple[str, int],
) -> Tensor:
    length = float((stop - start).detach().cpu())
    if length < 0.0:
        raise RuntimeError("internal error: rollout segment has negative duration")
    if length == 0.0:
        return state
    step_count = max(1, math.ceil(length / max_step_tau))
    step = (stop - start) / step_count
    time = start
    result = state
    for _ in range(step_count):
        result = _rk4_step(
            model=model,
            state=result,
            gates=gates,
            time=time,
            step=step,
            vin_start_tau=vin_start_tau,
            vin_stop_tau=vin_stop_tau,
            vin_start=vin_start,
            vin_stop=vin_stop,
            context=context,
        )
        time = time + step
    return result


def _rk4_step(
    *,
    model: nn.Module,
    state: Tensor,
    gates: Tensor,
    time: Tensor,
    step: Tensor,
    vin_start_tau: Tensor,
    vin_stop_tau: Tensor,
    vin_start: Tensor,
    vin_stop: Tensor,
    context: tuple[str, int],
) -> Tensor:
    half_step = step / 2

    def field(query_state: Tensor, query_time: Tensor) -> Tensor:
        fraction = (query_time - vin_start_tau) / (vin_stop_tau - vin_start_tau)
        vin = vin_start + fraction * (vin_stop - vin_start)
        inputs = torch.cat((query_state, gates, vin.reshape(1)))
        derivative = model(inputs)
        if not isinstance(derivative, Tensor) or derivative.shape != (4,):
            raise ValueError("model must return a tensor with shape [4]")
        if derivative.dtype != state.dtype or derivative.device != state.device:
            raise ValueError("model output must match rollout dtype and device")
        if not bool(torch.isfinite(derivative).all()):
            _raise_nonfinite(context, query_time, "model derivative")
        return derivative

    k1 = field(state, time)
    k2 = field(state + half_step * k1, time + half_step)
    k3 = field(state + half_step * k2, time + half_step)
    k4 = field(state + step * k3, time + step)
    result = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not bool(torch.isfinite(result).all()):
        _raise_nonfinite(context, time + step, "integrated state")
    return result


def _raise_nonfinite(context: tuple[str, int], time: Tensor, quantity: str) -> None:
    trajectory_id, start_index = context
    value = float(time.detach().cpu())
    raise FloatingPointError(
        f"nonfinite {quantity} for trajectory {trajectory_id!r}, "
        f"window start index {start_index}, tau={value:.17g}"
    )


def _validate_inputs(
    *,
    model: nn.Module,
    config: RolloutSolverConfig,
    tau: Tensor,
    z0: Tensor,
    initial_gates: Tensor,
    event_tau: Tensor,
    gates_after: Tensor,
    vin_normalized: Tensor,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(config, RolloutSolverConfig):
        raise TypeError("config must be a RolloutSolverConfig")
    tensors = {
        "tau": tau,
        "z0": z0,
        "initial_gates": initial_gates,
        "event_tau": event_tau,
        "gates_after": gates_after,
        "vin_normalized": vin_normalized,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.dtype != z0.dtype:
            raise TypeError(f"{name} must use dtype {z0.dtype}; got {tensor.dtype}")
        if tensor.device != z0.device:
            raise ValueError(
                f"{name} must be on device {z0.device}; got {tensor.device}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")

    if tau.ndim != 1 or tau.numel() < 2:
        raise ValueError("tau must contain at least two nodes")
    if float(tau[0].detach().cpu()) != 0.0:
        raise ValueError("tau must start at zero")
    if not bool((tau[1:] > tau[:-1]).all()):
        raise ValueError("tau must be strictly increasing")
    if z0.shape != (4,):
        raise ValueError("z0 must have shape [4]")
    if vin_normalized.shape != tau.shape:
        raise ValueError("vin_normalized must have one value per tau node")
    if initial_gates.shape != (4,):
        raise ValueError("initial_gates must have shape [4]")
    if event_tau.ndim != 1:
        raise ValueError("event_tau must be one-dimensional")
    if gates_after.shape != (event_tau.numel(), 4):
        raise ValueError("gates_after must have shape [event_count, 4]")
    if event_tau.numel():
        if not bool((event_tau[1:] > event_tau[:-1]).all()):
            raise ValueError("event_tau must be strictly increasing")
        if not bool((event_tau > 0).all()) or not bool((event_tau < tau[-1]).all()):
            raise ValueError("gate events must lie strictly inside the window")
    if not bool(((initial_gates == 0) | (initial_gates == 1)).all()) or not bool(
        ((gates_after == 0) | (gates_after == 1)).all()
    ):
        raise ValueError("gate values must be binary")

    for parameter in model.parameters():
        if parameter.dtype != z0.dtype:
            raise TypeError("model parameters must match rollout dtype")
        if parameter.device != z0.device:
            raise ValueError("model parameters must match rollout device")
    for buffer in model.buffers():
        if buffer.dtype != z0.dtype:
            raise TypeError("model buffers must match rollout dtype")
        if buffer.device != z0.device:
            raise ValueError("model buffers must match rollout device")
