"""Training windows that preserve exact gate events from LLC trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .adapters import MATLABTrajectory
from .normalization import LLCNormalizer


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Fixed window length and start stride, both measured in intervals."""

    length_intervals: int
    stride_intervals: int

    def __post_init__(self) -> None:
        for name, value in (
            ("length_intervals", self.length_intervals),
            ("stride_intervals", self.stride_intervals),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be strictly positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> WindowSpec:
        """Build from a root configuration or its ``window`` section."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        raw = config.get("window", config)
        if not isinstance(raw, Mapping):
            raise TypeError("config.window must be a mapping")
        expected = {"length_intervals", "stride_intervals"}
        missing = sorted(expected.difference(raw))
        unknown = sorted(set(raw).difference(expected))
        if missing:
            raise ValueError(
                f"window config is missing required keys: {', '.join(missing)}"
            )
        if unknown:
            raise ValueError(f"window config has unknown keys: {', '.join(unknown)}")
        return cls(
            length_intervals=raw["length_intervals"],
            stride_intervals=raw["stride_intervals"],
        )


@dataclass(frozen=True, slots=True)
class RolloutWindow:
    """One normalized rollout window with a variable-length gate-event table."""

    trajectory_id: str
    start_index: int
    start_time_s: float
    tau: Tensor
    z0: Tensor
    target_z: Tensor
    initial_gates: Tensor
    event_tau: Tensor
    gates_after: Tensor
    vin_normalized: Tensor


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """A batch of fixed-length windows with per-window event tensors."""

    trajectory_ids: tuple[str, ...]
    start_indices: Tensor
    start_times_s: Tensor
    tau: Tensor
    z0: Tensor
    target_z: Tensor
    initial_gates: Tensor
    event_tau: tuple[Tensor, ...]
    gates_after: tuple[Tensor, ...]
    vin_normalized: Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> RolloutBatch:
        """Return a batch whose tensors have been moved to ``device``."""

        return RolloutBatch(
            trajectory_ids=self.trajectory_ids,
            start_indices=self.start_indices.to(device, non_blocking=non_blocking),
            start_times_s=self.start_times_s.to(device, non_blocking=non_blocking),
            tau=self.tau.to(device, non_blocking=non_blocking),
            z0=self.z0.to(device, non_blocking=non_blocking),
            target_z=self.target_z.to(device, non_blocking=non_blocking),
            initial_gates=self.initial_gates.to(device, non_blocking=non_blocking),
            event_tau=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.event_tau
            ),
            gates_after=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.gates_after
            ),
            vin_normalized=self.vin_normalized.to(
                device, non_blocking=non_blocking
            ),
        )


class TrajectoryWindowDataset(Dataset[RolloutWindow]):
    """Lazily slice normalized windows from already split full trajectories.

    Dataset splitting and normalization fitting must happen before construction.
    A gate event at the window start determines ``initial_gates``; events at the
    window end are excluded because they do not affect integration to that node.
    """

    def __init__(
        self,
        trajectories: Sequence[MATLABTrajectory],
        *,
        normalizer: LLCNormalizer,
        spec: WindowSpec,
    ) -> None:
        if isinstance(trajectories, (str, bytes)) or not isinstance(
            trajectories, Sequence
        ):
            raise TypeError("trajectories must be a sequence")
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        if not isinstance(normalizer, LLCNormalizer):
            raise TypeError("normalizer must be an LLCNormalizer")
        if normalizer.device.type != "cpu":
            raise ValueError("window dataset normalizer must be on CPU")
        if not isinstance(spec, WindowSpec):
            raise TypeError("spec must be a WindowSpec")

        checked: list[MATLABTrajectory] = []
        identifiers: set[str] = set()
        indices: list[tuple[int, int]] = []
        trailing: list[int] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory, MATLABTrajectory):
                raise TypeError("every trajectory must be a MATLABTrajectory")
            if trajectory.trajectory_id in identifiers:
                raise ValueError(
                    f"duplicate trajectory_id: {trajectory.trajectory_id!r}"
                )
            identifiers.add(trajectory.trajectory_id)
            interval_count = trajectory.times_s.size - 1
            if interval_count < spec.length_intervals:
                raise ValueError(
                    f"trajectory {trajectory.trajectory_id!r} has {interval_count} "
                    f"intervals, fewer than window length {spec.length_intervals}"
                )
            starts = range(
                0,
                interval_count - spec.length_intervals + 1,
                spec.stride_intervals,
            )
            trajectory_starts = list(starts)
            indices.extend((trajectory_index, start) for start in trajectory_starts)
            last_end = trajectory_starts[-1] + spec.length_intervals
            trailing.append(interval_count - last_end)
            checked.append(trajectory)

        self._trajectories = tuple(checked)
        self._normalizer = normalizer
        self.spec = spec
        self._indices = tuple(indices)
        self._trailing_intervals = tuple(trailing)

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def window_counts(self) -> tuple[int, ...]:
        """Return the number of indexed windows for each input trajectory."""

        counts = [0] * len(self._trajectories)
        for trajectory_index, _ in self._indices:
            counts[trajectory_index] += 1
        return tuple(counts)

    @property
    def trailing_intervals(self) -> tuple[int, ...]:
        """Return uncovered tail intervals after the final complete window."""

        return self._trailing_intervals

    def __getitem__(self, index: int) -> RolloutWindow:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("window index must be an integer")
        trajectory_index, start_index = self._indices[index]
        trajectory = self._trajectories[trajectory_index]
        stop_index = start_index + self.spec.length_intervals
        start_time = float(trajectory.times_s[start_index])
        stop_time = float(trajectory.times_s[stop_index])

        times = torch.as_tensor(
            np.array(
                trajectory.times_s[start_index : stop_index + 1] - start_time,
                copy=True,
            ),
            dtype=self._normalizer.dtype,
        )
        states = torch.as_tensor(
            np.array(trajectory.states[start_index : stop_index + 1], copy=True),
            dtype=self._normalizer.dtype,
        )
        vin = torch.as_tensor(
            np.array(
                trajectory.vin_values_v[start_index : stop_index + 1], copy=True
            ),
            dtype=self._normalizer.dtype,
        )

        first_internal = int(
            np.searchsorted(trajectory.event_times_s, start_time, side="right")
        )
        first_at_end = int(
            np.searchsorted(trajectory.event_times_s, stop_time, side="left")
        )
        initial = (
            trajectory.initial_gates
            if first_internal == 0
            else trajectory.gates_after[first_internal - 1]
        )
        event_times = torch.as_tensor(
            np.array(
                trajectory.event_times_s[first_internal:first_at_end] - start_time,
                copy=True,
            ),
            dtype=self._normalizer.dtype,
        )
        gate_events = torch.as_tensor(
            np.array(
                trajectory.gates_after[first_internal:first_at_end], copy=True
            ),
            dtype=self._normalizer.dtype,
        ).reshape(-1, 4)

        target_z = self._normalizer.normalize_states(states)
        return RolloutWindow(
            trajectory_id=trajectory.trajectory_id,
            start_index=start_index,
            start_time_s=start_time,
            tau=self._normalizer.normalize_time(times),
            z0=target_z[0],
            target_z=target_z,
            initial_gates=torch.as_tensor(
                np.array(initial, copy=True), dtype=self._normalizer.dtype
            ),
            event_tau=self._normalizer.normalize_time(event_times),
            gates_after=gate_events,
            vin_normalized=self._normalizer.normalize_vin(vin),
        )


def collate_rollout_windows(windows: Sequence[RolloutWindow]) -> RolloutBatch:
    """Stack fixed-size window fields while retaining ragged gate events."""

    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise TypeError("windows must be a sequence")
    if not windows:
        raise ValueError("windows must not be empty")
    if any(not isinstance(window, RolloutWindow) for window in windows):
        raise TypeError("every item must be a RolloutWindow")
    try:
        return RolloutBatch(
            trajectory_ids=tuple(window.trajectory_id for window in windows),
            start_indices=torch.tensor(
                [window.start_index for window in windows], dtype=torch.int64
            ),
            start_times_s=torch.tensor(
                [window.start_time_s for window in windows],
                dtype=windows[0].tau.dtype,
            ),
            tau=torch.stack([window.tau for window in windows]),
            z0=torch.stack([window.z0 for window in windows]),
            target_z=torch.stack([window.target_z for window in windows]),
            initial_gates=torch.stack(
                [window.initial_gates for window in windows]
            ),
            event_tau=tuple(window.event_tau for window in windows),
            gates_after=tuple(window.gates_after for window in windows),
            vin_normalized=torch.stack(
                [window.vin_normalized for window in windows]
            ),
        )
    except RuntimeError as error:
        raise ValueError(
            "windows must have matching fixed tensor shapes, dtypes, and devices"
        ) from error
