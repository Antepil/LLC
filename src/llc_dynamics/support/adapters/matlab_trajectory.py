"""Read validated MATLAB trajectories for Python training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


_STATE_NAMES = ("i_Lr", "v_Cr", "i_Lm", "v_o")
_GATE_NAMES = ("Q1_A_high", "Q2_A_low", "Q3_B_high", "Q4_B_low")
_FIXED_SCHEMA = "llc-closed-loop-fixed-grid-trajectory-v3"
_SINUSOIDAL_SCHEMA = "llc-sinusoidal-excitation-fixed-grid-trajectory-v1"
_SUPPORTED_SCHEMAS = {_FIXED_SCHEMA, _SINUSOIDAL_SCHEMA}


@dataclass(frozen=True, slots=True)
class MATLABTrajectory:
    """A complete physical-unit trajectory with an exact gate-event table."""

    source_path: Path
    schema_version: str
    trajectory_id: str
    times_s: NDArray[np.float64]
    states: NDArray[np.float64]
    initial_gates: NDArray[np.uint8]
    event_times_s: NDArray[np.float64]
    gates_after: NDArray[np.uint8]
    vin_times_s: NDArray[np.float64]
    vin_values_v: NDArray[np.float64]
    load_resistance_ohm: float


def load_matlab_trajectory(
    path: str | Path,
    *,
    expected_load_ohm: float | None = 2.304,
    load_abs_tolerance: float = 1.0e-9,
) -> MATLABTrajectory:
    """Load one supported ``.mat`` trajectory and validate its training contract.

    ``expected_load_ohm`` defaults to the fixed-load neural-ODE contract. Pass
    ``None`` only when a later model explicitly includes load as an input.
    """

    source_path = _validate_path(path)
    expected_load = _optional_positive_float(
        expected_load_ohm, name="expected_load_ohm"
    )
    tolerance = _nonnegative_float(load_abs_tolerance, name="load_abs_tolerance")
    raw = _load_mat_file(source_path)

    schema = _scalar_text(_required(raw, "schema_version"), name="schema_version")
    if schema not in _SUPPORTED_SCHEMAS:
        supported = ", ".join(sorted(_SUPPORTED_SCHEMAS))
        raise ValueError(
            f"unsupported trajectory schema {schema!r}; supported values: {supported}"
        )
    trajectory_id = _scalar_text(
        _required(raw, "trajectory_id"), name="trajectory_id"
    )
    _require_names(raw, "state_names", _STATE_NAMES)
    _require_names(raw, "gate_names", _GATE_NAMES)

    times = _float_vector(_required(raw, "state_times_s"), name="state_times_s")
    if times.size < 2:
        raise ValueError("state_times_s must contain at least two nodes")
    if times[0] < 0.0 or not np.all(np.diff(times) > 0.0):
        raise ValueError("state_times_s must be nonnegative and strictly increasing")
    states = _float_matrix(
        _required(raw, "state_values"), columns=len(_STATE_NAMES), name="state_values"
    )
    if states.shape[0] != times.size:
        raise ValueError("state_values rows must match state_times_s")

    initial_gates = _binary_row(
        _required(raw, "initial_gate_values"), name="initial_gate_values"
    )
    event_times = _float_vector(
        _required(raw, "gate_event_times_s"),
        name="gate_event_times_s",
        allow_empty=True,
    )
    gates_after = _binary_matrix(
        _required(raw, "gate_values_after"), name="gate_values_after"
    )
    _validate_gate_events(
        initial_gates, event_times, gates_after, start=times[0], stop=times[-1]
    )

    load_ohm = _load_resistance(raw, schema)
    if expected_load is not None and not np.isclose(
        load_ohm, expected_load, rtol=0.0, atol=tolerance
    ):
        raise ValueError(
            f"trajectory load {load_ohm:g} ohm does not match the fixed-load "
            f"contract {expected_load:g} ohm"
        )
    vin = _vin_at_state_nodes(raw, schema, times)

    return MATLABTrajectory(
        source_path=source_path,
        schema_version=schema,
        trajectory_id=trajectory_id,
        times_s=_readonly(times),
        states=_readonly(states),
        initial_gates=_readonly(initial_gates),
        event_times_s=_readonly(event_times),
        gates_after=_readonly(gates_after),
        vin_times_s=_readonly(times.copy()),
        vin_values_v=_readonly(vin),
        load_resistance_ohm=load_ohm,
    )


def _load_mat_file(path: Path) -> Mapping[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - exercised by installation
        raise RuntimeError(
            "SciPy is required to read MATLAB trajectory files; install project dependencies"
        ) from error
    try:
        result = loadmat(path, simplify_cells=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to read MATLAB trajectory {path}: {error}") from error
    if not isinstance(result, Mapping):
        raise ValueError("MATLAB trajectory reader returned an invalid root value")
    return result


def _validate_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    result = Path(path).expanduser().resolve()
    if result.suffix.lower() != ".mat":
        raise ValueError("trajectory path must have a .mat suffix")
    if not result.is_file():
        raise FileNotFoundError(f"trajectory file does not exist: {result}")
    return result


def _required(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"trajectory is missing required field: {key}")
    return raw[key]


def _scalar_text(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar text")
    result = str(array.reshape(-1)[0]).strip()
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _text_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    else:
        array = np.asarray(value, dtype=object)
        values = tuple(array.reshape(-1))
    result = tuple(_scalar_text(item, name=name) for item in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _require_names(raw: Mapping[str, Any], key: str, expected: tuple[str, ...]) -> None:
    actual = _text_sequence(_required(raw, key), name=key)
    if actual != expected:
        raise ValueError(f"{key} must equal {list(expected)!r}; got {list(actual)!r}")


def _float_vector(
    value: Any, *, name: str, allow_empty: bool = False
) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _float_matrix(
    value: Any, *, columns: int, name: str
) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if result.ndim == 1 and result.size == columns:
        result = result.reshape(1, columns)
    if result.ndim != 2 or result.shape[1] != columns or result.shape[0] == 0:
        raise ValueError(f"{name} must have shape [N, {columns}] with N > 0")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _binary_row(value: Any, *, name: str) -> NDArray[np.uint8]:
    result = _binary_array(value, name=name)
    if result.size != len(_GATE_NAMES):
        raise ValueError(f"{name} must contain four gate values")
    return result.reshape(len(_GATE_NAMES))


def _binary_matrix(value: Any, *, name: str) -> NDArray[np.uint8]:
    result = _binary_array(value, name=name)
    if result.size == 0:
        return np.empty((0, len(_GATE_NAMES)), dtype=np.uint8)
    if result.ndim == 1 and result.size == len(_GATE_NAMES):
        result = result.reshape(1, len(_GATE_NAMES))
    if result.ndim != 2 or result.shape[1] != len(_GATE_NAMES):
        raise ValueError(f"{name} must have shape [E, 4]")
    return np.ascontiguousarray(result)


def _binary_array(value: Any, *, name: str) -> NDArray[np.uint8]:
    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be numeric or boolean")
    if not np.all(np.isfinite(array.astype(np.float64, copy=False))):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _validate_gate_events(
    initial: NDArray[np.uint8],
    event_times: NDArray[np.float64],
    gates_after: NDArray[np.uint8],
    *,
    start: float,
    stop: float,
) -> None:
    if gates_after.shape[0] != event_times.size:
        raise ValueError("gate_values_after rows must match gate_event_times_s")
    if event_times.size:
        if not np.all(np.diff(event_times) > 0.0):
            raise ValueError("gate_event_times_s must be strictly increasing")
        if event_times[0] < start or event_times[-1] > stop:
            raise ValueError("gate events must lie within the state time range")
        previous = np.vstack((initial, gates_after[:-1]))
        if np.any(np.all(gates_after == previous, axis=1)):
            raise ValueError("every gate event must change at least one gate value")
    all_gates = np.vstack((initial, gates_after))
    if np.any((all_gates[:, 0] + all_gates[:, 1]) > 1) or np.any(
        (all_gates[:, 2] + all_gates[:, 3]) > 1
    ):
        raise ValueError("gate event table contains bridge-leg overlap")


def _load_resistance(raw: Mapping[str, Any], schema: str) -> float:
    if schema == _FIXED_SCHEMA:
        condition = _required(raw, "condition")
        if not isinstance(condition, Mapping):
            raise ValueError("condition must be a MATLAB scalar struct")
        value = _required(condition, "load_ohm")
    else:
        value = _required(raw, "fixed_load_ohm")
    return _positive_float(value, name="load resistance")


def _vin_at_state_nodes(
    raw: Mapping[str, Any], schema: str, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    if schema == _FIXED_SCHEMA:
        condition = _required(raw, "condition")
        if not isinstance(condition, Mapping):
            raise ValueError("condition must be a MATLAB scalar struct")
        vin = _positive_float(_required(condition, "vin_v"), name="condition.vin_v")
        return np.full(times.shape, vin, dtype=np.float64)

    names = _text_sequence(
        _required(raw, "state_excitation_names"), name="state_excitation_names"
    )
    if "vin_v" not in names:
        raise ValueError("state_excitation_names is missing vin_v")
    values = _float_matrix(
        _required(raw, "state_excitation_values"),
        columns=len(names),
        name="state_excitation_values",
    )
    if values.shape[0] != times.size:
        raise ValueError("state_excitation_values rows must match state_times_s")
    vin = values[:, names.index("vin_v")]
    if np.any(vin <= 0.0):
        raise ValueError("vin_v samples must be strictly positive")
    return np.ascontiguousarray(vin)


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array
