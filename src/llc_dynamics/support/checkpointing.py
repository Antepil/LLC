"""Strict configuration for training checkpoints."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn


_PAYLOAD_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _require_exact_keys(
    config: Mapping[str, Any], expected: set[str], *, section: str
) -> None:
    actual = set(config)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{section} config is missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{section} config has unknown keys: {', '.join(unknown)}")


@dataclass(frozen=True)
class TopKConfig:
    """Validated policy for retaining metric-ranked checkpoints."""

    monitor_key: str
    mode: str
    k: int
    format_str: str


@dataclass(frozen=True)
class CheckpointConfig:
    """Validated checkpoint settings with an absolute output directory."""

    directory: Path
    topk: TopKConfig
    save_last_ckpt: bool
    save_last_snapshot: bool

    @classmethod
    def from_mapping(
        cls,
        root_config: Mapping[str, Any],
        *,
        project_root: Path,
    ) -> CheckpointConfig:
        """Parse the root configuration without depending on a YAML library."""

        if not isinstance(root_config, Mapping):
            raise TypeError("root config must be a mapping")
        if "checkpoint" not in root_config:
            raise ValueError("root config is missing checkpoint")
        raw = root_config["checkpoint"]
        if not isinstance(raw, Mapping):
            raise TypeError("checkpoint config must be a mapping")
        _require_exact_keys(
            raw,
            {"directory", "topk", "save_last_ckpt", "save_last_snapshot"},
            section="checkpoint",
        )

        if not isinstance(project_root, Path):
            raise TypeError("project_root must be a pathlib.Path")
        root = project_root.resolve()
        directory_value = raw["directory"]
        if not isinstance(directory_value, str):
            raise TypeError("checkpoint.directory must be a string")
        if not directory_value.strip():
            raise ValueError("checkpoint.directory must not be empty")
        relative_directory = Path(directory_value)
        if relative_directory.is_absolute() or ".." in relative_directory.parts:
            raise ValueError("checkpoint.directory must be a safe relative path")
        directory = (root / relative_directory).resolve()
        if not directory.is_relative_to(root) or directory == root:
            raise ValueError("checkpoint.directory must be beneath project_root")

        topk = cls._parse_topk(raw["topk"])
        save_last_ckpt = raw["save_last_ckpt"]
        save_last_snapshot = raw["save_last_snapshot"]
        if not isinstance(save_last_ckpt, bool):
            raise TypeError("checkpoint.save_last_ckpt must be a boolean")
        if not isinstance(save_last_snapshot, bool):
            raise TypeError("checkpoint.save_last_snapshot must be a boolean")
        if save_last_snapshot:
            raise NotImplementedError("workspace snapshot saving is not supported")

        return cls(
            directory=directory,
            topk=topk,
            save_last_ckpt=save_last_ckpt,
            save_last_snapshot=save_last_snapshot,
        )

    @staticmethod
    def _parse_topk(value: Any) -> TopKConfig:
        if not isinstance(value, Mapping):
            raise TypeError("checkpoint.topk must be a mapping")
        _require_exact_keys(
            value,
            {"monitor_key", "mode", "k", "format_str"},
            section="checkpoint.topk",
        )

        monitor_key = value["monitor_key"]
        if not isinstance(monitor_key, str):
            raise TypeError("checkpoint.topk.monitor_key must be a string")
        if not monitor_key or not monitor_key.isidentifier():
            raise ValueError(
                "checkpoint.topk.monitor_key must be a nonempty identifier"
            )

        mode = value["mode"]
        if not isinstance(mode, str):
            raise TypeError("checkpoint.topk.mode must be a string")
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint.topk.mode must be 'min' or 'max'")

        k = value["k"]
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("checkpoint.topk.k must be an integer")
        if k <= 0:
            raise ValueError("checkpoint.topk.k must be strictly positive")

        format_str = value["format_str"]
        if not isinstance(format_str, str):
            raise TypeError("checkpoint.topk.format_str must be a string")
        try:
            rendered = format_str.format(
                epoch=0, global_step=0, **{monitor_key: 0.0}
            )
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                "checkpoint.topk.format_str cannot be rendered"
            ) from error
        rendered_path = Path(rendered)
        if (
            not rendered
            or rendered_path.is_absolute()
            or len(rendered_path.parts) != 1
            or rendered_path.suffix != ".ckpt"
            or rendered == "last.ckpt"
        ):
            raise ValueError(
                "checkpoint.topk.format_str must render a safe .ckpt basename "
                "other than last.ckpt"
            )

        return TopKConfig(
            monitor_key=monitor_key,
            mode=mode,
            k=k,
            format_str=format_str,
        )


@dataclass(frozen=True)
class _RankedEntry:
    filename: str
    metric: float
    epoch: int
    global_step: int

    @classmethod
    def from_mapping(cls, value: Any, *, directory: Path) -> _RankedEntry:
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint manifest entries must be mappings")
        try:
            _require_exact_keys(
                value,
                {"filename", "metric", "epoch", "global_step"},
                section="checkpoint manifest entry",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint manifest entry") from error

        filename = value["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or len(Path(filename).parts) != 1
            or Path(filename).suffix != ".ckpt"
            or filename == "last.ckpt"
        ):
            raise ValueError("checkpoint manifest contains an unsafe filename")
        metric = _finite_number(value["metric"], name="manifest metric")
        epoch = _nonnegative_integer(value["epoch"], name="manifest epoch")
        global_step = _nonnegative_integer(
            value["global_step"], name="manifest global_step"
        )
        if not (directory / filename).is_file():
            raise ValueError(f"checkpoint manifest file is missing: {filename}")
        return cls(filename, metric, epoch, global_step)

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "metric": self.metric,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CheckpointMetadata:
    """Training counters and configuration recovered from a checkpoint."""

    epoch: int
    global_step: int
    metrics: Mapping[str, float]
    run_config: Mapping[str, Any]


class CheckpointManager:
    """Save trusted-run PyTorch checkpoints under a deterministic policy."""

    def __init__(self, config: CheckpointConfig) -> None:
        if not isinstance(config, CheckpointConfig):
            raise TypeError("config must be a CheckpointConfig")
        self.config = config
        self.config.directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.config.directory / "index.json"
        self._entries = self._read_manifest()

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, float],
        run_config: Mapping[str, Any],
        scheduler: Any | None = None,
    ) -> Path | None:
        """Save training state and return the retained latest/ranked path, if any."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        epoch = _nonnegative_integer(epoch, name="epoch")
        global_step = _nonnegative_integer(global_step, name="global_step")
        normalized_metrics = self._validate_metrics(metrics)
        if not isinstance(run_config, Mapping):
            raise TypeError("run_config must be a mapping")
        self._validate_run_config_contract(run_config, name="run_config")
        if scheduler is not None and not callable(getattr(scheduler, "state_dict", None)):
            raise TypeError("scheduler must provide state_dict()")

        monitor_value = normalized_metrics[self.config.topk.monitor_key]
        filename = self._render_filename(
            epoch=epoch, global_step=global_step, metrics=normalized_metrics
        )
        candidate = _RankedEntry(
            filename=filename,
            metric=monitor_value,
            epoch=epoch,
            global_step=global_step,
        )
        existing = next(
            (entry for entry in self._entries if entry.filename == filename), None
        )
        if existing is not None and existing != candidate:
            raise FileExistsError(
                f"checkpoint filename collision for different saved step: {filename}"
            )
        ranked_path = self.config.directory / filename
        if existing is None and ranked_path.exists():
            raise FileExistsError(f"checkpoint filename collision: {filename}")

        payload = {
            "schema_version": _PAYLOAD_SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_type": _qualified_type(optimizer),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "scheduler_type": None if scheduler is None else _qualified_type(scheduler),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": normalized_metrics,
            "run_config": copy.deepcopy(dict(run_config)),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
            ),
        }

        retained = existing is not None or self._qualifies(candidate)
        old_entries = list(self._entries)
        pruned: _RankedEntry | None = None
        staged: list[tuple[Path, Path]] = []
        try:
            if retained:
                if existing is None:
                    self._entries.append(candidate)
                self._entries.sort(key=self._ranking_key)
                if len(self._entries) > self.config.topk.k:
                    pruned = self._entries.pop()

            if retained:
                staged.append((self._stage_torch_save(payload, ranked_path), ranked_path))
            last_path = self.config.directory / "last.ckpt"
            if self.config.save_last_ckpt:
                staged.append((self._stage_torch_save(payload, last_path), last_path))
            if retained:
                staged.append(
                    (
                        self._stage_bytes_write(
                            self._manifest_bytes(), self._manifest_path
                        ),
                        self._manifest_path,
                    )
                )
            self._publish_transaction(staged)
        except BaseException:
            self._entries = old_entries
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
            raise

        if pruned is not None and pruned.filename != filename:
            try:
                (self.config.directory / pruned.filename).unlink()
            except OSError:
                # The committed manifest is authoritative. A stale unreferenced file
                # is safer than rolling back a transaction that already committed.
                pass

        if self.config.save_last_ckpt:
            return last_path
        return ranked_path if retained else None

    def load(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_run_config: Mapping[str, Any],
        scheduler: Any | None = None,
        map_location: torch.device | str,
        restore_rng: bool = True,
    ) -> CheckpointMetadata:
        """Strictly restore a trusted checkpoint into supplied training objects."""

        if not isinstance(path, Path):
            raise TypeError("checkpoint path must be a pathlib.Path")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if not isinstance(expected_run_config, Mapping):
            raise TypeError("expected_run_config must be a mapping")
        if not isinstance(map_location, (str, torch.device)):
            raise TypeError("map_location must be a device string or torch.device")
        torch.device(map_location)
        if not isinstance(restore_rng, bool):
            raise TypeError("restore_rng must be a boolean")
        if scheduler is not None and not callable(
            getattr(scheduler, "load_state_dict", None)
        ):
            raise TypeError("scheduler must provide load_state_dict()")

        payload = torch.load(path, map_location=map_location, weights_only=False)
        validated = self._validate_payload(
            payload,
            expected_run_config,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        saved_scheduler_state = validated["scheduler_state"]

        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(validated["model_state"], strict=True)
        optimizer_copy = copy.deepcopy(optimizer)
        optimizer_copy.load_state_dict(validated["optimizer_state"])
        if scheduler is not None:
            scheduler_copy = copy.deepcopy(scheduler)
            scheduler_copy.load_state_dict(saved_scheduler_state)

        previous_model = copy.deepcopy(model.state_dict())
        previous_optimizer = copy.deepcopy(optimizer.state_dict())
        previous_scheduler = (
            None if scheduler is None else copy.deepcopy(scheduler.state_dict())
        )
        previous_python_rng = random.getstate()
        previous_torch_rng = torch.get_rng_state()
        previous_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        )
        try:
            model.load_state_dict(validated["model_state"], strict=True)
            optimizer.load_state_dict(validated["optimizer_state"])
            if scheduler is not None:
                scheduler.load_state_dict(saved_scheduler_state)
            if restore_rng:
                random.setstate(validated["python_rng_state"])
                torch.set_rng_state(validated["torch_rng_state"].cpu())
                cuda_states = validated["cuda_rng_states"]
                if cuda_states:
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "checkpoint has CUDA RNG state but CUDA is unavailable"
                        )
                    if len(cuda_states) != torch.cuda.device_count():
                        raise RuntimeError(
                            "checkpoint CUDA RNG state count does not match devices"
                        )
                    torch.cuda.set_rng_state_all(
                        [state.cpu() for state in cuda_states]
                    )
        except BaseException:
            model.load_state_dict(previous_model, strict=True)
            optimizer.load_state_dict(previous_optimizer)
            if scheduler is not None:
                scheduler.load_state_dict(previous_scheduler)
            random.setstate(previous_python_rng)
            torch.set_rng_state(previous_torch_rng)
            if previous_cuda_rng:
                torch.cuda.set_rng_state_all(previous_cuda_rng)
            raise

        return CheckpointMetadata(
            epoch=validated["epoch"],
            global_step=validated["global_step"],
            metrics=copy.deepcopy(validated["metrics"]),
            run_config=copy.deepcopy(validated["run_config"]),
        )

    def load_last(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_run_config: Mapping[str, Any],
        scheduler: Any | None = None,
        map_location: torch.device | str,
        restore_rng: bool = True,
    ) -> CheckpointMetadata:
        """Restore the configured ``last.ckpt`` checkpoint."""

        return self.load(
            self.config.directory / "last.ckpt",
            model=model,
            optimizer=optimizer,
            expected_run_config=expected_run_config,
            scheduler=scheduler,
            map_location=map_location,
            restore_rng=restore_rng,
        )

    def read_run_config(
        self, path: Path, *, map_location: torch.device | str = "cpu"
    ) -> Mapping[str, Any]:
        """Read the configuration from a trusted checkpoint without restoring state."""

        if not isinstance(path, Path):
            raise TypeError("checkpoint path must be a pathlib.Path")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        if not isinstance(map_location, (str, torch.device)):
            raise TypeError("map_location must be a device string or torch.device")
        torch.device(map_location)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        if payload.get("schema_version") != _PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint payload schema version")
        run_config = payload.get("run_config")
        if not isinstance(run_config, Mapping):
            raise TypeError("checkpoint run_config must be a mapping")
        self._validate_run_config_contract(run_config, name="checkpoint run_config")
        return copy.deepcopy(dict(run_config))

    def best_checkpoint_path(self) -> Path:
        """Return the existing checkpoint ranked best by the configured policy."""

        if not self._entries:
            raise FileNotFoundError("checkpoint manifest contains no ranked checkpoints")
        path = self.config.directory / self._entries[0].filename
        if not path.is_file():
            raise FileNotFoundError(f"best checkpoint does not exist: {path}")
        return path

    def _validate_payload(
        self,
        payload: Any,
        expected_run_config: Mapping[str, Any],
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        expected_keys = {
            "schema_version",
            "model_state",
            "optimizer_state",
            "optimizer_type",
            "scheduler_state",
            "scheduler_type",
            "epoch",
            "global_step",
            "metrics",
            "run_config",
            "python_rng_state",
            "torch_rng_state",
            "cuda_rng_states",
        }
        try:
            _require_exact_keys(payload, expected_keys, section="checkpoint payload")
        except ValueError as error:
            raise ValueError("invalid checkpoint payload keys") from error
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _PAYLOAD_SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint payload schema version")
        if not isinstance(payload["model_state"], Mapping):
            raise TypeError("checkpoint model_state must be a mapping")
        if not isinstance(payload["optimizer_state"], Mapping):
            raise TypeError("checkpoint optimizer_state must be a mapping")
        optimizer_type = payload["optimizer_type"]
        if not isinstance(optimizer_type, str):
            raise TypeError("checkpoint optimizer_type must be a string")
        if optimizer_type != _qualified_type(optimizer):
            raise ValueError(
                "checkpoint optimizer type is incompatible with destination"
            )
        self._validate_optimizer_state(payload["optimizer_state"], optimizer)
        scheduler_state = payload["scheduler_state"]
        if scheduler_state is not None and not isinstance(scheduler_state, Mapping):
            raise TypeError("checkpoint scheduler_state must be a mapping or null")
        scheduler_type = payload["scheduler_type"]
        if scheduler_type is not None and not isinstance(scheduler_type, str):
            raise TypeError("checkpoint scheduler_type must be a string or null")
        if (scheduler_state is None) != (scheduler_type is None):
            raise ValueError("checkpoint scheduler state and type are inconsistent")
        if (scheduler_state is None) != (scheduler is None):
            raise ValueError(
                "checkpoint and destination scheduler presence must match"
            )
        if scheduler is not None:
            if scheduler_type != _qualified_type(scheduler):
                raise ValueError(
                    "checkpoint scheduler type is incompatible with destination"
                )
            expected_scheduler_keys = set(scheduler.state_dict())
            if set(scheduler_state) != expected_scheduler_keys:
                raise ValueError("checkpoint scheduler state keys are incompatible")

        epoch = _nonnegative_integer(payload["epoch"], name="checkpoint epoch")
        global_step = _nonnegative_integer(
            payload["global_step"], name="checkpoint global_step"
        )
        metrics = self._validate_metrics(payload["metrics"])
        saved_run_config = payload["run_config"]
        if not isinstance(saved_run_config, Mapping):
            raise TypeError("checkpoint run_config must be a mapping")
        self._validate_run_config_contract(
            saved_run_config, name="checkpoint run_config"
        )
        self._validate_run_config_contract(
            expected_run_config, name="expected_run_config"
        )
        for section in ("model", "normalization"):
            if saved_run_config[section] != expected_run_config[section]:
                raise ValueError(f"checkpoint {section} configuration is incompatible")
        saved_identity = saved_run_config.get("dataset_identity")
        expected_identity = expected_run_config.get("dataset_identity")
        if saved_identity != expected_identity:
            raise ValueError("checkpoint dataset identity is incompatible")

        python_rng_state = payload["python_rng_state"]
        try:
            random.Random().setstate(python_rng_state)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint Python RNG state is invalid") from error
        torch_rng_state = payload["torch_rng_state"]
        if not isinstance(torch_rng_state, torch.Tensor):
            raise TypeError("checkpoint PyTorch RNG state must be a tensor")
        try:
            torch.Generator(device="cpu").set_state(torch_rng_state.cpu())
        except RuntimeError as error:
            raise ValueError("checkpoint PyTorch RNG state is invalid") from error
        cuda_rng_states = payload["cuda_rng_states"]
        if not isinstance(cuda_rng_states, list) or any(
            not isinstance(state, torch.Tensor) for state in cuda_rng_states
        ):
            raise TypeError("checkpoint CUDA RNG states must be a list of tensors")

        return {
            "model_state": payload["model_state"],
            "optimizer_state": payload["optimizer_state"],
            "optimizer_type": optimizer_type,
            "scheduler_state": scheduler_state,
            "scheduler_type": scheduler_type,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "run_config": dict(saved_run_config),
            "python_rng_state": python_rng_state,
            "torch_rng_state": torch_rng_state,
            "cuda_rng_states": cuda_rng_states,
        }

    @staticmethod
    def _validate_run_config_contract(
        run_config: Mapping[str, Any], *, name: str
    ) -> None:
        for section in ("model", "normalization"):
            if section not in run_config:
                raise ValueError(f"{name} is missing {section}")
            if not isinstance(run_config[section], Mapping):
                raise TypeError(f"{name}.{section} must be a mapping")
        identity = run_config.get("dataset_identity")
        if identity is not None and not isinstance(identity, Mapping):
            raise TypeError(f"{name}.dataset_identity must be a mapping")

    @staticmethod
    def _validate_optimizer_state(
        state: Mapping[str, Any], optimizer: torch.optim.Optimizer
    ) -> None:
        raw_state = state.get("state")
        raw_groups = state.get("param_groups")
        if not isinstance(raw_state, Mapping) or not isinstance(raw_groups, list):
            raise ValueError("checkpoint optimizer state structure is invalid")
        if len(raw_groups) != len(optimizer.param_groups):
            raise ValueError("checkpoint optimizer state group count is incompatible")
        for saved_group, destination_group in zip(
            raw_groups, optimizer.param_groups, strict=True
        ):
            if not isinstance(saved_group, Mapping):
                raise ValueError("checkpoint optimizer parameter group is invalid")
            saved_parameter_ids = saved_group.get("params")
            destination_parameters = destination_group.get("params")
            if not isinstance(saved_parameter_ids, list) or not isinstance(
                destination_parameters, list
            ):
                raise ValueError("checkpoint optimizer parameter list is invalid")
            if len(saved_parameter_ids) != len(destination_parameters):
                raise ValueError(
                    "checkpoint optimizer state parameter count is incompatible"
                )
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                for parameter_id, parameter in zip(
                    saved_parameter_ids, destination_parameters, strict=True
                ):
                    parameter_state = raw_state.get(parameter_id, {})
                    if not isinstance(parameter_state, Mapping):
                        raise ValueError("checkpoint optimizer state entry is invalid")
                    if not parameter_state:
                        continue
                    required = {"step", "exp_avg", "exp_avg_sq"}
                    if not required.issubset(parameter_state):
                        raise ValueError("checkpoint optimizer state is incomplete")
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key not in parameter_state:
                            continue
                        tensor = parameter_state[key]
                        if (
                            not isinstance(tensor, torch.Tensor)
                            or tensor.shape != parameter.shape
                        ):
                            raise ValueError(
                                "checkpoint optimizer state tensor shape is incompatible"
                            )

    def _validate_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        monitor_key = self.config.topk.monitor_key
        if monitor_key not in metrics:
            raise ValueError(f"metrics is missing monitored key: {monitor_key}")
        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise TypeError("metric keys must be nonempty strings")
            normalized[key] = _finite_number(value, name=f"metric {key!r}")
        return normalized

    def _render_filename(
        self, *, epoch: int, global_step: int, metrics: Mapping[str, float]
    ) -> str:
        try:
            rendered = self.config.topk.format_str.format(
                epoch=epoch, global_step=global_step, **metrics
            )
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError("checkpoint filename template cannot be rendered") from error
        path = Path(rendered)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.suffix != ".ckpt"
            or rendered == "last.ckpt"
        ):
            raise ValueError("checkpoint filename template rendered an unsafe name")
        return rendered

    def _ranking_key(self, entry: _RankedEntry) -> tuple[float, int, int]:
        metric = -entry.metric if self.config.topk.mode == "max" else entry.metric
        return metric, -entry.epoch, -entry.global_step

    def _qualifies(self, candidate: _RankedEntry) -> bool:
        if len(self._entries) < self.config.topk.k:
            return True
        return self._ranking_key(candidate) < self._ranking_key(self._entries[-1])

    def _read_manifest(self) -> list[_RankedEntry]:
        if not self._manifest_path.exists():
            return []
        try:
            document = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint manifest is not valid JSON") from error
        if not isinstance(document, Mapping):
            raise ValueError("checkpoint manifest must be a mapping")
        try:
            _require_exact_keys(
                document,
                {"schema_version", "policy", "entries"},
                section="checkpoint manifest",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint manifest") from error
        manifest_version = document["schema_version"]
        if (
            isinstance(manifest_version, bool)
            or not isinstance(manifest_version, int)
            or manifest_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint manifest schema version")
        expected_policy = self._manifest_policy()
        if document["policy"] != expected_policy:
            raise ValueError("checkpoint manifest ranking policy is incompatible")
        raw_entries = document["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("checkpoint manifest entries must be a list")
        if len(raw_entries) > self.config.topk.k:
            raise ValueError("checkpoint manifest contains more than top-k entries")
        entries = [
            _RankedEntry.from_mapping(value, directory=self.config.directory)
            for value in raw_entries
        ]
        filenames = [entry.filename for entry in entries]
        if len(filenames) != len(set(filenames)):
            raise ValueError("checkpoint manifest contains duplicate filenames")
        entries.sort(key=self._ranking_key)
        return entries

    def _manifest_policy(self) -> dict[str, Any]:
        return {
            "monitor_key": self.config.topk.monitor_key,
            "mode": self.config.topk.mode,
            "k": self.config.topk.k,
            "format_str": self.config.topk.format_str,
        }

    def _manifest_bytes(self) -> bytes:
        document = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "policy": self._manifest_policy(),
            "entries": [entry.as_dict() for entry in self._entries],
        }
        return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()

    def _stage_torch_save(
        self, payload: Mapping[str, Any], destination: Path
    ) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.config.directory,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(payload, temporary_path)
            with temporary_path.open("rb") as file_handle:
                os.fsync(file_handle.fileno())
            return temporary_path
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _stage_bytes_write(self, data: bytes, destination: Path) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.config.directory,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file_handle:
                file_handle.write(data)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            return temporary_path
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _publish_transaction(self, staged: list[tuple[Path, Path]]) -> None:
        backups: list[tuple[Path, Path]] = []
        published: list[Path] = []
        committed = False
        try:
            for _, destination in staged:
                if destination.exists():
                    descriptor, backup_name = tempfile.mkstemp(
                        dir=self.config.directory,
                        prefix=f".{destination.name}.",
                        suffix=".bak",
                    )
                    os.close(descriptor)
                    backup = Path(backup_name)
                    try:
                        os.replace(destination, backup)
                    except BaseException:
                        backup.unlink(missing_ok=True)
                        raise
                    backups.append((destination, backup))

            for temporary_path, destination in staged:
                os.replace(temporary_path, destination)
                published.append(destination)
            committed = True
        except BaseException:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            for destination, backup in reversed(backups):
                os.replace(backup, destination)
            raise
        finally:
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
            if committed:
                for _, backup in backups:
                    try:
                        backup.unlink()
                    except OSError:
                        pass
