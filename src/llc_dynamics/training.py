"""End-to-end training orchestration for the LLC Neural ODE."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from llc_dynamics.core import (
    RolloutSolverConfig,
    build_neural_ode,
    build_rollout_loss,
    rollout_batch,
)
from llc_dynamics.support import (
    CheckpointConfig,
    CheckpointManager,
    LLCNormalizer,
    MATLABTrajectory,
    NormalizationConfig,
    TrajectoryWindowDataset,
    WindowSpec,
    collate_rollout_windows,
    load_matlab_trajectory,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    project_root: Path
    data_directory: Path
    dataset_manifest: Path
    file_pattern: str
    split_directories: Mapping[str, str]
    epochs: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    seed: int
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    resume_last: bool

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any], *, project_root: Path
    ) -> TrainingConfig:
        if not isinstance(config, Mapping):
            raise TypeError("root config must be a mapping")
        if "training" not in config:
            raise ValueError("root config is missing training")
        raw = config["training"]
        if not isinstance(raw, Mapping):
            raise TypeError("config.training must be a mapping")
        expected = {
            "data_directory",
            "dataset_manifest",
            "file_pattern",
            "split_directories",
            "epochs",
            "batch_size",
            "num_workers",
            "pin_memory",
            "seed",
            "optimizer",
            "gradient_clip_norm",
            "resume_last",
        }
        _require_exact_keys(raw, expected, "training")
        if not isinstance(project_root, Path):
            raise TypeError("project_root must be a pathlib.Path")
        data_directory = _safe_project_path(
            project_root, raw["data_directory"], name="training.data_directory"
        )
        dataset_manifest = _safe_project_path(
            project_root, raw["dataset_manifest"], name="training.dataset_manifest"
        )
        pattern = raw["file_pattern"]
        if not isinstance(pattern, str) or not pattern or Path(pattern).name != pattern:
            raise ValueError(
                "training.file_pattern must be a nonempty basename pattern"
            )
        split_directories = raw["split_directories"]
        if not isinstance(split_directories, Mapping):
            raise TypeError("training.split_directories must be a mapping")
        _require_exact_keys(
            split_directories,
            {"train", "validation", "test"},
            "training.split_directories",
        )
        parsed_splits: dict[str, str] = {}
        for role, value in split_directories.items():
            if (
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError(f"training.split_directories.{role} is unsafe")
            parsed_splits[str(role)] = value
        optimizer = raw["optimizer"]
        if not isinstance(optimizer, Mapping):
            raise TypeError("training.optimizer must be a mapping")
        _require_exact_keys(
            optimizer,
            {"type", "learning_rate", "weight_decay"},
            "training.optimizer",
        )
        if optimizer["type"] != "AdamW":
            raise ValueError("training.optimizer.type must be 'AdamW'")
        clip = raw["gradient_clip_norm"]
        if clip is not None:
            clip = _positive_float(clip, "training.gradient_clip_norm")
        return cls(
            project_root=project_root.resolve(),
            data_directory=data_directory,
            dataset_manifest=dataset_manifest,
            file_pattern=pattern,
            split_directories=parsed_splits,
            epochs=_positive_int(raw["epochs"], "training.epochs"),
            batch_size=_positive_int(raw["batch_size"], "training.batch_size"),
            num_workers=_nonnegative_int(raw["num_workers"], "training.num_workers"),
            pin_memory=_boolean(raw["pin_memory"], "training.pin_memory"),
            seed=_nonnegative_int(raw["seed"], "training.seed"),
            optimizer_type="AdamW",
            learning_rate=_positive_float(
                optimizer["learning_rate"], "training.optimizer.learning_rate"
            ),
            weight_decay=_nonnegative_float(
                optimizer["weight_decay"], "training.optimizer.weight_decay"
            ),
            gradient_clip_norm=clip,
            resume_last=_boolean(raw["resume_last"], "training.resume_last"),
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    start_epoch: int
    completed_epochs: int
    global_step: int
    final_metrics: Mapping[str, float]
    checkpoint_directory: Path
    selected_checkpoint: Path
    selected_epoch: int


@dataclass(frozen=True, slots=True)
class LoadedTrainingDataset:
    trajectories: Mapping[str, tuple[MATLABTrajectory, ...]]
    identity: Mapping[str, Any]


def train_from_config(
    root_config: Mapping[str, Any], *, project_root: Path
) -> TrainingResult:
    """Load split trajectories, train, validate, test, and save checkpoints."""

    training = TrainingConfig.from_mapping(root_config, project_root=project_root)
    _seed_everything(training.seed)
    dataset = load_training_dataset(root_config, training=training)
    trajectories = dataset.trajectories
    normalization = NormalizationConfig.from_mapping(root_config)
    manager = CheckpointManager(
        CheckpointConfig.from_mapping(root_config, project_root=project_root)
    )
    last_path = manager.config.directory / "last.ckpt"
    if training.resume_last and last_path.is_file():
        saved_run_config = manager.read_run_config(last_path)
        saved_normalization = NormalizationConfig.from_mapping(saved_run_config)
        if saved_normalization.state_scales is None:
            raise ValueError("checkpoint normalization state scales are unresolved")
        requested_normalization = normalization.as_mapping()
        if normalization.state_scales is None:
            requested_normalization["state_scales"] = list(
                saved_normalization.state_scales
            )
        if requested_normalization != saved_normalization.as_mapping():
            raise ValueError("checkpoint normalization policy is incompatible")
        normalizer = saved_normalization.build(dtype=_model_dtype(root_config))
        resolved_normalization = saved_normalization
    elif normalization.state_scales is None:
        normalizer = fit_training_normalizer(
            normalization, trajectories["train"], dtype=_model_dtype(root_config)
        )
        resolved_normalization = normalization.resolved(normalizer)
    else:
        normalizer = normalization.build(dtype=_model_dtype(root_config))
        resolved_normalization = normalization

    run_config = copy.deepcopy(dict(root_config))
    run_config["normalization"] = resolved_normalization.as_mapping()
    run_config["dataset_identity"] = copy.deepcopy(dict(dataset.identity))
    model = build_neural_ode(run_config)
    device = model.device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    loss_function = build_rollout_loss(run_config)
    solver = RolloutSolverConfig.from_mapping(run_config)
    window_spec = WindowSpec.from_mapping(run_config)
    loaders = {
        role: _make_loader(
            TrajectoryWindowDataset(items, normalizer=normalizer, spec=window_spec),
            training=training,
            shuffle=(role == "train"),
        )
        for role, items in trajectories.items()
    }
    start_epoch = 0
    global_step = 0
    if training.resume_last and last_path.is_file():
        metadata = manager.load_last(
            model=model,
            optimizer=optimizer,
            expected_run_config=run_config,
            map_location=device,
        )
        start_epoch = metadata.epoch + 1
        global_step = metadata.global_step
    if start_epoch >= training.epochs:
        raise ValueError(
            f"checkpoint already completed epoch {start_epoch}; "
            f"training.epochs is {training.epochs}"
        )

    final_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, training.epochs):
        train_mse, steps = run_epoch(
            model=model,
            loader=loaders["train"],
            solver=solver,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=training.gradient_clip_norm,
        )
        global_step += steps
        val_mse, _ = run_epoch(
            model=model,
            loader=loaders["validation"],
            solver=solver,
            loss_function=loss_function,
            device=device,
        )
        final_metrics = {
            "train_rollout_mse": train_mse,
            "val_rollout_mse": val_mse,
        }
        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            metrics=final_metrics,
            run_config=run_config,
        )
        print(
            f"epoch {epoch + 1}/{training.epochs} "
            f"train_rollout_mse={train_mse:.6e} val_rollout_mse={val_mse:.6e}",
            flush=True,
        )

    selected_checkpoint = manager.best_checkpoint_path()
    selected = manager.load(
        selected_checkpoint,
        model=model,
        optimizer=optimizer,
        expected_run_config=run_config,
        map_location=device,
        restore_rng=False,
    )
    test_mse, _ = run_epoch(
        model=model,
        loader=loaders["test"],
        solver=solver,
        loss_function=loss_function,
        device=device,
    )
    final_metrics["selected_val_rollout_mse"] = selected.metrics[
        "val_rollout_mse"
    ]
    final_metrics["test_rollout_mse"] = test_mse
    print(
        f"test_rollout_mse={test_mse:.6e} selected_epoch={selected.epoch + 1} "
        f"checkpoint={selected_checkpoint}",
        flush=True,
    )
    return TrainingResult(
        start_epoch=start_epoch,
        completed_epochs=training.epochs - start_epoch,
        global_step=global_step,
        final_metrics=final_metrics,
        checkpoint_directory=manager.config.directory,
        selected_checkpoint=selected_checkpoint,
        selected_epoch=selected.epoch,
    )


def load_training_dataset(
    root_config: Mapping[str, Any], *, training: TrainingConfig
) -> LoadedTrainingDataset:
    """Load the exact formal split declared by the generation manifest."""

    trajectories = load_trajectory_splits(root_config, training=training)
    expected, manifest_digest = _read_dataset_manifest(training)
    identity_splits: dict[str, list[dict[str, str]]] = {}
    for role in ("train", "validation", "test"):
        actual = {item.trajectory_id: item for item in trajectories[role]}
        expected_for_role = expected[role]
        missing = sorted(set(expected_for_role).difference(actual))
        unexpected = sorted(set(actual).difference(expected_for_role))
        if missing or unexpected:
            raise ValueError(
                f"{role} split does not match dataset manifest; "
                f"missing={missing}, unexpected={unexpected}"
            )
        entries: list[dict[str, str]] = []
        split_root = (
            training.data_directory / training.split_directories[role]
        ).resolve()
        for trajectory_id in sorted(expected_for_role):
            item = actual[trajectory_id]
            relative_path = item.source_path.resolve().relative_to(
                training.data_directory
            )
            expected_group = expected_for_role[trajectory_id]
            relative_to_split = item.source_path.resolve().relative_to(split_root)
            if not relative_to_split.parts or relative_to_split.parts[0] != expected_group:
                raise ValueError(
                    f"trajectory {trajectory_id!r} must be under "
                    f"{role}/{expected_group}"
                )
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "excitation_group": expected_group,
                    "relative_path": relative_path.as_posix(),
                    "sha256": _sha256_file(item.source_path),
                }
            )
        identity_splits[role] = entries
    return LoadedTrainingDataset(
        trajectories=trajectories,
        identity={
            "schema_version": "llc-training-dataset-identity-v1",
            "manifest_sha256": manifest_digest,
            "splits": identity_splits,
        },
    )


def load_trajectory_splits(
    root_config: Mapping[str, Any], *, training: TrainingConfig
) -> dict[str, tuple[MATLABTrajectory, ...]]:
    """Discover and strictly load non-overlapping trajectory-directory splits."""

    result: dict[str, tuple[MATLABTrajectory, ...]] = {}
    resolved_files: set[Path] = set()
    for role in ("train", "validation", "test"):
        directory = (
            training.data_directory / training.split_directories[role]
        ).resolve()
        if not directory.is_relative_to(training.data_directory):
            raise ValueError(f"resolved {role} split directory escapes data_directory")
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{role} split directory does not exist: {directory}"
            )
        files = tuple(sorted(directory.rglob(training.file_pattern)))
        if not files:
            raise FileNotFoundError(
                f"{role} split contains no {training.file_pattern} files"
            )
        duplicates = resolved_files.intersection(files)
        if duplicates:
            raise ValueError(
                f"trajectory files overlap between splits: {min(duplicates)}"
            )
        resolved_files.update(files)
        result[role] = tuple(
            load_matlab_trajectory(path, config=root_config) for path in files
        )
    identifiers = [item.trajectory_id for split in result.values() for item in split]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("trajectory_id values must be unique across all splits")
    return result


def fit_training_normalizer(
    config: NormalizationConfig,
    trajectories: Sequence[MATLABTrajectory],
    *,
    dtype: torch.dtype = torch.float64,
) -> LLCNormalizer:
    """Fit uncentered RMS scales from training trajectories without concatenating them."""

    if config.state_scales is not None:
        raise ValueError("state scales are already resolved; refusing to refit them")
    if not trajectories:
        raise ValueError("training trajectories must not be empty")
    sum_squares = np.zeros(4, dtype=np.float64)
    sample_count = 0
    for trajectory in trajectories:
        states = np.asarray(trajectory.states, dtype=np.float64)
        sum_squares += np.einsum("ij,ij->j", states, states)
        sample_count += states.shape[0]
    scales = np.sqrt(sum_squares / sample_count)
    # A single row containing the pre-aggregated RMS values gives the same RMS
    # and lets NormalizationConfig enforce its epsilon and state contract.
    return config.fit(torch.as_tensor(scales.reshape(1, 4), dtype=dtype))


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    solver: RolloutSolverConfig,
    loss_function: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, int]:
    """Run one training or evaluation epoch and return element-weighted MSE."""

    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_elements = 0
    steps = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = batch.to(device, non_blocking=loader.pin_memory)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted = rollout_batch(model, batch, solver)
            loss = loss_function(predicted, batch.target_z)
            context_description = _batch_description(batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"nonfinite rollout loss for {context_description}"
                )
            if training:
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        math.inf if gradient_clip_norm is None else gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    raise FloatingPointError(
                        f"nonfinite training gradient for {context_description}"
                    ) from error
                optimizer.step()
                steps += 1
            element_count = batch.target_z[:, 1:].numel()
            total_squared_error += float(loss.detach()) * element_count
            total_elements += element_count
    if total_elements == 0:
        raise ValueError("data loader produced no rollout targets")
    return total_squared_error / total_elements, steps


def _read_dataset_manifest(
    training: TrainingConfig,
) -> tuple[dict[str, dict[str, str]], str]:
    if not training.dataset_manifest.is_file():
        raise FileNotFoundError(
            f"dataset manifest does not exist: {training.dataset_manifest}"
        )
    raw_bytes = training.dataset_manifest.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is not valid UTF-8 JSON") from error
    if not isinstance(document, Mapping):
        raise TypeError("dataset manifest root must be a mapping")
    if document.get("schema_version") != (
        "llc-continuous-random-excitation-data-generation-v1"
    ):
        raise ValueError("dataset manifest has an unsupported schema_version")
    output_directory = document.get("output_directory")
    if not isinstance(output_directory, str) or not output_directory:
        raise ValueError("dataset manifest output_directory is invalid")
    manifest_output = _safe_project_path(
        training.project_root,
        output_directory,
        name="dataset manifest output_directory",
    )
    if manifest_output != training.data_directory:
        raise ValueError(
            "dataset manifest output_directory does not match training.data_directory"
        )
    conditions = document.get("conditions")
    if not isinstance(conditions, list):
        raise TypeError("dataset manifest conditions must be a list")
    expected: dict[str, dict[str, str]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            raise TypeError(f"dataset manifest conditions[{index}] must be a mapping")
        trajectory_id = condition.get("trajectory_id")
        role = condition.get("dataset_role")
        group = condition.get("excitation_group")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"conditions[{index}].trajectory_id is invalid")
        if role not in expected:
            raise ValueError(f"conditions[{index}].dataset_role is invalid")
        if group not in {"vin_only", "reference_only", "both"}:
            raise ValueError(f"conditions[{index}].excitation_group is invalid")
        if any(trajectory_id in values for values in expected.values()):
            raise ValueError(f"duplicate manifest trajectory_id: {trajectory_id!r}")
        expected[role][trajectory_id] = group
    expected_counts = {"train": 80, "validation": 20, "test": 20}
    actual_counts = {role: len(values) for role, values in expected.items()}
    if actual_counts != expected_counts:
        raise ValueError(
            f"formal dataset manifest counts must be {expected_counts}; got {actual_counts}"
        )
    expected_groups = {
        "train": {"vin_only": 20, "reference_only": 20, "both": 40},
        "validation": {"vin_only": 5, "reference_only": 5, "both": 10},
        "test": {"vin_only": 5, "reference_only": 5, "both": 10},
    }
    actual_groups = {
        role: {
            group: sum(value == group for value in values.values())
            for group in ("vin_only", "reference_only", "both")
        }
        for role, values in expected.items()
    }
    if actual_groups != expected_groups:
        raise ValueError(
            "formal dataset manifest excitation groups do not match the "
            f"80/20/20 plan: {actual_groups}"
        )
    return expected, hashlib.sha256(raw_bytes).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_description(batch: Any) -> str:
    windows = ", ".join(
        f"{identifier}[{int(start)}]"
        for identifier, start in zip(
            batch.trajectory_ids, batch.start_indices.tolist(), strict=True
        )
    )
    return f"windows: {windows}"


def _make_loader(
    dataset: TrajectoryWindowDataset, *, training: TrainingConfig, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=shuffle,
        num_workers=training.num_workers,
        pin_memory=training.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_rollout_windows,
        persistent_workers=training.num_workers > 0,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_dtype(config: Mapping[str, Any]) -> torch.dtype:
    if "model" not in config:
        raise ValueError("root config is missing model")
    model = config["model"]
    if not isinstance(model, Mapping):
        raise TypeError("config.model must be a mapping")
    dtype = model.get("dtype")
    if dtype not in {"float32", "float64"}:
        raise ValueError("model.dtype must be 'float32' or 'float64'")
    return {"float32": torch.float32, "float64": torch.float64}[dtype]


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing:
        raise ValueError(f"{name} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {', '.join(unknown)}")


def _safe_project_path(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe relative path")
    result = (root.resolve() / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"{name} must be beneath project_root")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result == 0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value
