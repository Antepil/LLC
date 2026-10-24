"""Tests for strict training-checkpoint management."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from llc_dynamics.support.checkpointing import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointMetadata,
)


def checkpoint_mapping(**checkpoint_overrides: object) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "directory": "artifacts/checkpoints",
        "topk": {
            "monitor_key": "test_mean_score",
            "mode": "max",
            "k": 5,
            "format_str": (
                "epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt"
            ),
        },
        "save_last_ckpt": True,
        "save_last_snapshot": False,
    }
    checkpoint.update(checkpoint_overrides)
    return {"checkpoint": checkpoint}


class CheckpointConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)

    def parse(self, mapping: object | None = None) -> CheckpointConfig:
        root = checkpoint_mapping() if mapping is None else mapping
        return CheckpointConfig.from_mapping(root, project_root=self.project_root)

    def test_parses_restored_values_and_resolves_directory_from_project_root(self) -> None:
        config = self.parse()

        self.assertEqual(
            config.directory, self.project_root.resolve() / "artifacts/checkpoints"
        )
        self.assertEqual(config.topk.monitor_key, "test_mean_score")
        self.assertEqual(config.topk.mode, "max")
        self.assertEqual(config.topk.k, 5)
        self.assertEqual(
            config.topk.format_str,
            "epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt",
        )
        self.assertIs(config.save_last_ckpt, True)
        self.assertIs(config.save_last_snapshot, False)

    def test_rejects_non_mapping_and_missing_checkpoint_section(self) -> None:
        for invalid in (None, []):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    CheckpointConfig.from_mapping(  # type: ignore[arg-type]
                        invalid, project_root=self.project_root
                    )
        with self.assertRaisesRegex(ValueError, "missing checkpoint"):
            self.parse({"model": {}})

    def test_rejects_missing_and_unknown_checkpoint_keys(self) -> None:
        missing = checkpoint_mapping()
        del missing["checkpoint"]["directory"]  # type: ignore[index]
        unknown = checkpoint_mapping(extra=True)

        with self.assertRaisesRegex(ValueError, "missing.*directory"):
            self.parse(missing)
        with self.assertRaisesRegex(ValueError, "unknown.*extra"):
            self.parse(unknown)

    def test_rejects_missing_and_unknown_topk_keys(self) -> None:
        missing = checkpoint_mapping()
        del missing["checkpoint"]["topk"]["mode"]  # type: ignore[index]
        unknown = checkpoint_mapping()
        unknown["checkpoint"]["topk"]["extra"] = 1  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "missing.*mode"):
            self.parse(missing)
        with self.assertRaisesRegex(ValueError, "unknown.*extra"):
            self.parse(unknown)

    def test_rejects_unsafe_checkpoint_directories(self) -> None:
        for directory in ("", "/tmp/checkpoints", "../checkpoints", 123):
            with self.subTest(directory=directory):
                with self.assertRaises((TypeError, ValueError)):
                    self.parse(checkpoint_mapping(directory=directory))

    def test_rejects_invalid_topk_count_including_boolean(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                mapping = checkpoint_mapping()
                mapping["checkpoint"]["topk"]["k"] = value  # type: ignore[index]
                with self.assertRaises((TypeError, ValueError)):
                    self.parse(mapping)

    def test_rejects_invalid_mode_and_monitor_key(self) -> None:
        for key, value in (("mode", "largest"), ("monitor_key", ""), ("monitor_key", 7)):
            with self.subTest(key=key, value=value):
                mapping = checkpoint_mapping()
                mapping["checkpoint"]["topk"][key] = value  # type: ignore[index]
                with self.assertRaises((TypeError, ValueError)):
                    self.parse(mapping)

    def test_rejects_unsafe_or_unrenderable_filename_templates(self) -> None:
        templates = (
            "/tmp/model.ckpt",
            "../model.ckpt",
            "nested/model.ckpt",
            "{missing}.ckpt",
            "model.pt",
            "last.ckpt",
        )
        for template in templates:
            with self.subTest(template=template):
                mapping = checkpoint_mapping()
                mapping["checkpoint"]["topk"]["format_str"] = template  # type: ignore[index]
                with self.assertRaises((TypeError, ValueError)):
                    self.parse(mapping)

    def test_rejects_non_boolean_save_flags(self) -> None:
        for key in ("save_last_ckpt", "save_last_snapshot"):
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    self.parse(checkpoint_mapping(**{key: 1}))

    def test_rejects_enabled_workspace_snapshot(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "snapshot"):
            self.parse(checkpoint_mapping(save_last_snapshot=True))


class CheckpointSaveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.25)
        self.run_config = {
            "model": {"type": "linear", "input_dim": 2},
            "normalization": {"scale": 3.0},
            "experiment": "save-tests",
        }

    def config(
        self,
        *,
        mode: str = "max",
        k: int = 5,
        save_last_ckpt: bool = True,
        monitor_key: str = "test_mean_score",
        format_str: str | None = None,
    ) -> CheckpointConfig:
        mapping = checkpoint_mapping(save_last_ckpt=save_last_ckpt)
        topk = mapping["checkpoint"]["topk"]  # type: ignore[index]
        topk["mode"] = mode  # type: ignore[index]
        topk["k"] = k  # type: ignore[index]
        topk["monitor_key"] = monitor_key  # type: ignore[index]
        if format_str is not None:
            topk["format_str"] = format_str  # type: ignore[index]
        return CheckpointConfig.from_mapping(mapping, project_root=self.project_root)

    def save(
        self,
        manager: CheckpointManager,
        *,
        epoch: int,
        score: float,
        global_step: int | None = None,
    ) -> Path | None:
        return manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=epoch * 10 if global_step is None else global_step,
            metrics={"test_mean_score": score},
            run_config=self.run_config,
        )

    def manifest(self, config: CheckpointConfig) -> dict[str, object]:
        return json.loads((config.directory / "index.json").read_text())

    def test_save_creates_directory_and_complete_versioned_payload(self) -> None:
        config = self.config()
        manager = CheckpointManager(config)

        saved_path = self.save(manager, epoch=3, score=1.25)

        self.assertEqual(saved_path, config.directory / "last.ckpt")
        self.assertTrue(config.directory.is_dir())
        payload = torch.load(saved_path, map_location="cpu", weights_only=False)
        self.assertEqual(
            set(payload),
            {
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
            },
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["global_step"], 30)
        self.assertEqual(payload["metrics"], {"test_mean_score": 1.25})
        self.assertEqual(payload["run_config"], self.run_config)
        self.assertIsNone(payload["scheduler_state"])
        self.assertIsInstance(payload["torch_rng_state"], torch.Tensor)
        self.assertIsInstance(payload["cuda_rng_states"], list)

    def test_rejects_missing_or_non_finite_metrics(self) -> None:
        manager = CheckpointManager(self.config())
        for metrics in (
            {},
            {"test_mean_score": math.nan},
            {"test_mean_score": math.inf},
            {"test_mean_score": 1.0, "other": math.nan},
            {"test_mean_score": True},
        ):
            with self.subTest(metrics=metrics):
                with self.assertRaises((TypeError, ValueError)):
                    manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=1,
                        global_step=1,
                        metrics=metrics,
                        run_config=self.run_config,
                    )

    def test_rejects_invalid_epoch_global_step_and_run_config(self) -> None:
        manager = CheckpointManager(self.config())
        for field, value in (
            ("epoch", -1),
            ("epoch", True),
            ("global_step", -1),
            ("global_step", False),
        ):
            arguments = {"epoch": 1, "global_step": 1}
            arguments[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        metrics={"test_mean_score": 1.0},
                        run_config=self.run_config,
                        **arguments,
                    )
        with self.assertRaises(TypeError):
            manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=1,
                global_step=1,
                metrics={"test_mean_score": 1.0},
                run_config=[],  # type: ignore[arg-type]
            )

    def test_max_mode_retains_only_best_k(self) -> None:
        config = self.config(k=2)
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, score=1.0)
        self.save(manager, epoch=2, score=3.0)
        self.save(manager, epoch=3, score=2.0)

        entries = self.manifest(config)["entries"]
        self.assertEqual([entry["metric"] for entry in entries], [3.0, 2.0])
        self.assertEqual(sorted(path.name for path in config.directory.glob("epoch=*")), [
            "epoch=0002-test_mean_score=3.000.ckpt",
            "epoch=0003-test_mean_score=2.000.ckpt",
        ])

    def test_min_mode_retains_only_best_k(self) -> None:
        config = self.config(
            mode="min",
            k=2,
            monitor_key="loss",
            format_str="epoch={epoch:04d}-loss={loss:.3f}.ckpt",
        )
        manager = CheckpointManager(config)
        for epoch, loss in ((1, 5.0), (2, 2.0), (3, 3.0)):
            manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=epoch,
                metrics={"loss": loss},
                run_config=self.run_config,
            )

        entries = self.manifest(config)["entries"]
        self.assertEqual([entry["metric"] for entry in entries], [2.0, 3.0])
        self.assertEqual(
            manager.best_checkpoint_path().name,
            "epoch=0002-loss=2.000.ckpt",
        )
        self.assertEqual(
            manager.read_run_config(manager.best_checkpoint_path()), self.run_config
        )

    def test_equal_metrics_prefer_later_epoch_then_global_step(self) -> None:
        config = self.config(
            k=2,
            format_str=(
                "epoch={epoch:04d}-step={global_step:04d}-"
                "test_mean_score={test_mean_score:.3f}.ckpt"
            ),
        )
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, global_step=10, score=4.0)
        self.save(manager, epoch=2, global_step=20, score=4.0)
        self.save(manager, epoch=2, global_step=30, score=4.0)

        entries = self.manifest(config)["entries"]
        self.assertEqual(
            [(entry["epoch"], entry["global_step"]) for entry in entries],
            [(2, 30), (2, 20)],
        )

    def test_nonqualifying_save_updates_only_last_checkpoint(self) -> None:
        config = self.config(k=1)
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, score=10.0)
        first_manifest = (config.directory / "index.json").read_bytes()

        returned_path = self.save(manager, epoch=2, score=1.0)

        self.assertEqual(returned_path, config.directory / "last.ckpt")
        self.assertEqual((config.directory / "index.json").read_bytes(), first_manifest)
        self.assertFalse(
            (config.directory / "epoch=0002-test_mean_score=1.000.ckpt").exists()
        )
        latest = torch.load(
            config.directory / "last.ckpt", map_location="cpu", weights_only=False
        )
        self.assertEqual(latest["epoch"], 2)

    def test_rejects_filename_collision_with_different_step(self) -> None:
        config = self.config(k=2, format_str="fixed.ckpt")
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, score=1.0)

        with self.assertRaisesRegex(FileExistsError, "collision"):
            self.save(manager, epoch=2, score=2.0)

    def test_restart_loads_existing_manifest_and_continues_retention(self) -> None:
        config = self.config(k=2)
        self.save(CheckpointManager(config), epoch=1, score=1.0)
        self.save(CheckpointManager(config), epoch=2, score=2.0)
        restarted = CheckpointManager(config)

        self.save(restarted, epoch=3, score=3.0)

        self.assertEqual(
            [entry["metric"] for entry in self.manifest(config)["entries"]],
            [3.0, 2.0],
        )

    def test_rejects_malformed_or_stale_manifest(self) -> None:
        config = self.config(k=1)
        config.directory.mkdir(parents=True)
        manifest_path = config.directory / "index.json"
        policy = {
            "monitor_key": "test_mean_score",
            "mode": "max",
            "k": 1,
            "format_str": (
                "epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt"
            ),
        }
        cases = (
            b"not json",
            json.dumps(
                {"schema_version": 2, "policy": policy, "entries": []}
            ).encode(),
            json.dumps(
                {
                    "schema_version": 1,
                    "policy": policy,
                    "entries": [
                        {
                            "filename": "missing.ckpt",
                            "metric": 1.0,
                            "epoch": 1,
                            "global_step": 1,
                        }
                    ],
                }
            ).encode(),
        )
        for content in cases:
            with self.subTest(content=content):
                manifest_path.write_bytes(content)
                with self.assertRaises(ValueError):
                    CheckpointManager(config)

    def test_rejects_invalid_manifest_entries_and_excess_retention(self) -> None:
        config = self.config(k=2)
        config.directory.mkdir(parents=True)
        manifest_path = config.directory / "index.json"
        (config.directory / "one.ckpt").write_bytes(b"one")
        (config.directory / "two.ckpt").write_bytes(b"two")
        (config.directory / "three.ckpt").write_bytes(b"three")
        valid_entry = {
            "filename": "one.ckpt",
            "metric": 1.0,
            "epoch": 1,
            "global_step": 1,
        }
        policy = {
            "monitor_key": "test_mean_score",
            "mode": "max",
            "k": 2,
            "format_str": (
                "epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt"
            ),
        }
        cases = (
            [valid_entry, valid_entry],
            [{**valid_entry, "filename": "../one.ckpt"}],
            [{**valid_entry, "metric": math.nan}],
            [
                valid_entry,
                {**valid_entry, "filename": "two.ckpt"},
                {**valid_entry, "filename": "three.ckpt"},
            ],
        )
        for entries in cases:
            with self.subTest(entries=entries):
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "policy": policy,
                            "entries": entries,
                        }
                    )
                )
                with self.assertRaises(ValueError):
                    CheckpointManager(config)

    def test_does_not_write_last_when_disabled(self) -> None:
        config = self.config(save_last_ckpt=False)
        manager = CheckpointManager(config)

        returned_path = self.save(manager, epoch=1, score=1.0)

        self.assertIsNotNone(returned_path)
        self.assertTrue(returned_path.exists())
        self.assertFalse((config.directory / "last.ckpt").exists())

    def test_failed_atomic_replace_preserves_previous_last_and_manifest(self) -> None:
        config = self.config(k=2)
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, score=1.0)
        previous_last = (config.directory / "last.ckpt").read_bytes()
        previous_manifest = (config.directory / "index.json").read_bytes()

        with patch(
            "llc_dynamics.support.checkpointing.os.replace",
            side_effect=OSError("injected replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected"):
                self.save(manager, epoch=2, score=2.0)

        self.assertEqual((config.directory / "last.ckpt").read_bytes(), previous_last)
        self.assertEqual(
            (config.directory / "index.json").read_bytes(), previous_manifest
        )
        self.assertEqual(list(config.directory.glob("*.tmp")), [])

    def test_last_or_manifest_publish_failure_rolls_back_all_files(self) -> None:
        for failing_name in ("last.ckpt", "index.json"):
            with self.subTest(failing_name=failing_name):
                case_root = self.project_root / failing_name.replace(".", "-")
                case_root.mkdir()
                mapping = checkpoint_mapping(directory="checkpoints")
                config = CheckpointConfig.from_mapping(mapping, project_root=case_root)
                manager = CheckpointManager(config)
                self.save(manager, epoch=1, score=1.0)
                previous = {
                    path.name: path.read_bytes()
                    for path in config.directory.iterdir()
                    if path.is_file()
                }
                real_replace = os.replace
                failed = False

                def fail_once(source: object, destination: object) -> None:
                    nonlocal failed
                    if Path(destination).name == failing_name and not failed:
                        failed = True
                        raise OSError(f"injected {failing_name} failure")
                    real_replace(source, destination)

                with patch(
                    "llc_dynamics.support.checkpointing.os.replace",
                    side_effect=fail_once,
                ):
                    with self.assertRaisesRegex(OSError, "injected"):
                        self.save(manager, epoch=2, score=2.0)

                current = {
                    path.name: path.read_bytes()
                    for path in config.directory.iterdir()
                    if path.is_file()
                }
                self.assertEqual(current, previous)
                CheckpointManager(config)

    def test_prune_cleanup_failure_keeps_committed_manifest_restartable(self) -> None:
        config = self.config(k=1)
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, score=1.0)
        old_path = config.directory / "epoch=0001-test_mean_score=1.000.ckpt"
        real_unlink = Path.unlink

        def fail_old_checkpoint(path: Path, *args: object, **kwargs: object) -> None:
            if path == old_path:
                raise OSError("injected prune failure")
            real_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", autospec=True, side_effect=fail_old_checkpoint):
            self.save(manager, epoch=2, score=2.0)

        restarted = CheckpointManager(config)
        self.assertEqual(
            [entry["metric"] for entry in self.manifest(config)["entries"]], [2.0]
        )
        self.assertIsInstance(restarted, CheckpointManager)
        self.assertTrue(
            (config.directory / "epoch=0002-test_mean_score=2.000.ckpt").is_file()
        )

    def test_invalid_run_config_cannot_replace_existing_recovery_points(self) -> None:
        config = self.config(k=1)
        manager = CheckpointManager(config)
        self.save(manager, epoch=1, score=1.0)
        previous = {
            path.name: path.read_bytes()
            for path in config.directory.iterdir()
            if path.is_file()
        }

        with self.assertRaisesRegex(ValueError, "model"):
            manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=2,
                global_step=20,
                metrics={"test_mean_score": 2.0},
                run_config={},
            )

        self.assertEqual(
            {
                path.name: path.read_bytes()
                for path in config.directory.iterdir()
                if path.is_file()
            },
            previous,
        )

    def test_restart_rejects_changed_ranking_policy(self) -> None:
        config = self.config()
        self.save(CheckpointManager(config), epoch=1, score=100.0)
        changed_mapping = checkpoint_mapping()
        changed_topk = changed_mapping["checkpoint"]["topk"]  # type: ignore[index]
        changed_topk.update(  # type: ignore[union-attr]
            {
                "monitor_key": "loss",
                "mode": "min",
                "format_str": "epoch={epoch:04d}-loss={loss:.3f}.ckpt",
            }
        )
        changed = CheckpointConfig.from_mapping(
            changed_mapping, project_root=self.project_root
        )

        with self.assertRaisesRegex(ValueError, "policy"):
            CheckpointManager(changed)


class CheckpointLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)
        self.config = CheckpointConfig.from_mapping(
            checkpoint_mapping(), project_root=self.project_root
        )
        self.manager = CheckpointManager(self.config)
        self.run_config = {
            "model": {"type": "linear", "input_dim": 2, "output_dim": 1},
            "normalization": {"scale": [1.0, 2.0]},
            "experiment": "load-tests",
        }

    def training_objects(
        self,
    ) -> tuple[nn.Linear, torch.optim.Adam, torch.optim.lr_scheduler.StepLR]:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return model, optimizer, scheduler

    def populate_optimizer(
        self, model: nn.Linear, optimizer: torch.optim.Optimizer
    ) -> None:
        optimizer.zero_grad()
        model(torch.tensor([[2.0, -1.0]])).square().sum().backward()
        optimizer.step()

    def save_checkpoint(
        self, *, scheduler: object | None
    ) -> tuple[Path, dict[str, object]]:
        model, optimizer, real_scheduler = self.training_objects()
        self.populate_optimizer(model, optimizer)
        if scheduler is not None:
            scheduler = real_scheduler
            real_scheduler.step()
        random.seed(101)
        torch.manual_seed(202)
        self.manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=7,
            global_step=123,
            metrics={"test_mean_score": 9.5, "loss": 0.25},
            run_config=self.run_config,
        )
        payload = torch.load(
            self.config.directory / "last.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        return self.config.directory / "last.ckpt", payload

    def assert_nested_equal(self, actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected, strict=True):
                self.assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def write_payload(self, payload: object, filename: str = "modified.ckpt") -> Path:
        path = self.config.directory / filename
        torch.save(payload, path)
        return path

    def test_explicit_load_restores_training_state_scheduler_and_rng(self) -> None:
        path, payload = self.save_checkpoint(scheduler=object())
        expected_python_random = random.random()
        expected_torch_random = torch.rand(4)
        model, optimizer, scheduler = self.training_objects()
        random.seed(999)
        torch.manual_seed(999)

        metadata = self.manager.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_config=self.run_config,
            map_location="cpu",
        )

        self.assert_nested_equal(model.state_dict(), payload["model_state"])
        self.assert_nested_equal(optimizer.state_dict(), payload["optimizer_state"])
        self.assert_nested_equal(scheduler.state_dict(), payload["scheduler_state"])
        self.assertEqual(random.random(), expected_python_random)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch_random))
        self.assertEqual(metadata.epoch, 7)
        self.assertEqual(metadata.global_step, 123)
        self.assertEqual(dict(metadata.metrics), {"test_mean_score": 9.5, "loss": 0.25})
        self.assertEqual(dict(metadata.run_config), self.run_config)
        with self.assertRaises(FrozenInstanceError):
            metadata.epoch = 8  # type: ignore[misc]

    def test_load_last_restores_without_scheduler_and_can_leave_rng_unchanged(self) -> None:
        self.save_checkpoint(scheduler=None)
        model, optimizer, _ = self.training_objects()
        random.seed(303)
        torch.manual_seed(404)
        expected_python_state = random.getstate()
        expected_torch_state = torch.get_rng_state().clone()

        metadata = self.manager.load_last(
            model=model,
            optimizer=optimizer,
            expected_run_config={**self.run_config, "experiment": "new-name"},
            scheduler=None,
            map_location=torch.device("cpu"),
            restore_rng=False,
        )

        self.assertIsInstance(metadata, CheckpointMetadata)
        self.assertEqual(random.getstate(), expected_python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))

    def test_rejects_missing_file_and_malformed_or_unsupported_payload(self) -> None:
        model, optimizer, _ = self.training_objects()
        with self.assertRaises(FileNotFoundError):
            self.manager.load(
                self.config.directory / "absent.ckpt",
                model=model,
                optimizer=optimizer,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

        _, valid = self.save_checkpoint(scheduler=None)
        malformed_payloads = (
            [],
            {**valid, "schema_version": 99},
            {key: value for key, value in valid.items() if key != "model_state"},
            {**valid, "epoch": True},
            {**valid, "metrics": {"test_mean_score": math.nan}},
            {**valid, "torch_rng_state": "invalid"},
        )
        for index, payload in enumerate(malformed_payloads):
            with self.subTest(index=index):
                path = self.write_payload(payload, f"malformed-{index}.ckpt")
                destination_model, destination_optimizer, _ = self.training_objects()
                with self.assertRaises((TypeError, ValueError)):
                    self.manager.load(
                        path,
                        model=destination_model,
                        optimizer=destination_optimizer,
                        expected_run_config=self.run_config,
                        map_location="cpu",
                    )

    def test_rejects_incompatible_model_without_mutating_it(self) -> None:
        path, _ = self.save_checkpoint(scheduler=None)
        model = nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        original_state = deepcopy(model.state_dict())

        with self.assertRaises((ValueError, RuntimeError)):
            self.manager.load(
                path,
                model=model,
                optimizer=optimizer,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

        self.assert_nested_equal(model.state_dict(), original_state)

    def test_rejects_invalid_optimizer_state_without_mutating_model(self) -> None:
        _, payload = self.save_checkpoint(scheduler=None)
        payload["optimizer_state"]["param_groups"][0]["params"] = []
        path = self.write_payload(payload)
        model, optimizer, _ = self.training_objects()
        original_state = deepcopy(model.state_dict())

        with self.assertRaises(ValueError):
            self.manager.load(
                path,
                model=model,
                optimizer=optimizer,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

        self.assert_nested_equal(model.state_dict(), original_state)

    def test_rejects_scheduler_presence_mismatch_in_both_directions(self) -> None:
        saved_with_scheduler, _ = self.save_checkpoint(scheduler=object())
        model, optimizer, _ = self.training_objects()
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.manager.load(
                saved_with_scheduler,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

        saved_without_scheduler, _ = self.save_checkpoint(scheduler=None)
        model, optimizer, scheduler = self.training_objects()
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.manager.load(
                saved_without_scheduler,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

    def test_rejects_model_or_normalization_contract_mismatch(self) -> None:
        path, _ = self.save_checkpoint(scheduler=None)
        for section, replacement in (
            ("model", {"type": "different"}),
            ("normalization", {"scale": [1.0]}),
        ):
            expected = deepcopy(self.run_config)
            expected[section] = replacement
            model, optimizer, _ = self.training_objects()
            with self.subTest(section=section):
                with self.assertRaisesRegex(ValueError, section):
                    self.manager.load(
                        path,
                        model=model,
                        optimizer=optimizer,
                        expected_run_config=expected,
                        map_location="cpu",
                    )

    def test_rejects_different_optimizer_type_before_training_resumes(self) -> None:
        path, _ = self.save_checkpoint(scheduler=None)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        with self.assertRaisesRegex(ValueError, "optimizer type"):
            self.manager.load(
                path,
                model=model,
                optimizer=optimizer,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

    def test_rejects_different_or_malformed_scheduler_state(self) -> None:
        path, payload = self.save_checkpoint(scheduler=object())
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        different_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.5
        )
        with self.assertRaisesRegex(ValueError, "scheduler type"):
            self.manager.load(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=different_scheduler,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

        payload["scheduler_state"] = {}
        malformed_path = self.write_payload(payload, "empty-scheduler.ckpt")
        model, optimizer, scheduler = self.training_objects()
        with self.assertRaisesRegex(ValueError, "scheduler state"):
            self.manager.load(
                malformed_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

    def test_rejects_optimizer_tensor_shape_mismatch(self) -> None:
        _, payload = self.save_checkpoint(scheduler=None)
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["exp_avg"] = torch.zeros(3)
        path = self.write_payload(payload, "bad-optimizer-shape.ckpt")
        model, optimizer, _ = self.training_objects()

        with self.assertRaisesRegex(ValueError, "optimizer state"):
            self.manager.load(
                path,
                model=model,
                optimizer=optimizer,
                expected_run_config=self.run_config,
                map_location="cpu",
            )

    def test_restored_adam_produces_same_next_training_step(self) -> None:
        source_model, source_optimizer, _ = self.training_objects()
        self.populate_optimizer(source_model, source_optimizer)
        self.manager.save(
            model=source_model,
            optimizer=source_optimizer,
            epoch=1,
            global_step=1,
            metrics={"test_mean_score": 1.0},
            run_config=self.run_config,
        )
        restored_model, restored_optimizer, _ = self.training_objects()
        self.manager.load_last(
            model=restored_model,
            optimizer=restored_optimizer,
            expected_run_config=self.run_config,
            map_location="cpu",
            restore_rng=False,
        )

        for model, optimizer in (
            (source_model, source_optimizer),
            (restored_model, restored_optimizer),
        ):
            optimizer.zero_grad()
            model(torch.tensor([[0.5, 1.5]])).square().sum().backward()
            optimizer.step()

        self.assert_nested_equal(
            restored_model.state_dict(), source_model.state_dict()
        )


if __name__ == "__main__":
    unittest.main()
