"""Tests for strict training-checkpoint management."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from llc_dynamics.support.checkpointing import CheckpointConfig, CheckpointManager


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
                "scheduler_state",
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
        cases = (
            b"not json",
            json.dumps({"schema_version": 2, "entries": []}).encode(),
            json.dumps(
                {
                    "schema_version": 1,
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
                    json.dumps({"schema_version": 1, "entries": entries})
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


if __name__ == "__main__":
    unittest.main()
