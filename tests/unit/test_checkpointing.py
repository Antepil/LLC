"""Tests for strict training-checkpoint management."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llc_dynamics.support.checkpointing import CheckpointConfig


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


if __name__ == "__main__":
    unittest.main()
