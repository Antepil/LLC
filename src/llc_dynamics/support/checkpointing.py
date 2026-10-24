"""Strict configuration for training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
            rendered = format_str.format(epoch=0, **{monitor_key: 0.0})
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
