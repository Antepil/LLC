#!/usr/bin/env python3
"""Train the LLC Neural ODE from a YAML run configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from llc_dynamics.training import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="YAML run configuration (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root used to resolve data and artifact paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("YAML configuration root must be a mapping")
    result = train_from_config(
        config, project_root=args.project_root.expanduser().resolve()
    )
    print(
        f"completed_epochs={result.completed_epochs} global_step={result.global_step} "
        f"checkpoints={result.checkpoint_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
