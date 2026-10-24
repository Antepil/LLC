#!/usr/bin/env python3
"""Export one trusted training checkpoint for native MATLAB validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from llc_dynamics.support import export_checkpoint_to_matlab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an LLC Neural ODE checkpoint to a MATLAB MAT file."
    )
    parser.add_argument("checkpoint", type=Path, help="Trusted .ckpt file to load")
    parser.add_argument("destination", type=Path, help="Destination .mat file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_checkpoint_to_matlab(args.checkpoint, args.destination)
    print(output)


if __name__ == "__main__":
    main()
