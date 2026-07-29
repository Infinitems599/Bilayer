from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .figure8 import Figure8Config, generate_figure8_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculation-only manuscript data interface.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    figure8 = subparsers.add_parser("figure8", help="write the Figure 8 synthetic phase-diagram tables")
    figure8.add_argument("--output-dir", type=Path, required=True)
    figure8.add_argument("--seed", type=int, default=13000)
    figure8.add_argument("--quick", action="store_true", help="use a 3x3 grid and one realization for a smoke test")
    args = parser.parse_args()
    if args.experiment == "figure8":
        config = Figure8Config(seed=args.seed)
        if args.quick:
            config = replace(config, grid_size=3, instances=1)
        print(json.dumps({name: str(path) for name, path in generate_figure8_data(args.output_dir, config).items()}, indent=2))


if __name__ == "__main__":
    main()
