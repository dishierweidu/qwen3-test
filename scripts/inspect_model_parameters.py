#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inspect_architecture import inspect_architecture


PARAMETER_FIELDS = (
    "total_parameters",
    "active_parameters_per_token",
    "routed_parameters",
    "shared_parameters",
    "dense_parameters",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect model parameter counts without allocating weights."
    )
    parser.add_argument(
        "model_config",
        type=Path,
        help="Path to a legacy model YAML file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print deterministic machine-readable JSON",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path used to reconcile special token IDs",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow tokenizer downloads; inspection is offline by default",
    )
    parser.add_argument(
        "--allow-remote-code",
        action="store_true",
        help="Allow tokenizer repository code independently of networking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_architecture(
        args.model_config,
        tokenizer_path=args.tokenizer,
        allow_network=args.allow_network,
        allow_remote_code=args.allow_remote_code,
    )
    fields = {
        key: summary.to_dict()[key]
        for key in PARAMETER_FIELDS
    }
    if args.json:
        print(json.dumps(fields, indent=2, sort_keys=True))
        return
    for key, value in fields.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
