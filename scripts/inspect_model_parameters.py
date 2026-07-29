#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.utils.model_stats import (
    collect_parameter_stats,
    format_parameter_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect total and estimated active parameters without allocating weights."
        )
    )
    parser.add_argument(
        "model_config", type=Path, help="Path to a model YAML file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.model_config.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)
    config = Qwen3OmniMoeConfig(**config_data)
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerTextModel(config)
    stats = collect_parameter_stats(model)
    if args.json:
        print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_parameter_stats(stats))


if __name__ == "__main__":
    main()
