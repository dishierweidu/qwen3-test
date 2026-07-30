#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import torch
from transformers import AutoTokenizer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen3_omni_pretrain.architecture.summary import (
    ArchitectureSummary,
    summarize_model,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.multimodal.tokenization.special_tokens import (
    reconcile_multimodal_token_ids,
)
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    adapt_legacy_config_dict,
)


def load_and_adapt_legacy_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None or raw == {}:
        raise ValueError("model configuration is empty")
    if not isinstance(raw, Mapping):
        raise TypeError("model configuration must be a mapping")
    return adapt_legacy_config_dict(raw)


def inspect_architecture(
    model_config: Path,
    *,
    tokenizer_path: Path | None = None,
    allow_network: bool = False,
    allow_remote_code: bool = False,
) -> ArchitectureSummary:
    config = Qwen3OmniMoeConfig(
        **load_and_adapt_legacy_yaml(model_config)
    )
    tokenizer_vocab_size: int | None = None
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=not allow_network,
            trust_remote_code=allow_remote_code,
        )
        reconcile_multimodal_token_ids(config, tokenizer)
        tokenizer_vocab_size = len(tokenizer)
    config.profile_manifest.validate()
    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerTextModel(config)
    return summarize_model(
        model,
        config.profile_manifest,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a validated model architecture without weights."
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
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return
    for key, value in summary.to_dict().items():
        print(f"{key}: {json.dumps(value, sort_keys=True)}")


if __name__ == "__main__":
    main()
