# src/qwen3_omni_pretrain/cli_train_thinker.py

import argparse

from qwen3_omni_pretrain.training.trainer_thinker import (
    train_thinker_stage1,
    train_thinker_stage2,
)
from qwen3_omni_pretrain.utils.config_utils import load_yaml
from qwen3_omni_pretrain.training.distributed import distributed_context


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-Omni Thinker")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to stage1_text_only.yaml",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="HF tokenizer name or path",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "stage2"],
        default="stage1",
        help="Which training stage to run."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 无论单卡还是多卡，统一在这里 init / destroy 进程组
    with distributed_context():
        if args.stage == "stage1":
            train_thinker_stage1(
                args.config,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
            )
        else:
            cfg = load_yaml(args.config)
            train_thinker_stage2(
                cfg=cfg,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
            )


if __name__ == "__main__":
    main()
