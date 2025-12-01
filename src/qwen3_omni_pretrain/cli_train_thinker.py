# src/qwen3_omni_pretrain/cli_train_thinker.py

import argparse

from qwen3_omni_pretrain.training.trainer_thinker import train_thinker_stage1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-Omni Thinker (Stage1 Text Only)")
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
    return parser.parse_args()


def main():
    args = parse_args()
    train_thinker_stage1(
        config_path=args.config,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
    )


if __name__ == "__main__":
    main()
