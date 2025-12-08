# src/qwen3_omni_pretrain/cli_train_thinker.py

import argparse
import sys

from qwen3_omni_pretrain.training.trainer_thinker import (
    train_thinker_stage1,
    train_thinker_stage2,
)
from qwen3_omni_pretrain.utils.config_utils import load_yaml
from qwen3_omni_pretrain.training.distributed import distributed_context, ddp_cleanup


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
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (rank 0 only).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training.",
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
                enable_tensorboard=args.tensorboard,
                log_dir=args.log_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        else:
            cfg = load_yaml(args.config)
            train_thinker_stage2(
                cfg=cfg,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                enable_tensorboard=args.tensorboard,
                log_dir=args.log_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, cleaning up DDP...")
        ddp_cleanup()
        sys.exit(0)
