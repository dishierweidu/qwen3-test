# src/qwen3_omni_pretrain/cli_train_thinker.py

import argparse
import os
import sys

from qwen3_omni_pretrain.training.trainer_thinker import (
    train_thinker_stage1,
    train_thinker_stage2,
)
from qwen3_omni_pretrain.utils.config_utils import load_yaml
from qwen3_omni_pretrain.training.distributed import distributed_context, ddp_cleanup
from qwen3_omni_pretrain.training.stage2_config import (
    Stage2RuntimeConfig,
    normalize_stage2_config,
)


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
        default="src/tokenizer/Qwen3",
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
    # DeepSpeed / torch.distributed launchers会自动注入 local_rank；接受但不使用
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="Local rank passed by launcher (unused here).",
    )
    # DeepSpeed 配置文件路径占位（传入后交给 trainer 初始化）
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON.",
    )
    # Accelerator 配置文件路径
    parser.add_argument(
        "--accelerator_config",
        type=str,
        default=None,
        help="Path to Accelerator config YAML (enables Accelerator mode).",
    )
    parser.add_argument(
        "--use_accelerator",
        action="store_true",
        help="Enable HuggingFace Accelerate for training.",
    )
    # ==========================================================================
    # Tensor Parallelism 参数 (新增)
    # ==========================================================================
    parser.add_argument(
        "--use_tensor_parallel",
        action="store_true",
        help="Enable Tensor Parallelism for training.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor Parallel size. Must divide world_size evenly.",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="Pipeline Parallel size (not fully supported yet).",
    )
    return parser.parse_args()


def _stage2_cli_runtime(args) -> Stage2RuntimeConfig:
    launched_distributed = (
        "LOCAL_RANK" in os.environ
        or "RANK" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    unsupported = (
        launched_distributed
        or args.use_accelerator
        or args.accelerator_config is not None
        or args.deepspeed is not None
        or args.use_tensor_parallel
        or args.tp_size > 1
        or args.pp_size > 1
        or args.local_rank is not None
    )
    if unsupported:
        raise NotImplementedError(
            "Stage2 distributed execution is not implemented in this P0"
        )
    overrides = {}
    if args.resume_from_checkpoint is not None:
        overrides["resume_from_checkpoint"] = (
            args.resume_from_checkpoint
        )
    return normalize_stage2_config(
        load_yaml(args.config),
        overrides=overrides,
    )


def main():
    args = parse_args()

    if args.stage == "stage2":
        runtime = _stage2_cli_runtime(args)
        train_thinker_stage2(
            cfg=runtime,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            enable_tensorboard=args.tensorboard,
            log_dir=args.log_dir,
            resume_from_checkpoint=None,
        )
        return

    # 如果使用 Accelerator，不需要手动管理 distributed context
    # Accelerator 会自动处理
    use_accelerator = args.use_accelerator or args.accelerator_config is not None
    
    # Tensor Parallel 配置
    use_tensor_parallel = args.use_tensor_parallel
    tp_size = args.tp_size
    pp_size = args.pp_size

    if use_accelerator:
        # Accelerator 模式 - 不需要 distributed_context
        train_thinker_stage1(
            args.config,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            enable_tensorboard=args.tensorboard,
            log_dir=args.log_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            deepspeed_config=args.deepspeed,
            accelerator_config_path=args.accelerator_config,
            use_tensor_parallel=use_tensor_parallel,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        )
    else:
        # 传统模式 - 使用 distributed_context 管理 DDP/DeepSpeed
        # 如果启用了 Tensor Parallel，需要使用特殊的 context
        if use_tensor_parallel and tp_size > 1:
            from qwen3_omni_pretrain.training.distributed import distributed_tensor_parallel_context
            context_manager = distributed_tensor_parallel_context(
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
            )
        else:
            context_manager = distributed_context()
        
        with context_manager:
            train_thinker_stage1(
                args.config,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                enable_tensorboard=args.tensorboard,
                log_dir=args.log_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
                deepspeed_config=args.deepspeed,
                use_tensor_parallel=use_tensor_parallel,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, cleaning up DDP...")
        ddp_cleanup()
        sys.exit(0)
