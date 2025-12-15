#!/bin/bash
# scripts/run_with_accelerate.sh
# 使用 Accelerate 启动训练的脚本

set -e

# =============================================================================
# 配置变量
# =============================================================================
CONFIG_PATH="${CONFIG_PATH:-configs/train/stage1_text_only.yaml}"
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-configs/train/accelerator.yaml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-src/tokenizer/Qwen3}"
LOG_DIR="${LOG_DIR:-./runs}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "使用 HuggingFace Accelerate 启动训练"
    echo ""
    echo "Options:"
    echo "  --config PATH           训练配置文件 (default: configs/train/stage1_text_only.yaml)"
    echo "  --accelerator PATH      Accelerator 配置文件 (default: configs/train/accelerator.yaml)"
    echo "  --tokenizer PATH        Tokenizer 路径 (default: src/tokenizer/Qwen3)"
    echo "  --log-dir PATH          日志目录 (default: ./runs)"
    echo "  --num-gpus N            GPU 数量 (default: 8)"
    echo "  --num-machines N        机器数量 (default: 1)"
    echo "  --machine-rank N        当前机器 rank (default: 0)"
    echo "  --main-ip IP            主进程 IP (default: 127.0.0.1)"
    echo "  --main-port PORT        主进程端口 (default: 29500)"
    echo "  --resume PATH           恢复训练的检查点路径"
    echo "  --deepspeed PATH        DeepSpeed JSON 配置文件（可选，会覆盖 accelerator 配置中的 deepspeed）"
    echo "  --fsdp                  使用 FSDP 模式"
    echo "  --ddp                   使用 DDP 模式（默认）"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # 单机 8 卡 DDP 训练"
    echo "  $0 --config configs/train/deepspeed.yaml --num-gpus 8"
    echo ""
    echo "  # 使用 DeepSpeed ZeRO-3"
    echo "  $0 --config configs/train/deepspeed.yaml --deepspeed configs/deepspeed/zero3_30B.json"
    echo ""
    echo "  # 使用 FSDP"
    echo "  $0 --config configs/train/stage1_text_only.yaml --fsdp"
    echo ""
    echo "  # 多机训练 (在每台机器上运行)"
    echo "  # 机器 0:"
    echo "  $0 --num-machines 2 --machine-rank 0 --main-ip 10.0.0.1"
    echo "  # 机器 1:"
    echo "  $0 --num-machines 2 --machine-rank 1 --main-ip 10.0.0.1"
}

# =============================================================================
# 解析命令行参数
# =============================================================================
RESUME_FROM=""
DEEPSPEED_CONFIG=""
DISTRIBUTED_TYPE="MULTI_GPU"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --accelerator)
            ACCELERATOR_CONFIG="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-machines)
            NUM_MACHINES="$2"
            shift 2
            ;;
        --machine-rank)
            MACHINE_RANK="$2"
            shift 2
            ;;
        --main-ip)
            MAIN_PROCESS_IP="$2"
            shift 2
            ;;
        --main-port)
            MAIN_PROCESS_PORT="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED_CONFIG="$2"
            DISTRIBUTED_TYPE="DEEPSPEED"
            shift 2
            ;;
        --fsdp)
            DISTRIBUTED_TYPE="FSDP"
            shift
            ;;
        --ddp)
            DISTRIBUTED_TYPE="MULTI_GPU"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# 构建 accelerate launch 命令
# =============================================================================
ACCELERATE_CMD="accelerate launch"

# 多 GPU 配置
ACCELERATE_CMD+=" --num_processes $((NUM_GPUS * NUM_MACHINES))"
ACCELERATE_CMD+=" --num_machines $NUM_MACHINES"
ACCELERATE_CMD+=" --machine_rank $MACHINE_RANK"

if [ "$NUM_MACHINES" -gt 1 ]; then
    ACCELERATE_CMD+=" --main_process_ip $MAIN_PROCESS_IP"
    ACCELERATE_CMD+=" --main_process_port $MAIN_PROCESS_PORT"
fi

# 混合精度
ACCELERATE_CMD+=" --mixed_precision bf16"

# 分布式类型
if [ "$DISTRIBUTED_TYPE" = "DEEPSPEED" ]; then
    ACCELERATE_CMD+=" --use_deepspeed"
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        ACCELERATE_CMD+=" --deepspeed_config_file $DEEPSPEED_CONFIG"
    fi
elif [ "$DISTRIBUTED_TYPE" = "FSDP" ]; then
    ACCELERATE_CMD+=" --use_fsdp"
    ACCELERATE_CMD+=" --fsdp_sharding_strategy FULL_SHARD"
    ACCELERATE_CMD+=" --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP"
    ACCELERATE_CMD+=" --fsdp_backward_prefetch BACKWARD_PRE"
    ACCELERATE_CMD+=" --fsdp_state_dict_type SHARDED_STATE_DICT"
fi

# Python 模块
ACCELERATE_CMD+=" -m qwen3_omni_pretrain.cli_train_thinker"

# 训练参数
ACCELERATE_CMD+=" --stage stage1"
ACCELERATE_CMD+=" --config $CONFIG_PATH"
ACCELERATE_CMD+=" --tokenizer_name_or_path $TOKENIZER_PATH"
ACCELERATE_CMD+=" --accelerator_config $ACCELERATOR_CONFIG"
ACCELERATE_CMD+=" --tensorboard"
ACCELERATE_CMD+=" --log_dir $LOG_DIR"

if [ -n "$RESUME_FROM" ]; then
    ACCELERATE_CMD+=" --resume_from_checkpoint $RESUME_FROM"
fi

# =============================================================================
# 执行训练
# =============================================================================
echo "=============================================="
echo "Accelerate Training Configuration"
echo "=============================================="
echo "Config:          $CONFIG_PATH"
echo "Accelerator:     $ACCELERATOR_CONFIG"
echo "Tokenizer:       $TOKENIZER_PATH"
echo "Distributed:     $DISTRIBUTED_TYPE"
echo "Num GPUs:        $NUM_GPUS"
echo "Num Machines:    $NUM_MACHINES"
echo "Machine Rank:    $MACHINE_RANK"
if [ -n "$DEEPSPEED_CONFIG" ]; then
    echo "DeepSpeed:       $DEEPSPEED_CONFIG"
fi
if [ -n "$RESUME_FROM" ]; then
    echo "Resume From:     $RESUME_FROM"
fi
echo "=============================================="
echo ""
echo "Running command:"
echo "$ACCELERATE_CMD"
echo ""

exec $ACCELERATE_CMD
