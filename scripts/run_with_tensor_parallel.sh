#!/bin/bash
# scripts/run_with_tensor_parallel.sh
# ============================================================================
# 3D Parallel Training Script (Tensor Parallelism + ZeRO)
# ============================================================================
#
# This script launches training with Tensor Parallelism (TP) combined with
# DeepSpeed ZeRO for memory-efficient large model training.
#
# Usage:
#   ./scripts/run_with_tensor_parallel.sh [TP_SIZE] [NUM_GPUS]
#
# Examples:
#   # 8 GPUs with TP=2 (4 data parallel groups)
#   ./scripts/run_with_tensor_parallel.sh 2 8
#
#   # 4 GPUs with TP=2 (2 data parallel groups)
#   ./scripts/run_with_tensor_parallel.sh 2 4
#
#   # 8 GPUs with TP=4 (2 data parallel groups)
#   ./scripts/run_with_tensor_parallel.sh 4 8
# ============================================================================

set -e

# Default values
TP_SIZE=${1:-2}
NUM_GPUS=${2:-$(nvidia-smi -L | wc -l)}
CONFIG_PATH=${3:-"configs/train/tensor_parallel.yaml"}
ACCELERATE_CONFIG=${4:-"configs/accelerate/deepspeed_zero3.yaml"}

# Validate TP_SIZE
if [ $((NUM_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "Error: NUM_GPUS ($NUM_GPUS) must be divisible by TP_SIZE ($TP_SIZE)"
    exit 1
fi

DP_SIZE=$((NUM_GPUS / TP_SIZE))

echo "=============================================="
echo "3D Parallel Training Configuration"
echo "=============================================="
echo "Total GPUs:         $NUM_GPUS"
echo "Tensor Parallel:    $TP_SIZE"
echo "Data Parallel:      $DP_SIZE"
echo "Config:             $CONFIG_PATH"
echo "Accelerate Config:  $ACCELERATE_CONFIG"
echo "=============================================="

# Set environment variables for better performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Launch with Accelerate
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --mixed_precision bf16 \
    -m qwen3_omni_pretrain.cli_train_thinker \
    --config "$CONFIG_PATH" \
    --use_accelerator \
    --accelerator_config "$ACCELERATE_CONFIG" \
    --use_tensor_parallel \
    --tp_size "$TP_SIZE" \
    --tensorboard
