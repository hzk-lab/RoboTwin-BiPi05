#!/bin/bash
# TwinVLA1 并行 Finetune 脚本
# 同时在不同GPU上训练左臂VLA和右臂VLA
#
# 用法:
#   ./finetune_parallel.sh <train_config_name> <model_name> <left_gpu> <right_gpu>
#
# 示例:
#   ./finetune_parallel.sh pi0_base_aloha_robotwin_lora shake_bottle-demo_clean-50 0 1

train_config_name=${1:-pi0_base_aloha_robotwin_lora}
model_name=${2:-shake_bottle-demo_clean-50}
left_gpu=${3:-0}
right_gpu=${4:-1}

echo "=============================================="
echo "TwinVLA1 Parallel Finetuning"
echo "Config: $train_config_name"
echo "Model: $model_name"
echo "Left GPU: $left_gpu"
echo "Right GPU: $right_gpu"
echo "=============================================="

# 切换到 TwinVLA1 目录
cd "$(dirname "$0")"

# 并行训练左右臂VLA
echo ""
echo "Starting parallel training..."

# 启动左臂训练
echo "[Left Arm] Starting on GPU $left_gpu..."
CUDA_VISIBLE_DEVICES=$left_gpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python scripts/train_dual_vla.py \
    --config-name=$train_config_name \
    --exp-name=$model_name \
    --arm=left \
    --overwrite &
left_pid=$!

# 启动右臂训练
echo "[Right Arm] Starting on GPU $right_gpu..."
CUDA_VISIBLE_DEVICES=$right_gpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python scripts/train_dual_vla.py \
    --config-name=$train_config_name \
    --exp-name=$model_name \
    --arm=right \
    --overwrite &
right_pid=$!

# 等待两个任务完成
echo ""
echo "Waiting for both training jobs to complete..."
echo "Left PID: $left_pid, Right PID: $right_pid"

wait $left_pid
left_status=$?
echo "[Left Arm] Training completed with status: $left_status"

wait $right_pid
right_status=$?
echo "[Right Arm] Training completed with status: $right_status"

echo ""
echo "=============================================="
echo "TwinVLA1 Parallel Finetuning Completed!"
echo "Left checkpoint: checkpoints/${train_config_name}/${model_name}_left"
echo "Right checkpoint: checkpoints/${train_config_name}/${model_name}_right"
echo "=============================================="

