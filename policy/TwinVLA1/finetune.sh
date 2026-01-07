#!/bin/bash
# TwinVLA1 Finetune 脚本
# 分别训练左臂VLA和右臂VLA
#
# 用法:
#   ./finetune.sh <train_config_name> <model_name> <gpu_id>
#
# 示例:
#   ./finetune.sh pi0_base_aloha_robotwin_lora shake_bottle-demo_clean-50 0
#   
# 这会同时启动左臂和右臂的训练 (如果有足够GPU)，或顺序训练

train_config_name=${1:-pi0_base_aloha_robotwin_lora}
model_name=${2:-shake_bottle-demo_clean-50}
gpu_use=${3:-0}

export CUDA_VISIBLE_DEVICES=$gpu_use
echo "=============================================="
echo "TwinVLA1 Finetuning"
echo "Config: $train_config_name"
echo "Model: $model_name"
echo "GPU: $gpu_use"
echo "=============================================="

# 切换到 TwinVLA1 目录
cd "$(dirname "$0")"

# 训练左臂VLA
echo ""
echo "[1/2] Training Left Arm VLA..."
echo "=============================================="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_dual_vla.py \
    --config-name=$train_config_name \
    --exp-name=$model_name \
    --arm=left \
    --overwrite

# 训练右臂VLA
echo ""
echo "[2/2] Training Right Arm VLA..."
echo "=============================================="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_dual_vla.py \
    --config-name=$train_config_name \
    --exp-name=$model_name \
    --arm=right \
    --overwrite

echo ""
echo "=============================================="
echo "TwinVLA1 Finetuning Completed!"
echo "Left checkpoint: checkpoints/${train_config_name}/${model_name}_left"
echo "Right checkpoint: checkpoints/${train_config_name}/${model_name}_right"
echo "=============================================="

