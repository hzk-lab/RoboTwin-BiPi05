#!/bin/bash
# TwinVLA1 8-GPU 并行 Finetune 脚本
# 同时训练多个任务的左右臂模型
#
# 用法:
#   ./finetune_8gpu.sh <train_config_name> <model_name>
#
# 示例:
#   ./finetune_8gpu.sh pi0_base_aloha_robotwin_lora shake_bottle-demo_clean-50
#
# GPU 分配:
#   - GPU 0: 左臂训练
#   - GPU 1: 右臂训练
#   (或者使用更多GPU做数据并行)

train_config_name=${1:-pi0_base_aloha_robotwin_lora}
model_name=${2:-shake_bottle-demo_clean-50}

echo "=============================================="
echo "TwinVLA1 8-GPU Finetuning"
echo "Config: $train_config_name"
echo "Model: $model_name"
echo "=============================================="

# 切换到 TwinVLA1 目录
cd "$(dirname "$0")"

# 方案1: 左右臂各用1个GPU，顺序训练更稳定
# 方案2: 左右臂各用4个GPU做数据并行 (fsdp_devices=4)

echo ""
echo "选择训练模式:"
echo "  1) 快速模式: 左右臂各用1 GPU 并行训练 (GPU 0,1)"
echo "  2) 大batch模式: 左右臂各用4 GPU 数据并行 (GPU 0-3 左臂, 4-7 右臂)"
echo ""
read -p "请选择 [1/2] (默认1): " mode
mode=${mode:-1}

if [ "$mode" == "2" ]; then
    echo ""
    echo "[模式2] 大batch数据并行训练"
    echo "=============================================="
    
    # 左臂: GPU 0,1,2,3 (4路数据并行)
    echo "[Left Arm] Starting on GPU 0,1,2,3..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name \
        --exp-name=$model_name \
        --arm=left \
        --overwrite &
    left_pid=$!
    
    # 右臂: GPU 4,5,6,7 (4路数据并行)
    echo "[Right Arm] Starting on GPU 4,5,6,7..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name \
        --exp-name=$model_name \
        --arm=right \
        --overwrite &
    right_pid=$!
    
else
    echo ""
    echo "[模式1] 快速并行训练"
    echo "=============================================="
    
    # 左臂: GPU 0
    echo "[Left Arm] Starting on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name \
        --exp-name=$model_name \
        --arm=left \
        --overwrite &
    left_pid=$!
    
    # 右臂: GPU 1
    echo "[Right Arm] Starting on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name \
        --exp-name=$model_name \
        --arm=right \
        --overwrite &
    right_pid=$!
fi

# 等待训练完成
echo ""
echo "Training started!"
echo "Left PID: $left_pid, Right PID: $right_pid"
echo ""
echo "监控训练进度:"
echo "  tail -f wandb/latest-run/logs/*.log"
echo ""

wait $left_pid
left_status=$?
echo "[Left Arm] Training completed with status: $left_status"

wait $right_pid
right_status=$?
echo "[Right Arm] Training completed with status: $right_status"

echo ""
echo "=============================================="
echo "TwinVLA1 Finetuning Completed!"
echo "Left checkpoint: checkpoints/${train_config_name}/${model_name}_left"
echo "Right checkpoint: checkpoints/${train_config_name}/${model_name}_right"
echo "=============================================="

