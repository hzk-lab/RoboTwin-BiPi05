#!/bin/bash
# TwinVLA1 多任务并行训练脚本
# 同时训练多个任务，充分利用8块GPU
#
# 用法:
#   ./finetune_multi_task.sh <train_config_name> <task1> <task2> <task3> <task4>
#
# 示例 (同时训练4个任务，每个任务用2个GPU):
#   ./finetune_multi_task.sh pi0_base_aloha_robotwin_lora \
#       shake_bottle-demo_clean-50 \
#       handover_block-demo_clean-50

train_config_name=${1:-pi0_base_aloha_robotwin_lora}
task1=${2:-shake_bottle-demo_clean-50}
task2=${3:-}
task3=${4:-}
task4=${5:-}

echo "=============================================="
echo "TwinVLA1 Multi-Task 8-GPU Finetuning"
echo "Config: $train_config_name"
echo "Tasks: $task1 $task2 $task3 $task4"
echo "=============================================="

cd "$(dirname "$0")"

pids=()

# Task 1: GPU 0 (左臂), GPU 1 (右臂)
if [ -n "$task1" ]; then
    echo "[Task 1: $task1] Starting on GPU 0,1..."
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task1 --arm=left --overwrite &
    pids+=($!)
    
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task1 --arm=right --overwrite &
    pids+=($!)
fi

# Task 2: GPU 2 (左臂), GPU 3 (右臂)
if [ -n "$task2" ]; then
    echo "[Task 2: $task2] Starting on GPU 2,3..."
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task2 --arm=left --overwrite &
    pids+=($!)
    
    CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task2 --arm=right --overwrite &
    pids+=($!)
fi

# Task 3: GPU 4 (左臂), GPU 5 (右臂)
if [ -n "$task3" ]; then
    echo "[Task 3: $task3] Starting on GPU 4,5..."
    CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task3 --arm=left --overwrite &
    pids+=($!)
    
    CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task3 --arm=right --overwrite &
    pids+=($!)
fi

# Task 4: GPU 6 (左臂), GPU 7 (右臂)
if [ -n "$task4" ]; then
    echo "[Task 4: $task4] Starting on GPU 6,7..."
    CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task4 --arm=left --overwrite &
    pids+=($!)
    
    CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/train_dual_vla.py \
        --config-name=$train_config_name --exp-name=$task4 --arm=right --overwrite &
    pids+=($!)
fi

echo ""
echo "Started ${#pids[@]} training jobs"
echo "PIDs: ${pids[@]}"
echo ""

# 等待所有任务完成
for pid in "${pids[@]}"; do
    wait $pid
    echo "Job $pid completed with status: $?"
done

echo ""
echo "=============================================="
echo "All training jobs completed!"
echo "=============================================="

