#!/bin/bash
# TwinVLA 并行评估脚本
# 4 个进程，每个用 2 张 GPU (JAX + Renderer)
# 总共使用 8 张 GPU

task_name=${1:-shake_dual_bottles}
task_config=${2:-demo_clean}
train_config_name=${3:-pi0_base_aloha_robotwin_lora}
model_name=${4:-shake_bottle-demo_clean-50}

policy_name=TwinVLA

echo -e "\033[33m=================================================="
echo "TwinVLA Parallel Evaluation (4 processes, 8 GPUs)"
echo "Task: $task_name"
echo "Task Config: $task_config"
echo "Train Config: $train_config_name"
echo "Model: $model_name"
echo -e "==================================================\033[0m"

cd ../..

# 进程 0: GPU 0,1, seed 0
echo "[Process 0] Starting on GPU 0,1 with seed 0..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 SAPIEN_GPU_ID=1 CUDA_VISIBLE_DEVICES=0,1 \
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_seed0 \
    --seed 0 \
    --policy_name ${policy_name} &
pid0=$!

sleep 60  # 等待 JIT 编译

# 进程 1: GPU 2,3, seed 1
echo "[Process 1] Starting on GPU 2,3 with seed 1..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 SAPIEN_GPU_ID=1 CUDA_VISIBLE_DEVICES=2,3 \
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_seed1 \
    --seed 1 \
    --policy_name ${policy_name} &
pid1=$!

sleep 60

# 进程 2: GPU 4,5, seed 2
echo "[Process 2] Starting on GPU 4,5 with seed 2..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 SAPIEN_GPU_ID=1 CUDA_VISIBLE_DEVICES=4,5 \
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_seed2 \
    --seed 2 \
    --policy_name ${policy_name} &
pid2=$!

sleep 60

# 进程 3: GPU 6,7, seed 3
echo "[Process 3] Starting on GPU 6,7 with seed 3..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 SAPIEN_GPU_ID=1 CUDA_VISIBLE_DEVICES=6,7 \
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_seed3 \
    --seed 3 \
    --policy_name ${policy_name} &
pid3=$!

echo ""
echo "All 4 processes launched!"
echo "PIDs: $pid0 $pid1 $pid2 $pid3"
echo ""
echo "Waiting for all processes to complete..."

wait $pid0 $pid1 $pid2 $pid3

echo "All processes completed!"
