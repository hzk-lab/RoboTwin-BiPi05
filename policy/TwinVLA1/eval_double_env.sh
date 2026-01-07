#!/bin/bash
# TwinVLA1 (Dual VLA + Joint Attention) 双环境评估脚本
# 用法: ./eval_double_env.sh task_name task_config train_config_name left_model_name right_model_name start_seed num_seeds gpu_pairs

policy_name=TwinVLA1
task_name=${1:-shake_dual_bottles}
task_config=${2:-demo_clean}
train_config_name=${3:-pi0_base_aloha_robotwin_lora}
left_model_name=${4:-shake_bottle-demo_clean-50}
right_model_name=${5:-shake_bottle-demo_clean-50}
start_seed=${6:-0}
num_seeds=${7:-4}
gpu_pairs=${8:-"0,1 2,3"}  # GPU pairs for parallel evaluation

# JAX 配置
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo -e "\033[33m=================================================="
echo "TwinVLA1 (Dual VLA) Parallel Evaluation"
echo "Task: $task_name"
echo "Task Config: $task_config"
echo "Train Config: $train_config_name"
echo "Left Model: $left_model_name"
echo "Right Model: $right_model_name"
echo "Seeds: $start_seed to $((start_seed + num_seeds - 1))"
echo "GPU Pairs: $gpu_pairs"
echo -e "==================================================\033[0m"

cd ../..

# 将 GPU pairs 转换为数组
IFS=' ' read -ra GPU_PAIRS <<< "$gpu_pairs"
num_pairs=${#GPU_PAIRS[@]}

# 启动并行评估
pids=()
for ((i=0; i<num_seeds; i++)); do
    seed=$((start_seed + i))
    pair_idx=$((i % num_pairs))
    gpu_pair=${GPU_PAIRS[$pair_idx]}
    
    # 设置 SAPIEN GPU (使用 pair 中的第二个 GPU)
    IFS=',' read -ra GPUS <<< "$gpu_pair"
    export SAPIEN_GPU_ID=${GPUS[1]}
    
    echo -e "\033[32m[Seed $seed] Starting on GPU $gpu_pair (SAPIEN: ${GPUS[1]})\033[0m"
    
    CUDA_VISIBLE_DEVICES=$gpu_pair \
PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --left_model_name ${left_model_name} \
        --right_model_name ${right_model_name} \
        --ckpt_setting ${left_model_name}_${right_model_name}_seed${seed} \
    --seed ${seed} \
        --policy_name ${policy_name} &
    
    pids+=($!)
    
    # 每 num_pairs 个任务等待一组完成
    if (( (i + 1) % num_pairs == 0 && i + 1 < num_seeds )); then
        echo -e "\033[33mWaiting for current batch to complete...\033[0m"
        for pid in "${pids[@]}"; do
            wait $pid
        done
        pids=()
    fi
done

# 等待剩余任务完成
echo -e "\033[33mWaiting for remaining tasks to complete...\033[0m"
for pid in "${pids[@]}"; do
    wait $pid
done

echo -e "\033[32m=================================================="
echo "All evaluations completed!"
echo -e "==================================================\033[0m"
