#!/bin/bash
# TwinVLA1 8-GPU 并行评测脚本
# 用法: ./eval_multi_gpu.sh task_name task_config train_config left_model right_model start_seed num_seeds
# 示例: ./eval_multi_gpu.sh shake_dual_bottles demo_clean pi0_base_aloha_robotwin_lora \
#         shake_bottle-demo_clean-50_left shake_bottle-demo_clean-50_right 0 8

policy_name=TwinVLA1
task_name=${1:-shake_dual_bottles}
task_config=${2:-demo_clean}
train_config_name=${3:-pi0_base_aloha_robotwin_lora}
left_model_name=${4:-shake_bottle-demo_clean-50_left}
right_model_name=${5:-shake_bottle-demo_clean-50_right}
start_seed=${6:-0}
num_seeds=${7:-8}

# GPU 列表 (0-7)，按种子轮转使用；sapien GPU 也轮转，但避免与 JAX GPU 相同
GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}

echo "=============================================="
echo "TwinVLA1 Multi-GPU Evaluation"
echo "Task: $task_name / $task_config"
echo "Train Config: $train_config_name"
echo "Left: $left_model_name | Right: $right_model_name"
echo "Seeds: $start_seed .. $((start_seed + num_seeds - 1))"
echo "GPUs: ${GPU_LIST[*]} (JAX only; SAPIEN 取下一张卡)"
echo "=============================================="

cd ../..

pids=()
for ((i=0; i<num_seeds; i++)); do
  seed=$((start_seed + i))
  jax_gpu=${GPU_LIST[$((i % NUM_GPUS))]}
  # SAPIEN GPU 与 JAX GPU 解耦：偏移 4 张卡，避免冲突
  sapien_gpu=${GPU_LIST[$(((i + 4) % NUM_GPUS))]}

  echo "[Seed $seed] JAX GPU: $jax_gpu | SAPIEN GPU: $sapien_gpu"
  # 简化：每个进程只用 1 张 GPU，JAX 和 SAPIEN 共用该卡，避免跨卡混乱
  CUDA_VISIBLE_DEVICES=${jax_gpu} \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  SAPIEN_GPU_ID=0 \
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

done

# 等待全部结束
for pid in "${pids[@]}"; do
  wait $pid
done

echo "=============================================="
echo "All evaluations finished."
echo "=============================================="
