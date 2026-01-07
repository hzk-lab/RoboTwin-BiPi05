#!/bin/bash
# TwinVLA1 (Dual VLA + Joint Attention) 评估脚本
# 用法: ./eval.sh task_name task_config train_config_name left_model_name right_model_name seed jax_gpu sapien_gpu

# Dual VLA 需要加载两个模型，建议使用独立的 GPU:
# - JAX (模型推理): 1块大显存GPU
# - SAPIEN (渲染): 1块独立GPU

policy_name=TwinVLA1
task_name=${1:-shake_dual_bottles}
task_config=${2:-demo_clean}
train_config_name=${3:-pi0_base_aloha_robotwin_lora}
left_model_name=${4:-shake_bottle-demo_clean-50_left}
right_model_name=${5:-shake_bottle-demo_clean-50_right}
seed=${6:-0}
jax_gpu=${7:-0}      # JAX 模型推理用的 GPU
sapien_gpu=${8:-1}   # SAPIEN 渲染用的 GPU

# JAX 配置 - 单进程尽量给足显存避免 OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# 简化：单进程只用 1 张 GPU，JAX 与 SAPIEN 共用该卡，避免跨卡混乱
export CUDA_VISIBLE_DEVICES=${jax_gpu}
export SAPIEN_GPU_ID=0
echo -e "\033[33m=================================================="
echo "TwinVLA1 (Dual VLA + Joint Attention) Evaluation"
echo "Task: $task_name"
echo "Task Config: $task_config"
echo "Train Config: $train_config_name"
echo "Left Model: $left_model_name"
echo "Right Model: $right_model_name"
echo "Seed: $seed"
echo "JAX GPU: $jax_gpu (model inference)"
echo "SAPIEN GPU: $sapien_gpu (rendering)"
echo -e "==================================================\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --left_model_name ${left_model_name} \
    --right_model_name ${right_model_name} \
    --ckpt_setting ${left_model_name}_${right_model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}
