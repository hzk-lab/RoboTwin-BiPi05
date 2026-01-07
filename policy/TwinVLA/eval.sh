#!/bin/bash
# TwinVLA 评估脚本
# 用法: ./eval.sh task_name task_config train_config_name model_name seed gpu_id

# JAX 用 GPU 0，渲染器用 GPU 1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 # TwinPi0 需要更多显存 (2H tokens)
export SAPIEN_GPU_ID=1  # 渲染器使用 GPU 1

policy_name=TwinVLA
task_name=${1:-shake_dual_bottles}
task_config=${2:-demo_clean}
train_config_name=${3:-pi0_base_aloha_robotwin_lora}
model_name=${4:-shake_bottle-demo_clean-50}
seed=${5:-0}
gpu_id=${6:-0,1}  # 默认使用 GPU 0,1

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m=================================================="
echo "TwinVLA Evaluation"
echo "Task: $task_name"
echo "Task Config: $task_config"
echo "Train Config: $train_config_name"
echo "Model: $model_name"
echo "GPU: $gpu_id"
echo -e "==================================================\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}
