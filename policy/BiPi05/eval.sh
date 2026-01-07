#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=BiPi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

# 根据 gpu_id 判断是 GPU 还是 CPU 模式
if [[ "${gpu_id}" =~ ^[0-9,]+$ ]]; then
    # 正常 GPU 模式
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
else
    # 非数字（例如 cpu），走 CPU 模式：不设置 CUDA_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    echo -e "\033[33mUse CPU (no CUDA_VISIBLE_DEVICES), gpu_id arg: ${gpu_id}\033[0m"
fi

# source .venv/bin/activate
cd ../.. # move to root

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
