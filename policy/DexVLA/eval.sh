#!/bin/bash

policy_name=DexVLA
task_name=place_empty_cup
task_config=demo_clean
ckpt_setting=0
seed=0
gpu_id=0
expert_check=False
eval_video_log=True
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root
export PYTHONPATH=$PWD/policy/DexVLA:$PWD:$PYTHONPATH

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --expert_check ${expert_check} \
    --eval_video_log ${eval_video_log}
    # [TODO] add parameters here
